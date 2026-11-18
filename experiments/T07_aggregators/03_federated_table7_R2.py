# -*- coding: utf-8 -*-
"""
Table VII (R2 revision): aggregator comparison, HAFSSL vs FL-AECNN.
4 aggregators (FedAvg, FedProx, FedNova, MOON) x 2 models x 3 seeds = 24 runs.

Fixed setting (pre-registered, do not change after seeing results):
  ISCX-VPN2016, duplicates removed, frozen split (SPLIT_SEED 2026, same test set as all other runs)
  Dirichlet alpha 0.5, K = 10 clients, 50 rounds, 5 local epochs per stage, batch 256, Adam 1e-3
  1,000 labeled flows per class in the WHOLE federation; aggregation weights n_k / n
  FedProx mu = 0.1 (both stages) | MOON mu = 5, tau = 0.5 (classifier stage) | FedNova: normalized averaging

Identical in all 24 runs: the architecture of each model, the data split, the client partition and
labeled subset of each seed, the local training schedule, the optimizer and the evaluation.
Only the aggregator changes. HAFSSL and FL-AECNN differ only in the encoder attention block.

This is the 03_federated_hafssl_R2.py pipeline with the "table7" preset fixed. Training code is
unchanged, so the HAFSSL + FedAvg runs are bit-identical to the MHA->SE row reused by the ablation.
Changes compared with 03_federated_hafssl_R2.py:
  1. FedNova: each parameter group is normalized by the number of local steps that actually updated
     it (encoder: AE + classifier steps; decoder: AE steps; classifier head: classifier steps).
     Clients with zero steps for a group add nothing to that group. Result cast to float32.
  2. Extra outputs: per-class test recall columns, collapse flag (any class recall < 10%), mean of
     the last 10 rounds (validation), communication to reach 90% (validation), per-round validation
     per-class recall and mean local losses.
  3. Client partition and labeled indices saved per seed, and checked to be identical across all
     runs of that seed (proves the paired design).
  4. MODELS / AGGREGATORS lists, so the 24 runs can be split over several notebooks.
  5. Paired statistics at the end (McNemar per seed, paired t-test over seeds, Holm correction).
     Set ONLY_SUMMARIZE = True and list several output folders in SUMMARY_DIRS to combine notebooks.

Communication = exact bytes of the float32 weights sent (encoder + decoder + head, up + down, all
clients). FedNova also uploads 3 step counts per client (12 bytes). MOON keeps the previous local
encoder on the client (no extra communication). FedProx sends nothing extra.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import json, time, shutil, glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             balanced_accuracy_score, confusion_matrix, classification_report)
from scipy import stats

# =============================== CONFIG ===============================
DATA_PATH = "data/ISCX-VPN2016/ISCX_5class_each_normalized_cuttedfloefeature.csv"
DATASET   = "ISCX"
LABELS    = ['chat', 'email', 'file', 'streaming', 'voip']

# Which runs this notebook does. Split the 24 runs over notebooks by editing these two lists,
# and give every notebook its own OUT_DIR.
MODELS      = ["HAFSSL", "FL-AECNN"]
AGGREGATORS = ["FedAvg", "FedProx", "FedNova", "MOON"]
SEEDS       = [0, 1, 2]
OUT_DIR     = "/kaggle/working/FED_ISCX_table7"

# Fixed Table VII setting (do not change)
ALPHA             = 0.5
NUM_CLIENTS       = 10
ROUNDS            = 50
LOCAL_EPOCHS      = 5
BATCH             = 256
LR                = 1e-3
LABELED_PER_CLASS = 1000
FEDPROX_MU        = 0.1
MOON_MU, MOON_TAU = 5.0, 0.5
COLLAPSE_RECALL   = 0.10      # dagger in Table VII if any class recall is below this in any seed

SPLIT_SEED        = 2026      # same as every other R2 run = same test set
SPLIT_FROM        = ""        # optional: splits/<DATASET>_split.npz from an earlier run
RESUME_FROM       = ""        # Kaggle: previous version's output folder, to continue after a stop
TIME_BUDGET_HOURS = 11.0      # stops cleanly before Kaggle's 12-hour limit
QUICK_TEST        = False     # True = 2 rounds, 1 local epoch, seed 0 (a few minutes)

ONLY_SUMMARIZE = False        # True = no training; combine results from SUMMARY_DIRS
SUMMARY_DIRS   = []           # e.g. ["/kaggle/input/t7-hafssl/FED_ISCX_table7", "/kaggle/input/t7-flaecnn/..."]

# Optional overrides from environment variables (used by run_table7_2gpu.py, one process per GPU).
# When the variables are not set, the values above are used.
MODELS         = json.loads(os.environ.get("T7_MODELS", json.dumps(MODELS)))
AGGREGATORS    = json.loads(os.environ.get("T7_AGGREGATORS", json.dumps(AGGREGATORS)))
DATA_PATH      = os.environ.get("T7_DATA_PATH", DATA_PATH)
OUT_DIR        = os.environ.get("T7_OUT_DIR", OUT_DIR)
RESUME_FROM    = os.environ.get("T7_RESUME_FROM", RESUME_FROM)
QUICK_TEST     = os.environ.get("T7_QUICK_TEST", str(QUICK_TEST)) == "True"
ONLY_SUMMARIZE = os.environ.get("T7_ONLY_SUMMARIZE", str(ONLY_SUMMARIZE)) == "True"
SUMMARY_DIRS   = json.loads(os.environ.get("T7_SUMMARY_DIRS", json.dumps(SUMMARY_DIRS)))

if QUICK_TEST:
    ROUNDS, LOCAL_EPOCHS, SEEDS = 2, 1, [0]
    OUT_DIR = OUT_DIR + "_quicktest"

MODEL_TO_VARIANT = {"HAFSSL": "MHA->SE", "FL-AECNN": "none"}
MODEL_NAMES = {"MHA->SE": "HAFSSL", "none": "FL-AECNN"}     # same run_id names as the general script
LATENT_DIM, NUM_HEADS, KEY_DIM, SE_RATIO = 39, 4, 32, 16
SAFE_LABELS = [l.replace(" ", "_") for l in LABELS]
SESSION_START = time.time()
_rounds_this_session = 0


# =============================== SETUP ===============================
def platform_name():
    if os.path.exists("/kaggle"):
        return "kaggle"
    return "colab" if os.path.exists("/content") else "pc"


def gpu_name():
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        return "CPU only"
    return tf.config.experimental.get_device_details(gpus[0]).get('device_name', 'GPU')


def restore_previous_output():
    if not RESUME_FROM:
        return
    if not os.path.isdir(RESUME_FROM):
        raise FileNotFoundError(f"RESUME_FROM folder not found: {RESUME_FROM}")
    if os.path.isdir(OUT_DIR) and os.listdir(OUT_DIR):
        print("OUT_DIR is not empty, so RESUME_FROM is not copied:", OUT_DIR)
        return
    shutil.copytree(RESUME_FROM, OUT_DIR, dirs_exist_ok=True)
    print("Copied previous progress from", RESUME_FROM)


def save_config():
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = dict(script="03_federated_table7_R2.py", dataset=DATASET, data_path=DATA_PATH, labels=LABELS,
               models=MODELS, aggregators=AGGREGATORS, seeds=SEEDS, alpha=ALPHA, clients=NUM_CLIENTS,
               rounds=ROUNDS, local_epochs=LOCAL_EPOCHS, batch=BATCH, lr=LR,
               labeled_per_class=LABELED_PER_CLASS, fedprox_mu=FEDPROX_MU, moon_mu=MOON_MU,
               moon_tau=MOON_TAU, split_seed=SPLIT_SEED, latent_dim=LATENT_DIM, num_heads=NUM_HEADS,
               key_dim=KEY_DIM, se_ratio=SE_RATIO, quick_test=QUICK_TEST, tf_version=tf.__version__)
    with open(os.path.join(OUT_DIR, "config_table7.json"), "w") as f:
        json.dump(cfg, f, indent=2)


# =============================== DATA (same as all R2 scripts) ===============================
def load_data():
    df = pd.read_csv(DATA_PATH)
    feat_cols = [c for c in df.columns if c != 'label']
    n_before = len(df)
    df = df.drop_duplicates(subset=feat_cols).reset_index(drop=True)
    print(f"Rows: {n_before} -> {len(df)} after removing {n_before - len(df)} exact duplicates")
    X = df[feat_cols].values.astype('float32').reshape(len(df), len(feat_cols), 1)
    y = df['label'].values.astype(int)
    return X, y, len(feat_cols)


def get_split(y):
    path = os.path.join(OUT_DIR, "splits", f"{DATASET}_split.npz")
    if SPLIT_FROM and not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        shutil.copy(SPLIT_FROM, path)
    if os.path.exists(path):
        s = np.load(path)
        assert int(s['n_rows']) == len(y), "Split file does not match this data."
        return s['train'], s['val'], s['test']
    idx = np.arange(len(y))
    trval, test = train_test_split(idx, test_size=0.10, stratify=y, random_state=SPLIT_SEED)
    train, val = train_test_split(trval, test_size=0.10, stratify=y[trval], random_state=SPLIT_SEED)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, train=train, val=val, test=test, n_rows=len(y))
    return train, val, test


def sample_labeled(y_train, per_class, seed):
    """Same function as the centralized runs: same seed and budget = same labeled flows."""
    rng = np.random.default_rng(seed)
    chosen = []
    for c in np.unique(y_train):
        pool = np.where(y_train == c)[0]
        chosen.extend(rng.choice(pool, min(per_class, len(pool)), replace=False))
    return np.array(chosen)


def partition_clients(y_train, kind, alpha, seed, n_clients, min_size=10):
    """Identical to 03_federated_hafssl_R2.py (same seed = same partition)."""
    rng = np.random.default_rng(10_000 + seed)
    n = len(y_train)
    if kind == "iid":
        return [np.sort(p) for p in np.array_split(rng.permutation(n), n_clients)]
    best = None
    for _ in range(2000):
        parts = [[] for _ in range(n_clients)]
        for c in np.unique(y_train):
            idx_c = rng.permutation(np.where(y_train == c)[0])
            p_raw = rng.dirichlet(alpha * np.ones(n_clients))
            if not np.all(np.isfinite(p_raw)) or p_raw.sum() <= 0:
                p_raw = np.eye(n_clients)[rng.integers(n_clients)]
            p = p_raw * np.array([len(parts[k]) < n / n_clients for k in range(n_clients)])
            p = p / p.sum() if p.sum() > 0 else p_raw / p_raw.sum()
            cuts = (np.cumsum(p) * len(idx_c)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(idx_c, cuts)):
                parts[k].extend(chunk.tolist())
        smallest = min(len(pt) for pt in parts)
        if best is None or smallest > min(len(pt) for pt in best):
            best = parts
        if smallest >= min_size:
            break
    parts = [list(pt) for pt in best]
    for k in range(n_clients):
        while len(parts[k]) < min_size:
            big = int(np.argmax([len(pt) for pt in parts]))
            parts[k].append(parts[big].pop(int(rng.integers(len(parts[big])))))
    return [np.sort(np.array(pt)) for pt in parts]


def save_or_check_partition(parts, labeled_idx, seed):
    """Indices refer to positions in the training split. Every run with this seed must use exactly
    these clients and labeled flows; a mismatch stops the script."""
    path = os.path.join(OUT_DIR, "splits",
                        f"{DATASET}_partition_a{ALPHA}_K{NUM_CLIENTS}_L{LABELED_PER_CLASS}_s{seed}.npz")
    arrays = {f"client_{k}": p for k, p in enumerate(parts)}
    arrays["labeled"] = np.sort(labeled_idx)
    if os.path.exists(path):
        old = np.load(path)
        for key, arr in arrays.items():
            if not np.array_equal(old[key], arr):
                raise RuntimeError(f"Partition for seed {seed} differs from the saved one ({key}). "
                                   "The paired design is broken; check the data file and split.")
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **arrays)


# =============================== MODEL (unchanged) ===============================
def se_block(x, ratio=SE_RATIO):
    ch = x.shape[-1]
    s = layers.GlobalAveragePooling1D()(x)
    s = layers.Dense(ch // ratio, activation='relu')(s)
    s = layers.Dense(ch, activation='sigmoid')(s)
    return layers.Multiply()([x, layers.Reshape((1, ch))(s)])


def attention_block(x, variant):
    if variant == "none":
        return x
    if variant != "MHA->SE":
        raise ValueError(variant)
    a = se_block(layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=KEY_DIM, dropout=0.1)(x, x))
    x = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, a]))
    f = layers.Dense(KEY_DIM * 4, activation='relu')(x)
    f = layers.Dense(x.shape[-1])(layers.Dropout(0.1)(f))
    return layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, f]))


def build_models(n_features, n_classes, variant):
    ie = layers.Input(shape=(n_features, 1))
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(ie)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D()(x)
    x = attention_block(x, variant)
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(x)
    steps, ch = x.shape[1], x.shape[2]
    enc = Model(ie, layers.Dense(LATENT_DIM, activation='relu')(layers.Flatten()(x)), name='encoder')

    idd = layers.Input(shape=(LATENT_DIM,))
    d = layers.Reshape((steps, ch))(layers.Dense(steps * ch)(idd))
    d = layers.Conv1D(32, 3, padding='same', activation='relu')(d)
    d = layers.Conv1D(32, 3, padding='same', activation='relu')(d)
    d = layers.UpSampling1D()(d)
    d = layers.Conv1D(64, 3, padding='same', activation='relu')(d)
    dec = Model(idd, layers.Conv1D(1, 3, padding='same', activation='relu')(d), name='decoder')

    ic = layers.Input(shape=(LATENT_DIM,))
    c = layers.Reshape((steps, ch))(layers.Dense(steps * ch)(ic))
    c = layers.Conv1D(64, 3, padding='same', activation='relu')(c)
    c = layers.Conv1D(64, 3, padding='same', activation='relu')(c)
    c = layers.MaxPooling1D(2)(c)
    c = layers.Conv1D(128, 3, padding='same', activation='relu')(c)
    c = layers.Conv1D(128, 3, padding='same', activation='relu')(c)
    c = layers.MaxPooling1D(2)(c)
    c = layers.Dense(128, activation='relu')(layers.Flatten()(c))
    head = Model(ic, layers.Dense(n_classes, activation='softmax')(c), name='classifier_head')

    inp = layers.Input(shape=(n_features, 1))
    ae = Model(inp, dec(enc(inp)))
    clf = Model(inp, head(enc(inp)))
    return enc, dec, head, ae, clf


class ModelSet:
    """Encoder, decoder and head, sent and aggregated as one list of weights."""
    def __init__(self, n_features, n_classes, variant):
        self.enc, self.dec, self.head, self.ae, self.clf = build_models(n_features, n_classes, variant)
        self.parts = [self.enc, self.dec, self.head]
        self.sizes = [len(m.get_weights()) for m in self.parts]
        self.groups = [g for g, s in zip(("enc", "dec", "head"), self.sizes) for _ in range(s)]

    def get_w(self):
        return [w for m in self.parts for w in m.get_weights()]

    def set_w(self, w):
        i = 0
        for m, s in zip(self.parts, self.sizes):
            m.set_weights(w[i:i + s])
            i += s

    def n_params(self):
        return int(sum(m.count_params() for m in self.parts))

    def all_vars(self):
        return [v for m in self.parts for v in m.variables]


# =============================== LOCAL TRAINING (unchanged except return values) ===============================
def reset_optimizer(opt):
    for v in opt.variables:
        if "learning_rate" not in v.path:
            v.assign(tf.zeros_like(v))


def make_trainer(local, glob_ms, prev_enc, aggregator):
    prox, moon = aggregator == "FedProx", aggregator == "MOON"
    ae_vars, clf_vars = local.ae.trainable_variables, local.clf.trainable_variables
    g_ae_vars, g_clf_vars = glob_ms.ae.trainable_variables, glob_ms.clf.trainable_variables
    opt_ae, opt_clf = tf.keras.optimizers.Adam(LR), tf.keras.optimizers.Adam(LR)
    opt_ae.build(ae_vars)
    opt_clf.build(clf_vars)
    cce = tf.keras.losses.CategoricalCrossentropy()

    def prox_term(vs, gs):
        return FEDPROX_MU / 2.0 * tf.add_n([tf.reduce_sum(tf.square(v - g)) for v, g in zip(vs, gs)])

    @tf.function(reduce_retracing=True)
    def ae_step(x):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(x - local.ae(x, training=True)))
            if prox:
                loss += prox_term(ae_vars, g_ae_vars)
        opt_ae.apply_gradients(zip(tape.gradient(loss, ae_vars), ae_vars))
        return loss

    @tf.function(reduce_retracing=True)
    def clf_step(x, y):
        with tf.GradientTape() as tape:
            z = local.enc(x, training=True)
            loss = cce(y, local.head(z, training=True))
            if prox:
                loss += prox_term(clf_vars, g_clf_vars)
            if moon:
                zn = tf.math.l2_normalize(z, axis=1)
                zg = tf.math.l2_normalize(glob_ms.enc(x, training=False), axis=1)
                zp = tf.math.l2_normalize(prev_enc(x, training=False), axis=1)
                sg = tf.reduce_sum(zn * zg, axis=1) / MOON_TAU
                sp = tf.reduce_sum(zn * zp, axis=1) / MOON_TAU
                loss += MOON_MU * tf.reduce_mean(tf.reduce_logsumexp(tf.stack([sg, sp], axis=1), axis=1) - sg)
        opt_clf.apply_gradients(zip(tape.gradient(loss, clf_vars), clf_vars))
        return loss

    def train_client(x_all, x_lab, y_lab, rng):
        """Stage 1: AE on all local flows. Stage 2: encoder + head on the local labeled flows.
        Returns step counts per parameter group and the mean local losses (logging only)."""
        reset_optimizer(opt_ae)
        reset_optimizer(opt_clf)
        ae_steps = clf_steps = 0
        ae_loss, clf_loss = tf.constant(0.0), tf.constant(0.0)
        for _ in range(LOCAL_EPOCHS):
            order = rng.permutation(len(x_all))
            for i in range(0, len(order), BATCH):
                ae_loss += ae_step(tf.constant(x_all[order[i:i + BATCH]]))
                ae_steps += 1
        if len(x_lab) > 0:
            for _ in range(LOCAL_EPOCHS):
                order = rng.permutation(len(x_lab))
                for i in range(0, len(order), BATCH):
                    clf_loss += clf_step(tf.constant(x_lab[order[i:i + BATCH]]),
                                         tf.constant(y_lab[order[i:i + BATCH]]))
                    clf_steps += 1
        steps = {"enc": ae_steps + clf_steps, "dec": ae_steps, "head": clf_steps}
        return steps, float(ae_loss) / max(ae_steps, 1), float(clf_loss) / max(clf_steps, 1)

    return train_client


def aggregate(global_w, client_ws, n_k, steps_k, aggregator, groups):
    """FedAvg / FedProx / MOON: w = sum_k p_k w_k, p_k = n_k / n.
    FedNova (Wang et al., 2020), per parameter group g:
        d_k = (w_glob - w_k) / tau_k,g ;  tau_eff = sum_k p_k tau_k,g ;  w = w_glob - tau_eff * sum_k p_k d_k
    where tau_k,g = local steps that updated group g on client k (clients with tau = 0 add nothing)."""
    p = np.asarray(n_k, dtype=np.float64) / np.sum(n_k)
    if aggregator != "FedNova":
        return [sum(pk * w[i] for pk, w in zip(p, client_ws)).astype(np.float32) for i in range(len(global_w))]
    new_w = []
    for i, g in enumerate(global_w):
        g64 = g.astype(np.float64)
        tau = np.array([s[groups[i]] for s in steps_k], dtype=np.float64)
        tau_eff = float(np.sum(p * tau))
        d_avg = np.zeros_like(g64)
        for pk, w, tk in zip(p, client_ws, tau):
            if tk > 0:
                d_avg += pk * (g64 - w[i]) / tk
        new_w.append((g64 - tau_eff * d_avg).astype(np.float32))
    return new_w


# =============================== CHECKPOINTS ===============================
class StopForNow(Exception):
    pass


def ckpt_path(run_id):
    return os.path.join(OUT_DIR, "checkpoints", f"{run_id}.npz")


def save_ckpt(run_id, meta, global_w, local, prev):
    os.makedirs(os.path.dirname(ckpt_path(run_id)), exist_ok=True)
    arr = {f"g__{i}": w for i, w in enumerate(global_w)}
    arr.update({f"l__{i}": np.asarray(v.numpy()) for i, v in enumerate(local.all_vars())})
    for k, w in prev.items():
        arr.update({f"p{k}__{i}": x for i, x in enumerate(w)})
    arr["meta"] = np.array(json.dumps(meta))
    tmp = ckpt_path(run_id) + ".tmp.npz"
    np.savez(tmp, **arr)
    os.replace(tmp, ckpt_path(run_id))


def load_ckpt(run_id, local, n_global, n_prev):
    d = np.load(ckpt_path(run_id), allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    global_w = [d[f"g__{i}"] for i in range(n_global)]
    for i, v in enumerate(local.all_vars()):
        v.assign(d[f"l__{i}"])
    prev = {}
    for k in range(NUM_CLIENTS):
        if f"p{k}__0" in d.files:
            prev[k] = [d[f"p{k}__{i}"] for i in range(n_prev)]
    return meta, global_w, prev


def check_time(last_round_seconds):
    if time.time() - SESSION_START + 1.3 * last_round_seconds > TIME_BUDGET_HOURS * 3600:
        raise StopForNow()


# =============================== EVALUATION ===============================
def predict(ms, x):
    return ms.clf.predict(x, batch_size=2048, verbose=0)


def evaluate(prob, y, n_classes):
    pred = prob.argmax(axis=1)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, labels=list(range(n_classes)),
                                                  average='macro', zero_division=0)
    m = {'acc': accuracy_score(y, pred), 'macro_precision': p, 'macro_recall': r, 'macro_f1': f1,
         'balanced_acc': balanced_accuracy_score(y, pred)}
    rec = confusion_matrix(y, pred, labels=list(range(n_classes))).diagonal() / np.bincount(y, minlength=n_classes)
    return m, pred, rec


def results_csv(folder=None):
    return os.path.join(folder or OUT_DIR, "results", "all_runs.csv")


def already_done(run_id):
    path = results_csv()
    return os.path.exists(path) and run_id in set(pd.read_csv(path)['run_id'])


def save_run(run_id, row, y_test, pred, prob, test_idx, meta, client_classes):
    for sub in ("results", "preds", "reports"):
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)
    np.savez_compressed(os.path.join(OUT_DIR, "preds", f"{run_id}.npz"),
                        y_true=y_test, y_pred=pred, y_prob=prob, test_idx=test_idx)
    rep = classification_report(y_test, pred, labels=list(range(len(LABELS))), target_names=LABELS,
                                digits=4, output_dict=True, zero_division=0)
    pd.DataFrame(rep).transpose().to_csv(os.path.join(OUT_DIR, "reports", f"{run_id}_per_class.csv"))
    pd.DataFrame(confusion_matrix(y_test, pred, labels=list(range(len(LABELS)))),
                 index=LABELS, columns=LABELS).to_csv(os.path.join(OUT_DIR, "reports", f"{run_id}_confusion.csv"))
    pd.DataFrame(meta["hist"]).to_csv(os.path.join(OUT_DIR, "reports", f"{run_id}_rounds.csv"), index=False)
    client_classes.to_csv(os.path.join(OUT_DIR, "reports", f"{run_id}_client_classes.csv"))
    path = results_csv()
    pd.DataFrame([row]).to_csv(path, mode='a', header=not os.path.exists(path), index=False)


# =============================== ONE RUN ===============================
def run_one(run_id, variant, aggregator, seed, data, plat):
    global _rounds_this_session
    X_tr, y_tr, x_val, y_val, x_te, y_te, test_idx, n_features, n_classes = data
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    local = ModelSet(n_features, n_classes, variant)
    glob_ms = ModelSet(n_features, n_classes, variant)
    prev_enc = build_models(n_features, n_classes, variant)[0]
    train_client = make_trainer(local, glob_ms, prev_enc, aggregator)

    parts = partition_clients(y_tr, "dirichlet", ALPHA, seed, NUM_CLIENTS)
    labeled_idx = sample_labeled(y_tr, LABELED_PER_CLASS, seed)
    save_or_check_partition(parts, labeled_idx, seed)
    labeled = np.zeros(len(y_tr), dtype=bool)
    labeled[labeled_idx] = True
    y_oh = tf.keras.utils.to_categorical(y_tr, n_classes).astype('float32')
    clients = [(X_tr[idx], X_tr[idx[labeled[idx]]], y_oh[idx[labeled[idx]]]) for idx in parts]
    client_classes = pd.DataFrame([np.bincount(y_tr[idx], minlength=n_classes) for idx in parts],
                                  columns=LABELS, index=[f"client_{k}" for k in range(NUM_CLIENTS)])
    client_classes["labeled"] = [len(c[1]) for c in clients]
    n_k = [len(idx) for idx in parts]

    n_params = local.n_params()
    extra = 12 if aggregator == "FedNova" else 0                 # 3 step counts uploaded per client
    bytes_per_round = NUM_CLIENTS * (2 * 4 * n_params + extra)
    global_w = local.get_w()
    prev, start = {}, 1
    meta = {"round_done": 0, "hist": [], "train_seconds": 0.0, "last_round_seconds": 0.0}
    if os.path.exists(ckpt_path(run_id)):
        meta, global_w, prev = load_ckpt(run_id, local, len(global_w), len(local.enc.get_weights()))
        start = meta["round_done"] + 1
        print(f"  Resuming from checkpoint: {meta['round_done']} rounds done")

    for rnd in range(start, ROUNDS + 1):
        check_time(meta["last_round_seconds"])
        t0 = time.time()
        glob_ms.set_w(global_w)
        client_ws, steps_k, ae_l, clf_l = [], [], [], []
        for k, (x_all, x_lab, y_lab) in enumerate(clients):
            tf.keras.utils.set_random_seed(seed * 100000 + rnd * 100 + k)
            rng = np.random.default_rng(seed * 100000 + rnd * 100 + k)
            local.set_w(global_w)
            if aggregator == "MOON":
                prev_enc.set_weights(prev.get(k, glob_ms.enc.get_weights()))
            steps, la, lc = train_client(x_all, x_lab, y_lab, rng)
            steps_k.append(steps)
            ae_l.append(la)
            clf_l.append(lc)
            client_ws.append(local.get_w())
            if aggregator == "MOON":
                prev[k] = local.enc.get_weights()
        global_w = aggregate(global_w, client_ws, n_k, steps_k, aggregator, local.groups)
        glob_ms.set_w(global_w)
        vm, _, vrec = evaluate(predict(glob_ms, x_val), y_val, n_classes)
        sec = time.time() - t0
        h = {"round": rnd, "val_acc": round(vm["acc"], 6), "val_macro_f1": round(vm["macro_f1"], 6),
             "val_min_recall": round(float(vrec.min()), 6),
             **{f"val_recall_{l}": round(float(r), 6) for l, r in zip(SAFE_LABELS, vrec)},
             "ae_loss": round(float(np.average(ae_l, weights=n_k)), 6),
             "clf_loss": round(float(np.average(clf_l, weights=n_k)), 6),
             "comm_MB_cum": round(rnd * bytes_per_round / 1e6, 3), "seconds": round(sec, 1)}
        meta["hist"].append(h)
        meta.update(round_done=rnd, last_round_seconds=sec, train_seconds=meta["train_seconds"] + sec)
        print(f"  round {rnd}/{ROUNDS}: val acc {vm['acc']:.4f} | val macro-F1 {vm['macro_f1']:.4f} | "
              f"val min recall {vrec.min():.3f} | {sec:.0f}s")
        save_ckpt(run_id, meta, global_w, local, prev)
        _rounds_this_session += 1

    glob_ms.set_w(global_w)
    prob = predict(glob_ms, x_te)
    m, pred, rec = evaluate(prob, y_te, n_classes)
    hist = pd.DataFrame(meta["hist"])
    r90 = hist.loc[hist.val_acc >= 0.90, "round"]
    last10 = hist.tail(10)
    row = dict(run_id=run_id, dataset=DATASET, setting="federated", method=MODEL_NAMES[variant], variant=variant,
               aggregator=aggregator, partition="dirichlet", alpha=ALPHA, clients=NUM_CLIENTS,
               labels_per_class=LABELED_PER_CLASS, labels_total=int(labeled.sum()), seed=seed, rounds=ROUNDS,
               local_epochs=LOCAL_EPOCHS, batch=BATCH, lr=LR,
               **{k: round(float(v), 6) for k, v in m.items()},
               min_class_recall=round(float(rec.min()), 6),
               collapse=bool(rec.min() < COLLAPSE_RECALL),
               **{f"recall_{l}": round(float(r), 6) for l, r in zip(SAFE_LABELS, rec)},
               val_acc_last10=round(float(last10.val_acc.mean()), 6),
               val_f1_last10=round(float(last10.val_macro_f1.mean()), 6),
               rounds_to_90=int(r90.iloc[0]) if len(r90) else -1,
               params_sent=n_params, MB_per_round=round(bytes_per_round / 1e6, 3),
               MB_total=round(ROUNDS * bytes_per_round / 1e6, 1),
               MB_to_90=round(int(r90.iloc[0]) * bytes_per_round / 1e6, 1) if len(r90) else -1,
               smallest_client=int(min(n_k)), largest_client=int(max(n_k)),
               train_seconds=round(meta["train_seconds"], 1), platform=plat, gpu=gpu_name(),
               tf_version=tf.__version__, timestamp=time.strftime("%Y-%m-%d %H:%M"))
    save_run(run_id, row, y_te, pred, prob, test_idx, meta, client_classes)
    os.remove(ckpt_path(run_id))
    print(f"Done {run_id}: test acc {m['acc']:.4f} | macro-F1 {m['macro_f1']:.4f} | "
          f"lowest class recall {rec.min():.3f}" + ("  <-- COLLAPSE" if rec.min() < COLLAPSE_RECALL else ""))


# =============================== STATISTICS ===============================
def mcnemar_exact(y, pred_a, pred_b):
    """Exact (binomial) McNemar test on the discordant pairs."""
    a_ok, b_ok = pred_a == y, pred_b == y
    n01, n10 = int(np.sum(a_ok & ~b_ok)), int(np.sum(~a_ok & b_ok))
    if n01 + n10 == 0:
        return n01, n10, 1.0
    return n01, n10, float(stats.binomtest(n01, n01 + n10, 0.5).pvalue)


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def summarize(folders):
    rows, pred_files = [], {}
    for f in folders:
        if os.path.exists(results_csv(f)):
            rows.append(pd.read_csv(results_csv(f)))
        for pf in glob.glob(os.path.join(f, "preds", "*.npz")):
            pred_files[os.path.basename(pf)[:-4]] = pf
    if not rows:
        print("No results found.")
        return
    res = pd.concat(rows).drop_duplicates("run_id", keep="last")
    print(f"\n{len(res)} runs found.")

    # Table VII rows: mean +- std over seeds
    g = res.groupby(["method", "aggregator"])
    tab = g.agg(n_seeds=("seed", "nunique"),
                acc_mean=("acc", "mean"), acc_std=("acc", "std"),
                f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std"),
                min_recall_mean=("min_class_recall", "mean"), any_collapse=("collapse", "any"),
                r90_mean=("rounds_to_90", "mean"), MB_per_round=("MB_per_round", "first"),
                MB_to_90_mean=("MB_to_90", "mean")).reset_index()
    tab["dagger"] = np.where(tab.any_collapse, "†", "")
    print("\nTable VII (test set, mean ± std over seeds; † = some class recall < "
          f"{COLLAPSE_RECALL:.0%} in at least one seed):")
    print(tab.round(4).to_string(index=False))
    tab.to_csv(os.path.join(OUT_DIR, "table7_summary.csv"), index=False)

    # Paired comparisons: HAFSSL vs FL-AECNN under the same aggregator and seed
    pairs = []
    for agg in AGGREGATORS:
        a = res[(res.method == "HAFSSL") & (res.aggregator == agg)].set_index("seed")
        b = res[(res.method == "FL-AECNN") & (res.aggregator == agg)].set_index("seed")
        seeds = sorted(set(a.index) & set(b.index))
        for s in seeds:
            pa, pb = pred_files.get(a.loc[s, "run_id"]), pred_files.get(b.loc[s, "run_id"])
            if pa is None or pb is None:
                continue
            da, db = np.load(pa), np.load(pb)
            assert np.array_equal(da["test_idx"], db["test_idx"]), "Different test sets!"
            n01, n10, p = mcnemar_exact(da["y_true"], da["y_pred"], db["y_pred"])
            pairs.append(dict(aggregator=agg, seed=s, acc_HAFSSL=a.loc[s, "acc"], acc_FLAECNN=b.loc[s, "acc"],
                              diff=a.loc[s, "acc"] - b.loc[s, "acc"], only_HAFSSL_correct=n01,
                              only_FLAECNN_correct=n10, mcnemar_p=p))
    if not pairs:
        print("\nNo complete HAFSSL / FL-AECNN pairs yet.")
        return
    pr = pd.DataFrame(pairs)
    pr["mcnemar_p_holm"] = holm(pr.mcnemar_p)                       # family: all per-seed McNemar tests
    pr.to_csv(os.path.join(OUT_DIR, "table7_mcnemar_per_seed.csv"), index=False)

    tt = []
    for agg, grp in pr.groupby("aggregator"):
        d = grp["diff"].values
        t_p = float(stats.ttest_rel(grp.acc_HAFSSL, grp.acc_FLAECNN).pvalue) if len(d) >= 2 and np.std(d) > 0 else np.nan
        tt.append(dict(aggregator=agg, n_seeds=len(d), mean_diff_pp=100 * d.mean(),
                       ci95_low_pp=100 * (d.mean() - stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))) if len(d) >= 2 else np.nan,
                       ci95_high_pp=100 * (d.mean() + stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))) if len(d) >= 2 else np.nan,
                       ttest_p=t_p,
                       mcnemar_all_seeds_sig=bool((grp.mcnemar_p_holm < 0.05).all() and (np.sign(d) == np.sign(d.mean())).all())))
    tt = pd.DataFrame(tt)
    ok = tt.ttest_p.notna()
    tt["ttest_p_holm"] = np.nan
    tt.loc[ok, "ttest_p_holm"] = holm(tt.loc[ok, "ttest_p"])        # family: the 4 aggregator comparisons
    tt["claim_better"] = (tt.ttest_p_holm < 0.05) & tt.mcnemar_all_seeds_sig & (tt.mean_diff_pp > 0)
    tt.to_csv(os.path.join(OUT_DIR, "table7_paired_tests.csv"), index=False)
    print("\nHAFSSL minus FL-AECNN, accuracy (percentage points), paired by seed:")
    print(tt.round(4).to_string(index=False))
    print("\nclaim_better = paired t-test (Holm) AND McNemar (Holm) significant in every seed, same sign.")


# =============================== MAIN ===============================
def main():
    if ONLY_SUMMARIZE:
        os.makedirs(OUT_DIR, exist_ok=True)
        summarize(SUMMARY_DIRS or [OUT_DIR])
        return
    plat = platform_name()
    restore_previous_output()
    tf.config.experimental.enable_op_determinism()
    save_config()
    X, y, n_features = load_data()
    tr, va, te = get_split(y)
    data = (X[tr], y[tr], X[va], y[va], X[te], y[te], te, n_features, len(LABELS))
    plan = [(seed, agg, model) for seed in SEEDS for agg in AGGREGATORS for model in MODELS]
    print(f"TABLE VII | {DATASET} | train {len(tr)} | val {len(va)} | test {len(te)} | {plat} | {gpu_name()}")
    print(f"models {MODELS} | aggregators {AGGREGATORS} | seeds {SEEDS} | alpha {ALPHA} | K {NUM_CLIENTS} | "
          f"rounds {ROUNDS} | local epochs {LOCAL_EPOCHS} | labels/class {LABELED_PER_CLASS} | "
          f"{len(plan)} runs | QUICK_TEST {QUICK_TEST}")
    try:
        for seed, aggregator, model in plan:     # seed outermost: complete paired sets come first
            variant = MODEL_TO_VARIANT[model]
            run_id = (f"{DATASET}_{MODEL_NAMES[variant]}_{aggregator}_a{ALPHA}_K{NUM_CLIENTS}"
                      f"_L{LABELED_PER_CLASS}_s{seed}")
            if already_done(run_id):
                print("Skip (already done):", run_id)
                continue
            print(f"\n===== {run_id} =====")
            run_one(run_id, variant, aggregator, seed, data, plat)
    except StopForNow:
        print("\n" + "=" * 70)
        print("STOPPED SAFELY (time budget). All progress is saved in:", OUT_DIR)
        print("Kaggle: add this version's output as input, set RESUME_FROM, Save & Run All.")
        print("=" * 70)
        return
    print("\nALL RUNS IN THIS NOTEBOOK FINISHED.")
    summarize([OUT_DIR])


if __name__ == "__main__":
    main()
