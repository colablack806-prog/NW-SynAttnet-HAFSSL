# -*- coding: utf-8 -*-
"""
HAFSSL / FL-AECNN - extreme heterogeneity (Dirichlet alpha = 0.01, K = 10), R2 revision.
THIS COPY: 10 POPULAR APPS (second dataset for Tables VIII and IX), MODEL = "HAFSSL".
Same protocol and same code as the ISCX alpha = 0.01 runs (this script with DATASET = "ISCX");
only the data path, the labels and the size checks differ. Run the HAFSSL copy and the FL-AECNN copy
in two notebooks: for each seed they get exactly the same client partition and labeled flows.

MODELS (standard, as in Section IV-D and Tables V, VI and VIII):
  HAFSSL : encoder Conv64-Conv64-MaxPool-[MHA->SE, 4 heads x key_dim 32, FFN 128]-Conv32-Conv32-Dense(39)
           classifier Dense-Reshape-Conv64-Conv64-MaxPool-Conv128-Conv128-MaxPool-Dense(128)-Softmax
  FL-AECNN: unchanged (same layers without the attention block) [10]
  => the only difference between the two models is the hybrid attention block in the encoder.
     On 10 Popular Apps HAFSSL sends 435,862 parameters, FL-AECNN 385,234 (10-class output layer).

PROTOCOL (identical to ISCX alpha = 0.01):
  every round, on each client:
    stage 1   : autoencoder on the client's UNLABELED flows, 3 epochs, batch 32, Adam (skipped if < 50 flows)
    stage 1.5 : encoder fine-tuned with the classifier head FROZEN, 2 epochs, batch 256, Adam lr 1e-4
    stage 2   : encoder + classifier, 3 epochs, batch 256, Adam
  labels       : 200 labeled flows per class on each client, for the classes it holds (seeded)
  aggregation  : weighted FedAvg n_k / n (Eq. 5); clients with < 50 flows are skipped
  rounds       : 100; seeds 0, 1, 2 (same partition and labeled flows as FL-AECNN for each seed)
  the GLOBAL model is evaluated once, after the last round, on the frozen test set (duplicates removed)

Optional: if the FL-AECNN copy has finished, set PAIR_CHECK_DIR in the HAFSSL copy to its output folder;
each run then stops immediately if its client partition or labeled flows differ from the FL-AECNN run.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import json, time, shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             balanced_accuracy_score, confusion_matrix, classification_report)

# =============================== CONFIG ===============================
MODEL     = "HAFSSL"          # THIS COPY ("HAFSSL" = with hybrid attention, "FL-AECNN" = without)

DATA_PATH = "data/10PopularApps/pcapdroid_10class_each_normalized_cuttedfloefeature.csv"
DATASET   = "Apps10"
# Names in the order of the label codes 0..9 in the csv (must match the centralized Table II script)
LABELS    = ['TikTok', 'iQiyi Video', 'Jingdong Shopping', 'Snack Video', 'QQ Music',
             'QQ', 'Taobao Shopping', 'NetEase Cloud Music', 'Arena of Valor', 'WeChat']
MODEL_TAG = "standard"        # in the HAFSSL run ids
OUT_DIR   = f"/kaggle/working/FED_Apps10_alpha001_{MODEL}"

# Expected sizes after de-duplication (from the centralized Table II runs); the run stops if they differ
EXPECTED_ROWS  = {"Apps10": 170459, "ISCX": 82435}
EXPECTED_SPLIT = {"Apps10": (138071, 15342, 17046), "ISCX": (66771, 7420, 8244)}

# Optional pairing check (HAFSSL copy): the FL-AECNN alpha = 0.01 output folder (contains reports/*_clients.csv)
PAIR_CHECK_DIR = ""

AGGREGATORS = ["FedAvg"]
ALPHA        = 0.01
NUM_CLIENTS  = 10
ROUNDS       = 100
LOCAL_EPOCHS = 3               # stage 1 and stage 2
STAGE15_EPOCHS, STAGE15_LR = 2, 1e-4
BATCH_AE, BATCH_CLF = 32, 256
LR = 1e-3
LABELS_PER_CLASS_PER_CLIENT = 200    # same rule as ISCX: 200 per class each client holds
MIN_CLIENT_FLOWS = 50
SEEDS = [0, 1, 2]

NUM_HEADS, KEY_DIM, SE_RATIO, LATENT_DIM, ATT_DROPOUT = 4, 32, 16, 39, 0.1
EXPECTED_PARAMS = {("ISCX", "HAFSSL"): 435217, ("ISCX", "FL-AECNN"): 384589,      # Tables V and VI
                   ("Apps10", "HAFSSL"): 435862, ("Apps10", "FL-AECNN"): 385234}  # 10 Popular Apps R2 runs
SPLIT_SEED        = 2026    # same as every other experiment = same test set
RESUME_FROM       = ""     # Kaggle: previous version's output folder, to continue after a stop
TIME_BUDGET_HOURS = 11.0
QUICK_TEST        = False   # True = 2 rounds, 1 epoch, seed 0 (a few minutes)

assert MODEL in ("HAFSSL", "FL-AECNN"), 'MODEL must be exactly "HAFSSL" or "FL-AECNN"'
if QUICK_TEST:
    ROUNDS, LOCAL_EPOCHS, STAGE15_EPOCHS, SEEDS = 2, 1, 1, [0]
    OUT_DIR = OUT_DIR + "_quicktest"
SESSION_START = time.time()
_STOP_AFTER_ROUNDS = int(os.environ.get("STOP_AFTER_ROUNDS", "0"))   # testing only
_rounds_this_session = 0


# =============================== SETUP ===============================
def platform_name():
    if os.path.exists("/kaggle"):
        return "kaggle"
    return "colab" if os.path.exists("/content") else "pc"


def gpu_name():
    gpus = tf.config.list_physical_devices('GPU')
    return tf.config.experimental.get_device_details(gpus[0]).get('device_name', 'GPU') if gpus else "CPU only"


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


def write_manifest():
    os.makedirs(OUT_DIR, exist_ok=True)
    keys = ["MODEL", "MODEL_TAG", "DATA_PATH", "DATASET", "LABELS", "AGGREGATORS", "ALPHA", "NUM_CLIENTS",
            "ROUNDS", "LOCAL_EPOCHS", "STAGE15_EPOCHS", "STAGE15_LR", "BATCH_AE", "BATCH_CLF", "LR",
            "LABELS_PER_CLASS_PER_CLIENT", "MIN_CLIENT_FLOWS", "SEEDS", "NUM_HEADS", "KEY_DIM", "SE_RATIO",
            "LATENT_DIM", "ATT_DROPOUT", "SPLIT_SEED"]
    cfg = {k: globals()[k] for k in keys}
    cfg.update(classifier_attention=False, head_dropout=0.0, tf_version=tf.__version__, gpu=gpu_name(),
               started=time.strftime("%Y-%m-%d %H:%M"))
    with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)


# =============================== DATA ===============================
def load_data():
    df = pd.read_csv(DATA_PATH)
    feat_cols = [c for c in df.columns if c != 'label']
    n_before = len(df)
    df = df.drop_duplicates(subset=feat_cols).reset_index(drop=True)
    print(f"Rows: {n_before} -> {len(df)} after removing {n_before - len(df)} exact duplicates")
    X = df[feat_cols].values.astype('float32').reshape(len(df), len(feat_cols), 1)
    y = df['label'].values.astype(int)
    print(f"Features: {len(feat_cols)} | rows per class: {dict(zip(*np.unique(y, return_counts=True)))}")
    assert len(feat_cols) == 66, f"Expected 66 flow features, found {len(feat_cols)}"
    assert np.array_equal(np.unique(y), np.arange(len(LABELS))), \
        f"Label codes must be 0..{len(LABELS) - 1}; found {np.unique(y)}"
    if DATASET in EXPECTED_ROWS:
        assert len(df) == EXPECTED_ROWS[DATASET], \
            f"De-duplicated rows {len(df)} differ from Table II ({EXPECTED_ROWS[DATASET]})"
    return X, y, len(feat_cols)


def get_split(y):
    path = os.path.join(OUT_DIR, "splits", f"{DATASET}_split.npz")
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


def partition_clients(y_train, alpha, seed, n_clients, min_size=10):
    """Dirichlet(alpha) per class over clients (label + quantity skew). Same seed = same partition."""
    rng = np.random.default_rng(10_000 + seed)
    n = len(y_train)
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
            for k, chunk in enumerate(np.split(idx_c, (np.cumsum(p) * len(idx_c)).astype(int)[:-1])):
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


def label_clients(parts, y_train, per_class, seed):
    """Each client labels up to `per_class` flows of every class it holds (original rule). Seeded."""
    rng = np.random.default_rng(20_000 + seed)
    labeled = np.zeros(len(y_train), dtype=bool)
    for idx in parts:
        for c in np.unique(y_train[idx]):
            pool = idx[y_train[idx] == c]
            labeled[rng.choice(pool, min(per_class, len(pool)), replace=False)] = True
    return labeled


def check_pairing(client_table, seed):
    """Stops the run if the partition or labeled flows differ from the FL-AECNN run of the same seed."""
    if not PAIR_CHECK_DIR:
        return
    f = os.path.join(PAIR_CHECK_DIR, "reports", f"{DATASET}_FL-AECNN_FedAvg_a{ALPHA}_K{NUM_CLIENTS}_s{seed}_clients.csv")
    if not os.path.exists(f):
        print("  Pairing check skipped (file not found):", f)
        return
    ref = pd.read_csv(f, index_col=0)
    cols = LABELS + ["labeled"]
    assert np.array_equal(ref[cols].values.astype(int), client_table[cols].values.astype(int)), \
        f"Seed {seed}: client partition or labeled flows differ from the FL-AECNN run in {PAIR_CHECK_DIR}"
    print(f"  Pairing check passed: same clients and labeled flows as FL-AECNN seed {seed}")


# =============================== MODEL ===============================
def se_block(x):
    ch = x.shape[-1]
    s = layers.GlobalAveragePooling1D()(x)
    s = layers.Dense(ch // SE_RATIO, activation='relu')(s)
    s = layers.Dense(ch, activation='sigmoid')(s)
    return layers.Multiply()([x, layers.Reshape((1, ch))(s)])


def hybrid_attention_block(x, key_dim):
    """MHA -> SE, residual, FFN (same block as in Tables V, VI and VIII)."""
    a = se_block(layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=key_dim, dropout=ATT_DROPOUT)(x, x))
    x = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, a]))
    f = layers.Dense(key_dim * 4, activation='relu')(x)
    f = layers.Dense(x.shape[-1])(layers.Dropout(ATT_DROPOUT)(f))
    return layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, f]))


def build_models(n_features, n_classes, use_attention):
    ie = layers.Input(shape=(n_features, 1))
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(ie)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D()(x)
    if use_attention:
        x = hybrid_attention_block(x, KEY_DIM)                 # encoder attention: the ONLY difference
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(x)
    steps, ch = x.shape[1], x.shape[2]
    enc = Model(ie, layers.Dense(LATENT_DIM, activation='relu')(layers.Flatten()(x)), name='encoder')

    idd = layers.Input(shape=(LATENT_DIM,))
    d = layers.Reshape((steps, ch))(layers.Dense(steps * ch)(idd))
    d = layers.Conv1D(32, 3, padding='same', activation='relu')(d)
    d = layers.Conv1D(32, 3, padding='same', activation='relu')(d)
    d = layers.UpSampling1D(2)(d)
    d = layers.Conv1D(64, 3, padding='same', activation='relu')(d)
    dec = Model(idd, layers.Conv1D(1, 3, padding='same', activation='relu')(d), name='decoder')

    # classifier head: identical for both models (no attention, no dropout)
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
    return enc, dec, head, Model(inp, dec(enc(inp))), Model(inp, head(enc(inp)))


class ModelSet:
    def __init__(self, n_features, n_classes, use_attention):
        self.enc, self.dec, self.head, self.ae, self.clf = build_models(n_features, n_classes, use_attention)
        self.parts = [self.enc, self.dec, self.head]
        self.sizes = [len(m.get_weights()) for m in self.parts]

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


# =============================== LOCAL TRAINING (original 3 stages) ===============================
def reset_optimizer(opt):
    for v in opt.variables:
        if "learning_rate" not in v.path:
            v.assign(tf.zeros_like(v))


def make_trainer(local):
    ae_vars, enc_vars, clf_vars = (local.ae.trainable_variables, local.enc.trainable_variables,
                                   local.clf.trainable_variables)
    opt_ae, opt_15, opt_clf = (tf.keras.optimizers.Adam(LR), tf.keras.optimizers.Adam(STAGE15_LR),
                               tf.keras.optimizers.Adam(LR))
    opt_ae.build(ae_vars)
    opt_15.build(enc_vars)
    opt_clf.build(clf_vars)
    cce = tf.keras.losses.CategoricalCrossentropy()

    @tf.function(reduce_retracing=True)
    def ae_step(x):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(x - local.ae(x, training=True)))
        opt_ae.apply_gradients(zip(tape.gradient(loss, ae_vars), ae_vars))

    @tf.function(reduce_retracing=True)
    def stage15_step(x, y):          # classifier head frozen: only the encoder is updated
        with tf.GradientTape() as tape:
            loss = cce(y, local.clf(x, training=True))
        opt_15.apply_gradients(zip(tape.gradient(loss, enc_vars), enc_vars))

    @tf.function(reduce_retracing=True)
    def clf_step(x, y):
        with tf.GradientTape() as tape:
            loss = cce(y, local.clf(x, training=True))
        opt_clf.apply_gradients(zip(tape.gradient(loss, clf_vars), clf_vars))

    def run_epochs(step, xs, ys, epochs, batch, rng):
        n = 0
        for _ in range(epochs):
            order = rng.permutation(len(xs))
            for i in range(0, len(order), batch):
                b = order[i:i + batch]
                step(tf.constant(xs[b])) if ys is None else step(tf.constant(xs[b]), tf.constant(ys[b]))
                n += 1
        return n

    def train_client(x_unl, x_lab, y_lab, rng):
        for opt in (opt_ae, opt_15, opt_clf):
            reset_optimizer(opt)
        steps = 0
        if len(x_unl) >= MIN_CLIENT_FLOWS:                                          # stage 1
            steps += run_epochs(ae_step, x_unl, None, LOCAL_EPOCHS, BATCH_AE, rng)
        if len(x_lab) > 0:
            steps += run_epochs(stage15_step, x_lab, y_lab, STAGE15_EPOCHS, BATCH_CLF, rng)   # stage 1.5
            steps += run_epochs(clf_step, x_lab, y_lab, LOCAL_EPOCHS, BATCH_CLF, rng)         # stage 2
        return max(steps, 1)

    return train_client


def aggregate(global_w, client_ws, n_k, tau_k, aggregator):
    p = np.array(n_k, dtype=np.float64) / np.sum(n_k)
    if aggregator == "FedNova":
        tau_eff = float(np.sum(p * np.array(tau_k)))
        return [(g - tau_eff * sum(pk * (g - w[i]) / tk for pk, w, tk in zip(p, client_ws, tau_k))).astype(np.float32)
                for i, g in enumerate(global_w)]
    return [sum(pk * w[i] for pk, w in zip(p, client_ws)).astype(np.float32) for i in range(len(global_w))]


# =============================== CHECKPOINTS ===============================
class StopForNow(Exception):
    pass


def ckpt_path(run_id):
    return os.path.join(OUT_DIR, "checkpoints", f"{run_id}.npz")


def save_ckpt(run_id, meta, global_w, local):
    os.makedirs(os.path.dirname(ckpt_path(run_id)), exist_ok=True)
    arr = {f"g__{i}": w for i, w in enumerate(global_w)}
    arr.update({f"l__{i}": np.asarray(v.numpy()) for i, v in enumerate(local.all_vars())})
    arr["meta"] = np.array(json.dumps(meta))
    tmp = ckpt_path(run_id) + ".tmp.npz"
    np.savez(tmp, **arr)
    os.replace(tmp, ckpt_path(run_id))


def load_ckpt(run_id, local, n_global):
    d = np.load(ckpt_path(run_id), allow_pickle=False)
    for i, v in enumerate(local.all_vars()):
        v.assign(d[f"l__{i}"])
    return json.loads(str(d["meta"])), [d[f"g__{i}"] for i in range(n_global)]


def check_time(last_round_seconds):
    if _STOP_AFTER_ROUNDS and _rounds_this_session >= _STOP_AFTER_ROUNDS:
        raise StopForNow()
    if time.time() - SESSION_START + 1.3 * last_round_seconds > TIME_BUDGET_HOURS * 3600:
        raise StopForNow()


# =============================== EVALUATION ===============================
def evaluate(prob, y):
    pred = prob.argmax(axis=1)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average='macro', zero_division=0)
    m = {'acc': accuracy_score(y, pred), 'macro_precision': p, 'macro_recall': r, 'macro_f1': f1,
         'balanced_acc': balanced_accuracy_score(y, pred)}
    assert abs(m['macro_recall'] - m['balanced_acc']) < 1e-9
    return m, pred


def already_done(run_id):
    path = os.path.join(OUT_DIR, "results", "all_runs.csv")
    return os.path.exists(path) and run_id in set(pd.read_csv(path)['run_id'])


def save_run(run_id, row, y_test, pred, prob, test_idx, hist, client_table):
    for sub in ("results", "preds", "reports"):
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)
    np.savez_compressed(os.path.join(OUT_DIR, "preds", f"{run_id}.npz"),
                        y_true=y_test, y_pred=pred, y_prob=prob, test_idx=test_idx)
    labs = list(range(len(LABELS)))
    rep = classification_report(y_test, pred, labels=labs, target_names=LABELS, digits=4,
                                output_dict=True, zero_division=0)
    pd.DataFrame(rep).transpose().to_csv(os.path.join(OUT_DIR, "reports", f"{run_id}_per_class.csv"))
    pd.DataFrame(confusion_matrix(y_test, pred, labels=labs), index=LABELS, columns=LABELS).to_csv(
        os.path.join(OUT_DIR, "reports", f"{run_id}_confusion.csv"))
    pd.DataFrame(hist).to_csv(os.path.join(OUT_DIR, "reports", f"{run_id}_rounds.csv"), index=False)
    client_table.to_csv(os.path.join(OUT_DIR, "reports", f"{run_id}_clients.csv"))
    path = os.path.join(OUT_DIR, "results", "all_runs.csv")
    pd.DataFrame([row]).to_csv(path, mode='a', header=not os.path.exists(path), index=False)


# =============================== ONE RUN ===============================
def run_one(run_id, aggregator, seed, data, plat):
    global _rounds_this_session
    X_tr, y_tr, x_val, y_val, x_te, y_te, test_idx, n_features, n_classes = data
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    use_att = MODEL == "HAFSSL"
    local, glob = ModelSet(n_features, n_classes, use_att), ModelSet(n_features, n_classes, use_att)
    train_client = make_trainer(local)

    n_params = local.n_params()
    print(f"  Parameters sent per client and direction: {n_params:,}")
    if (DATASET, MODEL) in EXPECTED_PARAMS:
        assert n_params == EXPECTED_PARAMS[(DATASET, MODEL)], \
            f"{MODEL} has {n_params} parameters; expected {EXPECTED_PARAMS[(DATASET, MODEL)]} (Tables V and VI)"

    parts = partition_clients(y_tr, ALPHA, seed, NUM_CLIENTS)
    labeled = label_clients(parts, y_tr, LABELS_PER_CLASS_PER_CLIENT, seed)
    y_oh = tf.keras.utils.to_categorical(y_tr, n_classes).astype('float32')
    clients = [(X_tr[idx[~labeled[idx]]], X_tr[idx[labeled[idx]]], y_oh[idx[labeled[idx]]]) for idx in parts]
    active = [k for k, idx in enumerate(parts) if len(idx) >= MIN_CLIENT_FLOWS]
    client_table = pd.DataFrame([np.bincount(y_tr[idx], minlength=n_classes) for idx in parts],
                                columns=LABELS, index=[f"client_{k}" for k in range(NUM_CLIENTS)])
    client_table["labeled"] = [len(c[1]) for c in clients]
    client_table["active"] = [k in active for k in range(NUM_CLIENTS)]
    print(client_table.to_string())
    check_pairing(client_table, seed)

    bytes_per_round = len(active) * 2 * 4 * n_params            # float32, up + down, active clients
    global_w = local.get_w()
    meta = {"round_done": 0, "hist": [], "train_seconds": 0.0, "last_round_seconds": 0.0}
    if os.path.exists(ckpt_path(run_id)):
        meta, global_w = load_ckpt(run_id, local, len(global_w))
        print(f"  Resuming from checkpoint: {meta['round_done']} rounds done")

    for rnd in range(meta["round_done"] + 1, ROUNDS + 1):
        check_time(meta["last_round_seconds"])
        t0 = time.time()
        client_ws, n_k, tau_k = [], [], []
        for k in active:
            x_unl, x_lab, y_lab = clients[k]
            tf.keras.utils.set_random_seed(seed * 100000 + rnd * 100 + k)
            rng = np.random.default_rng(seed * 100000 + rnd * 100 + k)
            local.set_w(global_w)
            tau_k.append(train_client(x_unl, x_lab, y_lab, rng))
            client_ws.append(local.get_w())
            n_k.append(len(parts[k]))
        global_w = aggregate(global_w, client_ws, n_k, tau_k, aggregator)
        glob.set_w(global_w)
        vm, _ = evaluate(glob.clf.predict(x_val, batch_size=2048, verbose=0), y_val)
        sec = time.time() - t0
        meta["hist"].append({"round": rnd, "val_acc": round(vm["acc"], 6), "val_macro_f1": round(vm["macro_f1"], 6),
                             "comm_MB_cum": round(rnd * bytes_per_round / 1e6, 3), "seconds": round(sec, 1)})
        meta.update(round_done=rnd, last_round_seconds=sec, train_seconds=meta["train_seconds"] + sec)
        print(f"  round {rnd}/{ROUNDS}: val acc {vm['acc']:.4f} | val macro-F1 {vm['macro_f1']:.4f} | {sec:.0f}s")
        save_ckpt(run_id, meta, global_w, local)
        _rounds_this_session += 1

    glob.set_w(global_w)
    prob = glob.clf.predict(x_te, batch_size=2048, verbose=0)
    m, pred = evaluate(prob, y_te)
    recall = confusion_matrix(y_te, pred, labels=list(range(n_classes))).diagonal() / np.bincount(y_te, minlength=n_classes)
    hist = pd.DataFrame(meta["hist"])
    r90 = hist.loc[hist.val_acc >= 0.90, "round"]
    row = dict(run_id=run_id, dataset=DATASET, setting="federated", method=MODEL,
               model_config=(MODEL_TAG if MODEL == "HAFSSL" else "original"),
               classifier_attention=False, head_dropout=0.0, aggregator=aggregator,
               partition="dirichlet", alpha=ALPHA, clients=NUM_CLIENTS, active_clients=len(active),
               labels_per_class_per_client=LABELS_PER_CLASS_PER_CLIENT, labels_total=int(labeled.sum()),
               seed=seed, rounds=ROUNDS, local_epochs=LOCAL_EPOCHS, stage15_epochs=STAGE15_EPOCHS,
               batch_ae=BATCH_AE, batch_clf=BATCH_CLF, lr=LR,
               **{k: round(float(v), 6) for k, v in m.items()},
               min_class_recall=round(float(recall.min()), 6),
               **{f"recall_{name}": round(float(r), 6) for name, r in zip(LABELS, recall)},
               rounds_to_90=int(r90.iloc[0]) if len(r90) else -1, params_sent=n_params,
               MB_per_round=round(bytes_per_round / 1e6, 3), MB_total=round(ROUNDS * bytes_per_round / 1e6, 1),
               train_seconds=round(meta["train_seconds"], 1), platform=plat, gpu=gpu_name(),
               tf_version=tf.__version__, timestamp=time.strftime("%Y-%m-%d %H:%M"))
    save_run(run_id, row, y_te, pred, prob, test_idx, meta["hist"], client_table)
    os.remove(ckpt_path(run_id))
    print(f"Done {run_id}: test acc {m['acc']:.4f} | macro-F1 {m['macro_f1']:.4f} | "
          f"per-class recall {np.round(recall, 3).tolist()}")


# =============================== MAIN ===============================
def main():
    plat = platform_name()
    restore_previous_output()
    write_manifest()
    tf.config.experimental.enable_op_determinism()
    X, y, n_features = load_data()
    tr, va, te = get_split(y)
    if DATASET in EXPECTED_SPLIT:
        assert (len(tr), len(va), len(te)) == EXPECTED_SPLIT[DATASET], \
            f"Split {len(tr)}/{len(va)}/{len(te)} differs from Table II {EXPECTED_SPLIT[DATASET]}"
    data = (X[tr], y[tr], X[va], y[va], X[te], y[te], te, n_features, len(LABELS))
    print(f"Train {len(tr)} | Val {len(va)} | Test {len(te)} | {plat} | {gpu_name()}")
    desc = "hybrid attention in the encoder" if MODEL == "HAFSSL" else "no attention"
    print(f"MODEL {MODEL} ({desc}; no classifier attention, no head dropout) | "
          f"alpha {ALPHA} | K={NUM_CLIENTS} | rounds {ROUNDS} | local epochs {LOCAL_EPOCHS} "
          f"(+ stage 1.5: {STAGE15_EPOCHS}) | labels {LABELS_PER_CLASS_PER_CLIENT}/class/client | "
          f"aggregators {AGGREGATORS} | seeds {SEEDS}")
    tag = f"_{MODEL_TAG}" if MODEL == "HAFSSL" else ""
    try:
        for seed in SEEDS:
            for aggregator in AGGREGATORS:
                run_id = f"{DATASET}_{MODEL}{tag}_{aggregator}_a{ALPHA}_K{NUM_CLIENTS}_s{seed}"
                if already_done(run_id):
                    print("Skip (already done):", run_id)
                    continue
                print(f"\n===== {run_id} =====")
                run_one(run_id, aggregator, seed, data, plat)
    except StopForNow:
        print("\n" + "=" * 70)
        print("STOPPED SAFELY (time budget). All progress is saved in:", OUT_DIR)
        print("Kaggle: add this version's output as input, set RESUME_FROM, Save & Run All.")
        print("=" * 70)
        return
    res = pd.read_csv(os.path.join(OUT_DIR, "results", "all_runs.csv"))
    print("\nALL RUNS FINISHED. Mean ± std over seeds (test set):")
    print(res.groupby(['method', 'aggregator'])[['acc', 'macro_f1', 'min_class_recall']].agg(['mean', 'std']).round(4))


if __name__ == "__main__":
    main()
