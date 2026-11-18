# -*- coding: utf-8 -*-
"""
Federated attention ablation (Table V), R2 revision.

This script trains the four forms of the hybrid attention block:
    MHA->SE (= HAFSSL), SE->MHA, MHA-only, SE-only   x   FedAvg   x   seeds 0, 1, 2   (12 runs)
on ISCX-VPN2016 with K = 10 clients, Dirichlet alpha = 0.5, 1,000 labeled flows per class,
50 rounds and 5 local epochs (the settings of the federated comparison, Table VI).

All four forms must be run with the same software version (TensorFlow 2.21.0 for Table V): a repeat with
another TensorFlow version, on the same partitions and labeled flows, changed the seed-0 results.

Protocol:
  1. The GLOBAL model after the final round is evaluated once on the frozen test set.
  2. Macro-averaged metrics with zero_division=0.
  3. Aggregation is weighted by client data size n_k / n (FedAvg).
  4. Exact duplicate rows are removed; the SAME frozen split as the centralized runs (split seed 2026).
  5. Seeds 0, 1, 2 fix the model initialization, the client partition and the labeled flows (paired design).
  6. Communication = bytes of the weights actually sent (float32, upload + download).
  7. Checkpoint after every round: if the session stops, run again and it continues.

Local training on each client, every round:
  stage 1: autoencoder on all local flows (MSE), LOCAL_EPOCHS epochs
  stage 2: encoder + classifier on the local labeled flows (cross-entropy), LOCAL_EPOCHS epochs
  the encoder, decoder and classifier weights are sent to the server and aggregated.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import json, time, shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, f1_score,
                             balanced_accuracy_score, confusion_matrix, classification_report)

# =============================== CONFIG ===============================
# ISCX-VPN2016 (the dataset of Table V)
DATA_PATH = "data/ISCX-VPN2016/ISCX_5class_each_normalized_cuttedfloefeature.csv"
DATASET   = "ISCX"
LABELS    = ['chat', 'email', 'file', 'streaming', 'voip']
OUT_DIR   = "outputs/FED_ISCX_ablation"

# The four forms of the block ("MHA->SE" = HAFSSL). To add the MHA->SE runs to an earlier output that
# already holds the other three, set RESUME_FROM to that output folder: finished runs are skipped.
VARIANTS     = ["MHA->SE", "SE->MHA", "MHA", "SE"]
AGGREGATORS  = ["FedAvg"]
PARTITIONS   = [("dirichlet", 0.5)]
LABELED_PER_CLASS_LIST = [1000]                    # labeled flows per class, in the WHOLE federation

NUM_CLIENTS  = 10
ROUNDS       = 50
LOCAL_EPOCHS = 5
BATCH        = 256
LR           = 1e-3
SEEDS        = [0, 1, 2]
FEDPROX_MU   = 0.1                                 # not used with FedAvg
MOON_MU, MOON_TAU = 5.0, 0.5                       # not used with FedAvg

SPLIT_SEED        = 2026    # same as the centralized runs = same test set
SPLIT_FROM        = ""      # optional: path of the centralized splits/<DATASET>_split.npz
RESUME_FROM       = ""      # previous output folder, to continue after a stop
TIME_BUDGET_HOURS = 11.0    # stops cleanly before a 12-hour session limit
QUICK_TEST        = False   # True = 2 rounds, 1 local epoch, seed 0 (a few minutes)

if QUICK_TEST:
    ROUNDS, LOCAL_EPOCHS, SEEDS = 2, 1, [0]
    OUT_DIR = OUT_DIR + "_quicktest"

MODEL_NAMES = {"MHA->SE": "HAFSSL", "SE->MHA": "HAFSSL-SE>MHA", "MHA": "HAFSSL-MHAonly", "SE": "HAFSSL-SEonly"}
LATENT_DIM, NUM_HEADS, KEY_DIM, SE_RATIO = 39, 4, 32, 16
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


# =============================== DATA (same as centralized) ===============================
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
    """IID: equal random shards. Dirichlet: per class, proportions ~ Dir(alpha) over clients
    (label + quantity skew; capped like NIID-Bench). If some client still has fewer than
    min_size flows, flows are moved to it at random from the largest client."""
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
            if not np.all(np.isfinite(p_raw)) or p_raw.sum() <= 0:      # tiny alpha: all mass on one client
                p_raw = np.eye(n_clients)[rng.integers(n_clients)]
            p = p_raw * np.array([len(parts[k]) < n / n_clients for k in range(n_clients)])
            p = p / p.sum() if p.sum() > 0 else p_raw / p_raw.sum()     # never an even split
            cuts = (np.cumsum(p) * len(idx_c)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(idx_c, cuts)):
                parts[k].extend(chunk.tolist())
        smallest = min(len(pt) for pt in parts)
        if best is None or smallest > min(len(pt) for pt in best):
            best = parts
        if smallest >= min_size:
            break
    parts = [list(pt) for pt in best]
    for k in range(n_clients):                     # fallback for extreme alpha
        while len(parts[k]) < min_size:
            big = int(np.argmax([len(pt) for pt in parts]))
            parts[k].append(parts[big].pop(int(rng.integers(len(parts[big])))))
    return [np.sort(np.array(pt)) for pt in parts]


# =============================== MODEL ===============================
def se_block(x, ratio=SE_RATIO):
    ch = x.shape[-1]
    s = layers.GlobalAveragePooling1D()(x)
    s = layers.Dense(ch // ratio, activation='relu')(s)
    s = layers.Dense(ch, activation='sigmoid')(s)
    return layers.Multiply()([x, layers.Reshape((1, ch))(s)])


def attention_block(x, variant):
    """Forms of the hybrid attention block: MHA->SE (HAFSSL), SE->MHA, MHA only, SE only;
    residual + layer normalization + feed-forward sublayer as in SynAttnNet."""
    if variant == "MHA->SE":
        a = se_block(layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=KEY_DIM, dropout=0.1)(x, x))
    elif variant == "SE->MHA":
        s = se_block(x)
        a = layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=KEY_DIM, dropout=0.1)(s, s)
    elif variant == "MHA":
        a = layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=KEY_DIM, dropout=0.1)(x, x)
    elif variant == "SE":
        a = se_block(x)
    else:
        raise ValueError(f"Unknown variant {variant}; use 'MHA->SE', 'SE->MHA', 'MHA' or 'SE'")
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
    """Encoder, decoder and head that are sent and aggregated as one list of weights."""
    def __init__(self, n_features, n_classes, variant):
        self.enc, self.dec, self.head, self.ae, self.clf = build_models(n_features, n_classes, variant)
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

    def all_vars(self):            # weights + dropout random state, for exact checkpoints
        return [v for m in self.parts for v in m.variables]


# =============================== LOCAL TRAINING ===============================
def reset_optimizer(opt):
    for v in opt.variables:
        if "learning_rate" not in v.path:
            v.assign(tf.zeros_like(v))


def make_trainer(local, glob, prev_enc, aggregator):
    prox, moon = aggregator == "FedProx", aggregator == "MOON"
    ae_vars, clf_vars = local.ae.trainable_variables, local.clf.trainable_variables
    g_ae_vars, g_clf_vars = glob.ae.trainable_variables, glob.clf.trainable_variables
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
                zg = tf.math.l2_normalize(glob.enc(x, training=False), axis=1)
                zp = tf.math.l2_normalize(prev_enc(x, training=False), axis=1)
                sg = tf.reduce_sum(zn * zg, axis=1) / MOON_TAU
                sp = tf.reduce_sum(zn * zp, axis=1) / MOON_TAU
                loss += MOON_MU * tf.reduce_mean(tf.reduce_logsumexp(tf.stack([sg, sp], axis=1), axis=1) - sg)
        opt_clf.apply_gradients(zip(tape.gradient(loss, clf_vars), clf_vars))
        return loss

    def train_client(x_all, x_lab, y_lab, rng):
        reset_optimizer(opt_ae)
        reset_optimizer(opt_clf)
        steps = 0
        for _ in range(LOCAL_EPOCHS):                       # stage 1: autoencoder
            order = rng.permutation(len(x_all))
            for i in range(0, len(order), BATCH):
                ae_step(tf.constant(x_all[order[i:i + BATCH]]))
                steps += 1
        if len(x_lab) > 0:                                  # stage 2: encoder + classifier
            for _ in range(LOCAL_EPOCHS):
                order = rng.permutation(len(x_lab))
                for i in range(0, len(order), BATCH):
                    clf_step(tf.constant(x_lab[order[i:i + BATCH]]), tf.constant(y_lab[order[i:i + BATCH]]))
                    steps += 1
        return steps

    return train_client


def aggregate(global_w, client_ws, n_k, tau_k, aggregator):
    p = np.array(n_k, dtype=np.float64) / np.sum(n_k)
    if aggregator == "FedNova":
        tau_eff = float(np.sum(p * np.array(tau_k)))
        return [g - tau_eff * sum(pk * (g - w[i]) / tk for pk, w, tk in zip(p, client_ws, tau_k))
                for i, g in enumerate(global_w)]
    return [sum(pk * w[i] for pk, w in zip(p, client_ws)).astype(np.float32) for i in range(len(global_w))]


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
    if _STOP_AFTER_ROUNDS and _rounds_this_session >= _STOP_AFTER_ROUNDS:
        raise StopForNow()
    if time.time() - SESSION_START + 1.3 * last_round_seconds > TIME_BUDGET_HOURS * 3600:
        raise StopForNow()


# =============================== EVALUATION ===============================
def predict(ms, x):
    return ms.clf.predict(x, batch_size=2048, verbose=0)


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
    path = os.path.join(OUT_DIR, "results", "all_runs.csv")
    pd.DataFrame([row]).to_csv(path, mode='a', header=not os.path.exists(path), index=False)


# =============================== ONE RUN ===============================
def run_one(run_id, variant, aggregator, kind, alpha, lpc, seed, data, plat):
    global _rounds_this_session
    X_tr, y_tr, x_val, y_val, x_te, y_te, test_idx, n_features, n_classes = data
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    local = ModelSet(n_features, n_classes, variant)
    glob = ModelSet(n_features, n_classes, variant)
    prev_enc = build_models(n_features, n_classes, variant)[0]
    train_client = make_trainer(local, glob, prev_enc, aggregator)

    # clients: partition, labeled flows (same labeled subset as the centralized runs)
    parts = partition_clients(y_tr, kind, alpha, seed, NUM_CLIENTS)
    labeled = np.zeros(len(y_tr), dtype=bool)
    labeled[sample_labeled(y_tr, lpc, seed)] = True
    y_oh = tf.keras.utils.to_categorical(y_tr, n_classes).astype('float32')
    clients = [(X_tr[idx], X_tr[idx[labeled[idx]]], y_oh[idx[labeled[idx]]]) for idx in parts]
    client_classes = pd.DataFrame([np.bincount(y_tr[idx], minlength=n_classes) for idx in parts],
                                  columns=LABELS, index=[f"client_{k}" for k in range(NUM_CLIENTS)])
    client_classes["labeled"] = [len(c[1]) for c in clients]
    n_k = [len(idx) for idx in parts]

    n_params = local.n_params()
    bytes_per_round = NUM_CLIENTS * (2 * 4 * n_params + (8 if aggregator == "FedNova" else 0))
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
        glob.set_w(global_w)
        client_ws, tau_k = [], []
        for k, (x_all, x_lab, y_lab) in enumerate(clients):
            tf.keras.utils.set_random_seed(seed * 100000 + rnd * 100 + k)
            rng = np.random.default_rng(seed * 100000 + rnd * 100 + k)
            local.set_w(global_w)
            if aggregator == "MOON":
                prev_enc.set_weights(prev.get(k, glob.enc.get_weights()))
            tau_k.append(max(train_client(x_all, x_lab, y_lab, rng), 1))
            client_ws.append(local.get_w())
            if aggregator == "MOON":
                prev[k] = local.enc.get_weights()
        global_w = aggregate(global_w, client_ws, n_k, tau_k, aggregator)
        glob.set_w(global_w)
        vm, _ = evaluate(predict(glob, x_val), y_val)
        sec = time.time() - t0
        meta["hist"].append({"round": rnd, "val_acc": round(vm["acc"], 6), "val_macro_f1": round(vm["macro_f1"], 6),
                             "comm_MB_cum": round(rnd * bytes_per_round / 1e6, 3), "seconds": round(sec, 1)})
        meta.update(round_done=rnd, last_round_seconds=sec, train_seconds=meta["train_seconds"] + sec)
        print(f"  round {rnd}/{ROUNDS}: val acc {vm['acc']:.4f} | val macro-F1 {vm['macro_f1']:.4f} | {sec:.0f}s")
        save_ckpt(run_id, meta, global_w, local, prev)
        _rounds_this_session += 1

    glob.set_w(global_w)
    prob = predict(glob, x_te)
    m, pred = evaluate(prob, y_te)
    hist = pd.DataFrame(meta["hist"])
    r90 = hist.loc[hist.val_acc >= 0.90, "round"]
    per_class_recall = confusion_matrix(y_te, pred, labels=list(range(n_classes))).diagonal() / np.bincount(y_te, minlength=n_classes)
    row = dict(run_id=run_id, dataset=DATASET, setting="federated", method=MODEL_NAMES[variant], variant=variant,
               aggregator=aggregator, partition=kind, alpha=alpha, clients=NUM_CLIENTS, labels_per_class=lpc,
               labels_total=int(labeled.sum()), seed=seed, rounds=ROUNDS, local_epochs=LOCAL_EPOCHS, batch=BATCH, lr=LR,
               **{k: round(float(v), 6) for k, v in m.items()},
               min_class_recall=round(float(per_class_recall.min()), 6),
               rounds_to_90=int(r90.iloc[0]) if len(r90) else -1,
               params_sent=n_params, MB_per_round=round(bytes_per_round / 1e6, 3),
               MB_total=round(ROUNDS * bytes_per_round / 1e6, 1), smallest_client=int(min(n_k)),
               largest_client=int(max(n_k)), train_seconds=round(meta["train_seconds"], 1), platform=plat,
               gpu=gpu_name(), tf_version=tf.__version__, timestamp=time.strftime("%Y-%m-%d %H:%M"))
    save_run(run_id, row, y_te, pred, prob, test_idx, meta, client_classes)
    os.remove(ckpt_path(run_id))
    print(f"Done {run_id}: test acc {m['acc']:.4f} | macro-F1 {m['macro_f1']:.4f} | "
          f"lowest class recall {per_class_recall.min():.3f}")


# =============================== MAIN ===============================
def main():
    plat = platform_name()
    restore_previous_output()
    tf.config.experimental.enable_op_determinism()
    X, y, n_features = load_data()
    tr, va, te = get_split(y)
    data = (X[tr], y[tr], X[va], y[va], X[te], y[te], te, n_features, len(LABELS))
    print(f"Train {len(tr)} | Val {len(va)} | Test {len(te)} | {plat} | {gpu_name()}")
    print(f"Federated ablation: variants {VARIANTS} | aggregators {AGGREGATORS} | partitions {PARTITIONS} | "
          f"labels/class {LABELED_PER_CLASS_LIST} | K={NUM_CLIENTS} | rounds {ROUNDS} | seeds {SEEDS}")
    try:
        for seed in SEEDS:                  # seed outermost: complete seed sets come first
            for kind, alpha in PARTITIONS:
                for lpc in LABELED_PER_CLASS_LIST:
                    for aggregator in AGGREGATORS:
                        for variant in VARIANTS:
                            part = "iid" if kind == "iid" else f"a{alpha}"
                            run_id = (f"{DATASET}_{MODEL_NAMES[variant]}_{aggregator}_{part}_K{NUM_CLIENTS}"
                                      f"_L{lpc}_s{seed}")
                            if already_done(run_id):
                                print("Skip (already done):", run_id)
                                continue
                            print(f"\n===== {run_id} =====")
                            run_one(run_id, variant, aggregator, kind, alpha, lpc, seed, data, plat)
    except StopForNow:
        print("\n" + "=" * 70)
        print("STOPPED SAFELY (time budget). All progress is saved in:", OUT_DIR)
        print("Set RESUME_FROM to this output folder and run again to continue.")
        print("=" * 70)
        return

    res = pd.read_csv(os.path.join(OUT_DIR, "results", "all_runs.csv"))
    print("\nALL RUNS FINISHED. Mean ± std over seeds (test set):")
    print(res.groupby(['method', 'aggregator', 'partition', 'alpha', 'labels_per_class'])
          [['acc', 'macro_f1', 'min_class_recall', 'rounds_to_90', 'MB_total']].agg(['mean', 'std']).round(4))


if __name__ == "__main__":
    main()
