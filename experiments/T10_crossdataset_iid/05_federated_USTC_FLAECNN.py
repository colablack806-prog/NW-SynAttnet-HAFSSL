# -*- coding: utf-8 -*-
"""
USTC-TFC2016, 20 classes; corrected paired IEEE TAI R2 protocol.
Model: FL-AECNN. Both scripts use the SAME data preprocessing, split, labeled
examples, client partition, aggregation, epochs, and evaluation settings.

MAIN EXPERIMENT: 2,000 labeled training examples TOTAL = 100 per class GLOBALLY;
10 clients; Dirichlet alpha=0.5; 50 FedAvg rounds; 10 independent paired seeds.
Training matches the manuscript's TWO-stage objective: 5 epochs local AE
pretraining on unlabeled samples, then 5 epochs supervised encoder+head.
The former 'stage 1.5' (encoder trained through a frozen random classifier head)
is DISABLED; enable only as a separately documented sensitivity experiment.

Baseline DEFAULT retains its original NO-dropout classifier head. An optional
MATCH_HEAD_DROPOUT=True controlled variant adds the same head dropout as
HAFSSL; report it separately from the published baseline. The attention
block includes MHA, SE, FFN and LayerNorm: neither contrast alone isolates
MHA or SE.

The three source train/valid/test files are concatenated, exact duplicate
payload vectors/contradictory labels removed, and a new stratified 81/9/10
split created with seed 2026. Raw source partitions are NOT retained.
Publish this preprocessing choice and the dataset fingerprint. No packet-
source/session IDs are available to rule out leakage across related flows.

Kaggle: run each script separately; keep result folders. Both write frozen
split fingerprints and per-seed client/labeled indices for paired auditing.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import json, time, shutil, hashlib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             balanced_accuracy_score, confusion_matrix, classification_report)

# =============================== CONFIG ===============================
MODEL     = "FL-AECNN"         # this file: attention-free baseline
BASE_PATH = "data/USTC-TFC2016/"   # folder with the 6 .npy files
DATASET   = "USTC"
MATCH_HEAD_DROPOUT = True  # False: original FL-AECNN; True: optional controlled head-matched variant
LABELS    = ['BitTorrent', 'Facetime', 'FTP', 'Gmail', 'MySQL', 'Outlook', 'Skype', 'SMB', 'Weibo',
             'WorldOfWarcraft', 'Cridex', 'Geodo', 'Htbot', 'Miuref', 'Neris', 'Nsis-ay', 'Shifu',
             'Tinba', 'Virut', 'Zeus']
OUT_DIR   = ""  # set after stage15 and baseline head options

AGGREGATORS  = ["FedAvg"]
ALPHA        = 0.5
NUM_CLIENTS  = 10
ROUNDS       = 50
LOCAL_EPOCHS = 5
STAGE15_EPOCHS = 0   # MAIN manuscript protocol: TWO stages only
STAGE15_LR = 1e-4  # optional Stage-1.5 ablation, off for main protocol
HEAD_TAG = "MatchedHead" if MATCH_HEAD_DROPOUT else "OriginalHead"
OUT_DIR = f"/kaggle/working/FED_USTC_FLAECNN_L2000_S15{STAGE15_EPOCHS}_{HEAD_TAG}_v2"
BATCH_AE, BATCH_CLF = 32, 256
LR = 1e-3
LABELS_PER_CLASS_GLOBAL = 100  # 20 classes x 100 = 2000 labels TOTAL
MIN_CLIENT_FLOWS = 50
SEEDS = list(range(10))  # 10 paired independent retrainings; first three seeds still 0,1,2
MAX_LEN = 128

LATENT_DIM = 39
PROTOCOL_VERSION = f"USTC_GBL2000_S15{STAGE15_EPOCHS}_{HEAD_TAG}_v2"
SPLIT_SEED        = 2026
RESUME_FROM       = ""      # Kaggle: previous version's output folder, to continue after a stop
TIME_BUDGET_HOURS = 11.0
QUICK_TEST        = False  # True = 2 rounds, 1 epoch, seed 0

if QUICK_TEST:
    ROUNDS, LOCAL_EPOCHS, SEEDS = 2, 1, [0]
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


# =============================== DATA ===============================
def hex_to_tokens(hex_str, max_len=MAX_LEN):
    """Same conversion as the original script: 4-hex-digit words -> [0, 1], padded/cut to max_len."""
    ids = [int(t, 16) for t in str(hex_str).strip().split() if len(t) == 4][:max_len]
    ids += [0] * max(0, max_len - len(ids))
    return np.array(ids, dtype=np.float32) / 65535.0


def load_data():
    parts_x, parts_y = [], []
    for name in ("train", "valid", "test"):                 # pool the distributed files, then re-split
        parts_x.append(np.load(os.path.join(BASE_PATH, f"x_payload_{name}.npy"), allow_pickle=True))
        parts_y.append(np.load(os.path.join(BASE_PATH, f"y_{name}.npy"), allow_pickle=True))
    X = np.stack([hex_to_tokens(s) for s in np.concatenate(parts_x)])
    y = np.concatenate(parts_y).astype(int)
    n0 = len(y)
    df = pd.DataFrame(X)
    key = pd.util.hash_pandas_object(df, index=False).values
    labels_per_key = pd.Series(y).groupby(key).nunique()
    ambiguous = np.isin(key, labels_per_key[labels_per_key > 1].index.values)   # same vector, several labels
    keep = (~ambiguous) & (~pd.Series(key).duplicated().values)
    X, y = X[keep], y[keep]
    print(f"Rows: {n0} -> {len(y)} after removing {int(ambiguous.sum())} rows with ambiguous labels "
          f"and {int((~ambiguous).sum() - keep.sum())} exact duplicates")
    print("Rows per class:", dict(zip(*np.unique(y, return_counts=True))))
    return X.reshape(-1, MAX_LEN, 1), y, MAX_LEN


def data_fingerprint(X, y):
    """SHA-256 of processed data, so different preprocessing cannot share a split."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X).tobytes())
    h.update(np.ascontiguousarray(y).tobytes())
    return h.hexdigest()


def create_and_verify_manifest(fingerprint):
    """Fail closed if a previous output dir used different model or settings."""
    global CONFIG_SHA256
    cfg = {
        "model": MODEL, "protocol_version": PROTOCOL_VERSION, "dataset_sha256": fingerprint,
        "label_names": LABELS, "labels_per_class_global": LABELS_PER_CLASS_GLOBAL,
        "partition": "Dirichlet-class-skew", "alpha": ALPHA,
        "clients": NUM_CLIENTS, "min_client_flows": MIN_CLIENT_FLOWS,
        "train_valid_test_split_seed": SPLIT_SEED,
        "rounds": ROUNDS, "local_epochs": LOCAL_EPOCHS, "stage15_epochs": STAGE15_EPOCHS,
        "stage15_lr": STAGE15_LR, "batch_ae": BATCH_AE, "batch_clf": BATCH_CLF,
        "learning_rate": LR, "latent_dim": LATENT_DIM, "max_input_words": MAX_LEN,
        "agg": AGGREGATORS, "reporting": "test-at-final-round; val-for-rounds-to-90",
        "communication": "full-model encoder+decoder+head, float32, upload+download",
    }
    cfg.update(classifier_dropout=0.3 if MATCH_HEAD_DROPOUT else 0.0,
               match_head_dropout=MATCH_HEAD_DROPOUT)
    CONFIG_SHA256 = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    manifest_path = os.path.join(OUT_DIR, "protocol_manifest.json")
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            saved = json.load(f)
        if saved != cfg:
            raise ValueError("Experiment config/data changed. Use a new OUT_DIR; do not reuse checkpoints.")
    else:
        if os.listdir(OUT_DIR):
            raise ValueError("Existing results lack protocol manifest. Use a fresh OUT_DIR.")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, sort_keys=True)
    print("Experiment configuration SHA-256:", CONFIG_SHA256)


def get_split(y, fingerprint):
    path = os.path.join(OUT_DIR, "splits", f"{DATASET}_split.npz")
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as s:
            assert int(s['n_rows']) == len(y), "Split size does not match input."
            assert str(s['data_sha256']) == fingerprint, "Split DATA HASH mismatch."
            assert str(s['protocol_version']) == PROTOCOL_VERSION, "Split protocol mismatch."
            train, val, test = (s['train'], s['val'], s['test'])
    else:
        idx = np.arange(len(y))
        trval, test = train_test_split(idx, test_size=0.10, stratify=y, random_state=SPLIT_SEED)
        train, val = train_test_split(trval, test_size=0.10, stratify=y[trval], random_state=SPLIT_SEED)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, train=train, val=val, test=test, n_rows=len(y),
                            data_sha256=fingerprint, protocol_version=PROTOCOL_VERSION)
    full = np.concatenate((train, val, test))
    assert len(full) == len(y) and np.array_equal(np.sort(full), np.arange(len(y))), \
           "Train/validation/test indices overlap or omit rows"
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
    """Exactly `per_class` labeled training examples per class, GLOBALLY.

    Same seed and same client partitions -> same selection in BOTH scripts.
    Labels cannot be drawn from validation/test; y_train contains train only.
    """
    rng = np.random.default_rng(20_000 + seed)
    labeled = np.zeros(len(y_train), dtype=bool)
    eligible = np.concatenate(parts)
    assert np.array_equal(np.sort(eligible), np.arange(len(y_train))), \
           "Client partitions must cover each train example exactly once."
    for c in np.unique(y_train):
        pool = eligible[y_train[eligible] == c]
        if len(pool) < per_class:
            raise ValueError(f"Class {int(c)} has only {len(pool)} train samples; need {per_class}.")
        labeled[rng.choice(pool, size=per_class, replace=False)] = True
    assert int(labeled.sum()) == int(per_class * len(np.unique(y_train)))
    return labeled


def save_client_audit(run_id, parts, labeled, y_train, fingerprint):
    """Publicly inspectable proof of per-seed partitions and selected labels."""
    path = os.path.join(OUT_DIR, "splits", f"{run_id}_selection.npz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrays = {f"client_{k}_train_indices": idx for k, idx in enumerate(parts)}
    arrays.update(selected_label_indices=np.flatnonzero(labeled),
                  selected_labels=y_train[labeled],
                  data_sha256=fingerprint, protocol_version=PROTOCOL_VERSION)
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as saved:
            for key, value in arrays.items():
                assert np.array_equal(saved[key], value), f"Resumed selection mismatch: {key}"
    else:
        np.savez_compressed(path, **arrays)


# =============================== MODEL (FL-AECNN, no attention) ===============================
def build_models(n_features, n_classes):
    ie = layers.Input(shape=(n_features, 1))
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(ie)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D()(x)
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

    ic = layers.Input(shape=(LATENT_DIM,))
    c = layers.Reshape((steps, ch))(layers.Dense(steps * ch)(ic))
    c = layers.Conv1D(64, 3, padding='same', activation='relu')(c)
    c = layers.Conv1D(64, 3, padding='same', activation='relu')(c)
    c = layers.MaxPooling1D(2)(c)
    c = layers.Conv1D(128, 3, padding='same', activation='relu')(c)
    c = layers.Conv1D(128, 3, padding='same', activation='relu')(c)
    c = layers.MaxPooling1D(2)(c)
    c = layers.Dense(128, activation='relu')(layers.Flatten()(c))
    if MATCH_HEAD_DROPOUT:
        c = layers.Dropout(0.3)(c)  # optional controlled match to HAFSSL head
    head = Model(ic, layers.Dense(n_classes, activation='softmax')(c), name='classifier_head')

    inp = layers.Input(shape=(n_features, 1))
    return enc, dec, head, Model(inp, dec(enc(inp))), Model(inp, head(enc(inp)))


class ModelSet:
    def __init__(self, n_features, n_classes):
        self.enc, self.dec, self.head, self.ae, self.clf = build_models(n_features, n_classes)
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
        if "learning_rate" not in str(getattr(v, "path", getattr(v, "name", ""))):
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
            if STAGE15_EPOCHS > 0:  # optional, NOT used in manuscript main experiment
                steps += run_epochs(stage15_step, x_lab, y_lab, STAGE15_EPOCHS, BATCH_CLF, rng)
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
def run_one(run_id, aggregator, seed, data, plat, runs_left):
    global _rounds_this_session
    X_tr, y_tr, x_val, y_val, x_te, y_te, test_idx, n_features, n_classes = data
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    local, glob = ModelSet(n_features, n_classes), ModelSet(n_features, n_classes)
    train_client = make_trainer(local)

    parts = partition_clients(y_tr, ALPHA, seed, NUM_CLIENTS, min_size=MIN_CLIENT_FLOWS)
    labeled = label_clients(parts, y_tr, LABELS_PER_CLASS_GLOBAL, seed)
    save_client_audit(run_id, parts, labeled, y_tr, DATA_SHA256)
    y_oh = tf.keras.utils.to_categorical(y_tr, n_classes).astype('float32')
    clients = [(X_tr[idx[~labeled[idx]]], X_tr[idx[labeled[idx]]], y_oh[idx[labeled[idx]]]) for idx in parts]
    active = [k for k, idx in enumerate(parts) if len(idx) >= MIN_CLIENT_FLOWS]
    client_table = pd.DataFrame([np.bincount(y_tr[idx], minlength=n_classes) for idx in parts],
                                columns=LABELS, index=[f"client_{k}" for k in range(NUM_CLIENTS)])
    client_table["labeled"] = [len(c[1]) for c in clients]
    client_table["active"] = [k in active for k in range(NUM_CLIENTS)]
    print(client_table[["labeled", "active"]].assign(flows=[len(p) for p in parts]).T.to_string())

    n_params = local.n_params()
    global_w = local.get_w()
    payload_bytes = sum(w.nbytes for w in global_w)  # exact float dtype of model tensors
    assert n_params == sum(w.size for w in global_w)
    bytes_per_round = len(active) * 2 * payload_bytes  # full AE+head, upload+download
    assert len(active) == NUM_CLIENTS, "Inactive client changes the supervision budget"
    meta = {"round_done": 0, "hist": [], "train_seconds": 0.0, "last_round_seconds": 0.0,
            "protocol_version": PROTOCOL_VERSION, "data_sha256": DATA_SHA256, "config_sha256": CONFIG_SHA256}
    if os.path.exists(ckpt_path(run_id)):
        meta, global_w = load_ckpt(run_id, local, len(global_w))
        assert meta.get("protocol_version") == PROTOCOL_VERSION
        assert meta.get("data_sha256") == DATA_SHA256
        assert meta.get("config_sha256") == CONFIG_SHA256
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
        eta = sec * ((ROUNDS - rnd) + ROUNDS * (runs_left - 1)) / 3600
        print(f"  round {rnd}/{ROUNDS}: val acc {vm['acc']:.4f} | val macro-F1 {vm['macro_f1']:.4f} | {sec:.0f}s"
              f" | about {eta:.1f} h left for all remaining runs")
        save_ckpt(run_id, meta, global_w, local)
        _rounds_this_session += 1

    glob.set_w(global_w)
    prob = glob.clf.predict(x_te, batch_size=2048, verbose=0)
    m, pred = evaluate(prob, y_te)
    recall = confusion_matrix(y_te, pred, labels=list(range(n_classes))).diagonal() / np.maximum(
        np.bincount(y_te, minlength=n_classes), 1)
    hist = pd.DataFrame(meta["hist"])
    r90 = hist.loc[hist.val_acc >= 0.90, "round"]
    row = dict(run_id=run_id, dataset=DATASET, setting="federated", method=MODEL, aggregator=aggregator,
               partition="dirichlet", alpha=ALPHA, clients=NUM_CLIENTS, active_clients=len(active),
               labels_per_class_global=LABELS_PER_CLASS_GLOBAL, labels_total=int(labeled.sum()),
                protocol_version=PROTOCOL_VERSION, data_sha256=DATA_SHA256, config_sha256=CONFIG_SHA256,
                dropout_head=0.3 if MATCH_HEAD_DROPOUT else 0.0, full_model_communication=True,
               seed=seed, rounds=ROUNDS, local_epochs=LOCAL_EPOCHS, stage15_epochs=STAGE15_EPOCHS,
               batch_ae=BATCH_AE, batch_clf=BATCH_CLF, lr=LR,
               **{k: round(float(v), 6) for k, v in m.items()},
               min_class_recall=round(float(recall.min()), 6),
               classes_below_50=int((recall < 0.5).sum()),
               **{f"recall_{name}": round(float(r), 6) for name, r in zip(LABELS, recall)},
               rounds_to_90=int(r90.iloc[0]) if len(r90) else -1, params_sent=n_params,
               MB_per_round=round(bytes_per_round / 1e6, 3), MB_total=round(ROUNDS * bytes_per_round / 1e6, 1),
               train_seconds=round(meta["train_seconds"], 1), platform=plat, gpu=gpu_name(),
               tf_version=tf.__version__, timestamp=time.strftime("%Y-%m-%d %H:%M"))
    save_run(run_id, row, y_te, pred, prob, test_idx, meta["hist"], client_table)
    os.remove(ckpt_path(run_id))
    print(f"Done {run_id}: test acc {m['acc']:.4f} | macro-F1 {m['macro_f1']:.4f} | "
          f"classes below 50% recall: {int((recall < 0.5).sum())}")


# =============================== MAIN ===============================
def main():
    global DATA_SHA256
    plat = platform_name()
    restore_previous_output()
    tf.config.experimental.enable_op_determinism()
    X, y, n_features = load_data()
    DATA_SHA256 = data_fingerprint(X, y)
    assert np.array_equal(np.unique(y), np.arange(len(LABELS))), "Labels must be indices 0..19"
    assert len(np.unique(y)) == len(LABELS), f"Found {len(np.unique(y))} classes, LABELS has {len(LABELS)}"
    create_and_verify_manifest(DATA_SHA256)
    tr, va, te = get_split(y, DATA_SHA256)
    data = (X[tr], y[tr], X[va], y[va], X[te], y[te], te, n_features, len(LABELS))
    print(f"Train {len(tr)} | Val {len(va)} | Test {len(te)} | {plat} | {gpu_name()}")
    print("Processed dataset SHA-256:", DATA_SHA256)
    print(f"MODEL {MODEL} | {DATASET} | alpha {ALPHA} | K={NUM_CLIENTS} | rounds {ROUNDS} | local epochs {LOCAL_EPOCHS} "
          f"(optional stage 1.5 epochs: {STAGE15_EPOCHS}) | GLOBAL labels: {LABELS_PER_CLASS_GLOBAL}/class = 2000 TOTAL | "
          f"aggregators {AGGREGATORS} | seeds {SEEDS}")
    jobs = [(s, a) for s in SEEDS for a in AGGREGATORS]
    try:
        for i, (seed, aggregator) in enumerate(jobs):
            run_id = f"{DATASET}_{MODEL}_{aggregator}_a{ALPHA}_K{NUM_CLIENTS}_L2000_S15{STAGE15_EPOCHS}_{HEAD_TAG}_v2_s{seed}"
            if already_done(run_id):
                print("Skip (already done):", run_id)
                continue
            print(f"\n===== {run_id} =====")
            run_one(run_id, aggregator, seed, data, plat, len(jobs) - i)
    except StopForNow:
        print("\n" + "=" * 70)
        print("STOPPED SAFELY (time budget). All progress is saved in:", OUT_DIR)
        print("Kaggle: add this version's output as input, set RESUME_FROM, Save & Run All.")
        print("=" * 70)
        return
    res = pd.read_csv(os.path.join(OUT_DIR, "results", "all_runs.csv"))
    print("\nALL RUNS FINISHED. Mean ± std over seeds (test set):")
    print(res.groupby(['method', 'aggregator'])[['acc', 'macro_f1', 'min_class_recall', 'classes_below_50']]
          .agg(['mean', 'std']).round(4))


if __name__ == "__main__":
    main()
