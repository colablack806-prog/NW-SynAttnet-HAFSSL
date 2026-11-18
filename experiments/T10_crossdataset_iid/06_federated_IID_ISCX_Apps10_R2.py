# -*- coding: utf-8 -*-
"""
HAFSSL vs FL-AECNN with IID clients on ISCX-VPN2016 AND 10 Popular Apps - 3 seeds each, paired tests (R2 revision).

12 runs in total: 2 datasets x 2 models x 3 seeds (0, 1, 2).

PROTOCOL = the federated protocol of the paper (Table I / Section V-A), with ONE change: IID clients.
  10 IID clients (training flows shuffled with the seed and cut into 10 equal parts), 50 rounds, FedAvg.
  Every round, on each client: autoencoder 5 epochs, then encoder + classifier 5 epochs,
  batch 256 in both stages, Adam lr 1e-3.
  Labels: 500 labeled flows per client, equal per class
     ISCX-VPN2016    : 100 per class per client = 1,000 per class in total (as in Tables V and VI)
     10 Popular Apps :  50 per class per client =   500 per class in total (as in Table VIII)
  Dropout: 0.1 inside the attention block (Table I); no dropout in the classifier head.
     This is what the ISCX-VPN2016 and 10 Popular Apps federated runs use. The 0.3 head dropout belongs
     ONLY to the USTC-TFC2016 runs, so it is not used here.

MODELS (same layers as Tables V-VIII)
  FL-AECNN : no attention                                    ISCX 384,589 | Apps10 385,234 parameters sent
  HAFSSL   : MHA (4 heads x key_dim 32) -> SE (ratio 16) -> residual + LayerNorm -> FFN (128, 64)
             after the first MaxPool of the encoder        ISCX 435,217 | Apps10 435,862 parameters sent
  On 10 Popular Apps this is the Table I attention configuration (4 heads, key 32, SE 16).

SAME AS EVERY OTHER R2 EXPERIMENT
  duplicates removed before splitting; frozen split (seed 2026), the same test set as Table II;
  each seed fixes the clients, the labeled flows, the initial weights and the batch order, and both models
  get the same clients and labeled flows (checked); the global model after round 50 is tested once;
  the validation set is only monitored; macro metrics with zero_division=0; McNemar + randomization test,
  Holm correction, Wilcoxon across seeds (reported, cannot go below 0.25 with 3 seeds).

TIME (Kaggle T4, from the measured cost of Table IX): about 3.5 h for ISCX-VPN2016 and 7 h for 10 Popular Apps.
  Fastest: run two notebooks at the same time, one with DATASETS = ["ISCX"], one with DATASETS = ["Apps10"].
  One notebook with both datasets also works: it stops safely before 11 h and continues after a restart.

HOW TO RUN ON KAGGLE
  1. Add both CSV files as input (the script also searches /kaggle/input for the file names).
  2. QUICK_TEST = True -> Save & Run All (a few minutes). Check the log. Then QUICK_TEST = False.
  3. If the log ends with "STOPPED SAFELY": Add Input -> this notebook's output, keep
     RESUME_FROM = ["/kaggle/input"], Save & Run All again. Finished runs are skipped,
     an interrupted run continues from its last saved round.
OUTPUT (in /kaggle/working), per dataset:
  FED_<dataset>_iid_K10_L500_<model>/ (config.json, results/all_runs.csv, preds/, reports/, splits/),
  summary_<dataset>_iid_K10_L500.csv and paired_tests_<dataset>_iid_K10_L500/ (report.md, per_seed.csv, ...).
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import json, time, shutil, zlib, warnings, glob
import numpy as np
import pandas as pd
import scipy
import tensorflow as tf
from tensorflow.keras import layers, Model
from scipy.stats import binomtest, wilcoxon
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             balanced_accuracy_score, confusion_matrix, classification_report)

# =============================== CONFIG ===============================
DATASETS = ["ISCX", "Apps10"]   # both, one after the other; or ["ISCX"] / ["Apps10"] in two notebooks at once

DATASET_CFG = {
    "ISCX": dict(
        path="data/ISCX-VPN2016/ISCX_5class_each_normalized_cuttedfloefeature.csv",
        labels=['chat', 'email', 'file', 'streaming', 'voip'],
        expected_rows=82435, expected_split=(66771, 7420, 8244),
        params={"FL-AECNN": 384589, "HAFSSL": 435217}),
    "Apps10": dict(
        path="data/10PopularApps/pcapdroid_10class_each_normalized_cuttedfloefeature.csv",
        # names in the order of the label codes 0..9 (must match the centralized Table II script)
        labels=['TikTok', 'iQiyi Video', 'Jingdong Shopping', 'Snack Video', 'QQ Music',
                'QQ', 'Taobao Shopping', 'NetEase Cloud Music', 'Arena of Valor', 'WeChat'],
        expected_rows=170459, expected_split=(138071, 15342, 17046),
        params={"FL-AECNN": 385234, "HAFSSL": 435862}),
}

PARTITION   = "iid"      # "iid" or "dirichlet" (non-IID, uses ALPHA)
ALPHA       = 0.5        # only used when PARTITION = "dirichlet"
NUM_CLIENTS = 10
ROUNDS      = 50
AE_EPOCHS   = 5
STAGE15_EPOCHS, STAGE15_LR = 0, 1e-4   # 0 = no stage 1.5 (as in Tables V-VIII)
CLF_EPOCHS  = 5
BATCH_AE, BATCH_CLF = 256, 256         # Table I: batch 256 in both stages
LR = 1e-3
LABELS_PER_CLIENT = 500  # labeled flows per client, equal per class (ISCX 100/class, Apps10 50/class)
MIN_CLIENT_FLOWS  = 50
SEEDS = [0, 1, 2]
AGGREGATORS = ["FedAvg"]

NUM_HEADS, KEY_DIM, SE_RATIO, LATENT_DIM = 4, 32, 16, 39
ATT_DROPOUT  = 0.1       # dropout inside the attention block (Table I)
HEAD_DROPOUT = 0.0       # classifier head: no dropout, as in the ISCX / 10 Popular Apps federated runs

# The two models; "params" is filled in per dataset by configure()
MODELS = {
    "FL-AECNN": dict(method="FL-AECNN", enc_att=False, tag="original", params=None),
    "HAFSSL":   dict(method="HAFSSL",   enc_att=True,  tag="standard", params=None),
}
MODELS_TO_RUN = ["FL-AECNN", "HAFSSL"]

OUT_ROOT    = "/kaggle/working"
RESUME_FROM = ["/kaggle/input"]        # where an earlier version's output is searched (recursively)

SPLIT_SEED        = 2026    # same as every other experiment = same test set
TIME_BUDGET_HOURS = 11.0
QUICK_TEST        = False  # True = 2 rounds, 1 epoch, seed 0, separate "_quicktest" folders (a few minutes)

# Paired tests, run automatically when all runs of a dataset are finished (protocol fixed before the results)
RUN_PAIRED_TESTS = True
TEST_COMPARISONS = [("main", "FL-AECNN", "HAFSSL")]   # (family, baseline A, compared model B)
SIG_LEVEL, N_PERM, N_BOOT, TEST_SEED = 0.05, 10_000, 10_000, 2026

if isinstance(RESUME_FROM, str):
    RESUME_FROM = [RESUME_FROM] if RESUME_FROM else []
assert set(MODELS_TO_RUN) <= set(MODELS), f"MODELS_TO_RUN must only use the names {list(MODELS)}"
assert set(DATASETS) <= set(DATASET_CFG), f"DATASETS must only use the names {list(DATASET_CFG)}"
assert PARTITION in ("iid", "dirichlet"), 'PARTITION must be "iid" or "dirichlet"'
if QUICK_TEST:
    ROUNDS, AE_EPOCHS, CLF_EPOCHS, SEEDS = 2, 1, 1, [0]
    STAGE15_EPOCHS = min(STAGE15_EPOCHS, 1)
FOLDER_SUFFIX = "_quicktest" if QUICK_TEST else ""
SESSION_START = time.time()
_STOP_AFTER_ROUNDS = int(os.environ.get("STOP_AFTER_ROUNDS", "0"))   # testing only
_rounds_this_session = 0

# Set per dataset by configure()
DATASET = DATA_PATH = None
LABELS, EXPECTED_ROWS, EXPECTED_SPLIT = [], None, None
LABELS_PER_CLASS_PER_CLIENT, SETTING_TAG = None, None

MANIFEST_KEYS = ["DATA_PATH", "DATASET", "LABELS", "PARTITION", "ALPHA", "NUM_CLIENTS", "ROUNDS", "AE_EPOCHS",
                 "STAGE15_EPOCHS", "STAGE15_LR", "CLF_EPOCHS", "BATCH_AE", "BATCH_CLF", "LR", "LABELS_PER_CLIENT",
                 "MIN_CLIENT_FLOWS", "SEEDS", "AGGREGATORS", "NUM_HEADS", "KEY_DIM", "SE_RATIO", "LATENT_DIM",
                 "ATT_DROPOUT", "HEAD_DROPOUT", "SPLIT_SEED"]
MUST_MATCH = [k for k in MANIFEST_KEYS if k not in ("DATA_PATH", "AGGREGATORS", "SEEDS")]   # for reused runs


def find_data(path):
    if os.path.exists(path):
        return path
    name = os.path.basename(path)
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True)) + sorted(glob.glob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found. Add it as input or fix its path in DATASET_CFG.")
    print("Data file found at", hits[0])
    return hits[0]


def configure(ds):
    """Switch every dataset-dependent setting to dataset `ds`."""
    global DATASET, DATA_PATH, LABELS, EXPECTED_ROWS, EXPECTED_SPLIT, LABELS_PER_CLASS_PER_CLIENT, SETTING_TAG
    cfg = DATASET_CFG[ds]
    DATASET, LABELS = ds, list(cfg["labels"])
    DATA_PATH = find_data(cfg["path"])
    EXPECTED_ROWS, EXPECTED_SPLIT = cfg["expected_rows"], cfg["expected_split"]
    assert LABELS_PER_CLIENT % len(LABELS) == 0, "LABELS_PER_CLIENT must split equally over the classes"
    LABELS_PER_CLASS_PER_CLIENT = LABELS_PER_CLIENT // len(LABELS)
    SETTING_TAG = ("iid" if PARTITION == "iid" else f"dir{ALPHA:g}") + f"_K{NUM_CLIENTS}_L{LABELS_PER_CLIENT}"
    for name in MODELS:
        MODELS[name]["params"] = cfg["params"][name]


# =============================== FOLDERS AND NAMES ===============================
def model_dir(name):
    return os.path.join(OUT_ROOT, f"FED_{DATASET}_{SETTING_TAG}_{name}{FOLDER_SUFFIX}")


def run_id_for(name, aggregator, seed):
    spec = MODELS[name]
    tag = "" if spec["tag"] == "original" else f"_{spec['tag']}"
    return f"{DATASET}_{spec['method']}{tag}_{aggregator}_{SETTING_TAG}_s{seed}"


def model_label(name):
    return f"{MODELS[name]['method']} ({MODELS[name]['tag']})"


# =============================== SETUP ===============================
def platform_name():
    if os.path.exists("/kaggle"):
        return "kaggle"
    return "colab" if os.path.exists("/content") else "pc"


def gpu_name():
    gpus = tf.config.list_physical_devices('GPU')
    return tf.config.experimental.get_device_details(gpus[0]).get('device_name', 'GPU') if gpus else "CPU only"


def finished_runs(folder):
    path = os.path.join(folder, "results", "all_runs.csv")
    return len(pd.read_csv(path)) if os.path.exists(path) else 0


def checkpoint_round(folder):
    best, d = 0, os.path.join(folder, "checkpoints")
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.endswith(".npz") and not f.endswith(".tmp.npz"):
                try:
                    best = max(best, json.loads(str(np.load(os.path.join(d, f))["meta"]))["round_done"])
                except Exception:
                    pass
    return best


def check_reused(folder, name):
    """A reused folder must come from the same protocol and contain only this model's runs."""
    spec = MODELS[name]
    with open(os.path.join(folder, "config.json")) as f:
        cfg = json.load(f)
    differ = [k for k in MUST_MATCH if k not in cfg or cfg[k] != globals()[k]]
    if differ:
        raise ValueError(f"Cannot reuse {folder}: settings differ from this notebook in {differ}")
    path = os.path.join(folder, "results", "all_runs.csv")
    if os.path.exists(path):
        runs = pd.read_csv(path)
        bad = runs[(runs["method"] != spec["method"]) | (runs["model_config"] != spec["tag"])
                   | (runs["params_sent"] != spec["params"]) | (runs["rounds"] != ROUNDS)]
        if len(bad):
            raise ValueError(f"Cannot reuse {folder}: these runs are not {name} with this protocol: "
                             f"{bad['run_id'].tolist()}")


def import_previous():
    """Copy each model's folder from RESUME_FROM (searched recursively); the copy with the most progress wins."""
    if QUICK_TEST or not RESUME_FROM:
        return
    roots = [r for r in RESUME_FROM if os.path.isdir(r)]
    for name in MODELS:
        dst = model_dir(name)
        if os.path.isdir(dst) and os.listdir(dst):
            continue
        target = os.path.basename(dst)
        cands = sorted({d for root in roots for d, _, files in os.walk(root)
                        if os.path.basename(d) == target and "config.json" in files})
        if not cands:
            continue
        best = max(cands, key=lambda c: (finished_runs(c), checkpoint_round(c)))
        check_reused(best, name)
        shutil.copytree(best, dst)
        ck = checkpoint_round(best)
        print(f"{name}: reused {finished_runs(best)} finished run(s)"
              + (f" and a checkpoint at round {ck}" if ck else "") + f" from {best}")


def write_manifest(name):
    folder = model_dir(name)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "config.json")
    if os.path.exists(path):              # keep the record of the session that started this model
        return
    spec = MODELS[name]
    cfg = {k: globals()[k] for k in MANIFEST_KEYS}
    cfg.update(MODEL=spec["method"], MODEL_TAG=spec["tag"], encoder_attention=spec["enc_att"],
               classifier_attention=False, tf_version=tf.__version__,
               gpu=gpu_name(), started=time.strftime("%Y-%m-%d %H:%M"))
    with open(path, "w") as f:
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
    print(f"Features: {len(feat_cols)} | value range {X.min():.4f} to {X.max():.4f} | "
          f"rows per class: {dict(zip(*np.unique(y, return_counts=True)))}")
    assert len(feat_cols) == 66, f"Expected 66 flow features, found {len(feat_cols)}"
    assert np.array_equal(np.unique(y), np.arange(len(LABELS))), \
        f"Label codes must be 0..{len(LABELS) - 1}; found {np.unique(y)}"
    if EXPECTED_ROWS:
        assert len(df) == EXPECTED_ROWS, f"De-duplicated rows {len(df)} differ from Table II ({EXPECTED_ROWS})"
    return X, y, len(feat_cols)


def get_split(y):
    """One split for all models: a saved split is reused (all saved copies must agree), else it is created."""
    paths = [os.path.join(model_dir(n), "splits", f"{DATASET}_split.npz") for n in MODELS]
    saved = [p for p in paths if os.path.exists(p)]
    if saved:
        first = np.load(saved[0])
        assert int(first['n_rows']) == len(y), f"Split file does not match this data: {saved[0]}"
        train, val, test = first['train'], first['val'], first['test']
        for p in saved[1:]:
            s = np.load(p)
            assert all(np.array_equal(s[k], v) for k, v in (('train', train), ('val', val), ('test', test))), \
                f"The saved splits differ: {saved[0]} vs {p}"
    else:
        idx = np.arange(len(y))
        trval, test = train_test_split(idx, test_size=0.10, stratify=y, random_state=SPLIT_SEED)
        train, val = train_test_split(trval, test_size=0.10, stratify=y[trval], random_state=SPLIT_SEED)
    for name, p in zip(MODELS, paths):
        if name in MODELS_TO_RUN and not os.path.exists(p):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            np.savez(p, train=train, val=val, test=test, n_rows=len(y))
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


def partition_iid(n_train, seed, n_clients):
    """IID: the training flows shuffled (seeded) and cut into equal parts."""
    rng = np.random.default_rng(10_000 + seed)
    return [np.sort(p) for p in np.array_split(rng.permutation(n_train), n_clients)]


def label_clients(parts, y_train, per_class, seed):
    """Each client labels up to `per_class` flows of every class it holds. Seeded."""
    rng = np.random.default_rng(20_000 + seed)
    labeled = np.zeros(len(y_train), dtype=bool)
    for idx in parts:
        for c in np.unique(y_train[idx]):
            pool = idx[y_train[idx] == c]
            labeled[rng.choice(pool, min(per_class, len(pool)), replace=False)] = True
    return labeled


def check_pairing(name, client_table, seed):
    """Stops the run if its clients or labeled flows differ from another model's run with the same seed."""
    cols = LABELS + ["labeled"]
    for other in MODELS:
        if other == name:
            continue
        for aggregator in AGGREGATORS:
            f = os.path.join(model_dir(other), "reports", f"{run_id_for(other, aggregator, seed)}_clients.csv")
            if os.path.exists(f):
                ref = pd.read_csv(f, index_col=0)
                assert np.array_equal(ref[cols].values.astype(int), client_table[cols].values.astype(int)), \
                    f"Seed {seed}: client partition or labeled flows differ from the {other} run ({f})"
                print(f"  Pairing check passed: same clients and labeled flows as {other} seed {seed}")
                break


# =============================== MODEL ===============================
def se_block(x):
    ch = x.shape[-1]
    s = layers.GlobalAveragePooling1D()(x)
    s = layers.Dense(ch // SE_RATIO, activation='relu')(s)
    s = layers.Dense(ch, activation='sigmoid')(s)
    return layers.Multiply()([x, layers.Reshape((1, ch))(s)])


def hybrid_attention_block(x, key_dim):
    """MHA -> SE, residual, FFN (same block as Tables V-VIII)."""
    a = se_block(layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=key_dim, dropout=ATT_DROPOUT)(x, x))
    x = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, a]))
    f = layers.Dense(key_dim * 4, activation='relu')(x)
    f = layers.Dense(x.shape[-1])(layers.Dropout(ATT_DROPOUT)(f))
    return layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, f]))


def build_models(n_features, n_classes, enc_attention):
    """enc_attention: hybrid attention block in the encoder (HAFSSL). The classifier is the same for both models."""
    ie = layers.Input(shape=(n_features, 1))
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(ie)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D()(x)
    if enc_attention:
        x = hybrid_attention_block(x, KEY_DIM)
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
    if HEAD_DROPOUT > 0:
        c = layers.Dropout(HEAD_DROPOUT)(c)
    head = Model(ic, layers.Dense(n_classes, activation='softmax')(c), name='classifier_head')

    inp = layers.Input(shape=(n_features, 1))
    return enc, dec, head, Model(inp, dec(enc(inp))), Model(inp, head(enc(inp)))


class ModelSet:
    def __init__(self, n_features, n_classes, enc_attention):
        self.enc, self.dec, self.head, self.ae, self.clf = build_models(n_features, n_classes, enc_attention)
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


# =============================== LOCAL TRAINING ===============================
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
        if len(x_unl) >= MIN_CLIENT_FLOWS:                                          # autoencoder
            steps += run_epochs(ae_step, x_unl, None, AE_EPOCHS, BATCH_AE, rng)
        if len(x_lab) > 0:
            if STAGE15_EPOCHS:
                steps += run_epochs(stage15_step, x_lab, y_lab, STAGE15_EPOCHS, BATCH_CLF, rng)
            steps += run_epochs(clf_step, x_lab, y_lab, CLF_EPOCHS, BATCH_CLF, rng)  # encoder + classifier
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


def ckpt_path(name, run_id):
    return os.path.join(model_dir(name), "checkpoints", f"{run_id}.npz")


def save_ckpt(name, run_id, meta, global_w, local):
    path = ckpt_path(name, run_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = {f"g__{i}": w for i, w in enumerate(global_w)}
    arr.update({f"l__{i}": np.asarray(v.numpy()) for i, v in enumerate(local.all_vars())})
    arr["meta"] = np.array(json.dumps(meta))
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arr)
    os.replace(tmp, path)


def load_ckpt(name, run_id, local, n_global):
    d = np.load(ckpt_path(name, run_id), allow_pickle=False)
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
    wp, wr, wf1, _ = precision_recall_fscore_support(y, pred, average='weighted', zero_division=0)
    m = {'acc': accuracy_score(y, pred), 'macro_precision': p, 'macro_recall': r, 'macro_f1': f1,
         'balanced_acc': balanced_accuracy_score(y, pred),
         'weighted_precision': wp, 'weighted_recall': wr, 'weighted_f1': wf1}
    assert abs(m['macro_recall'] - m['balanced_acc']) < 1e-9
    return m, pred


def already_done(name, run_id):
    path = os.path.join(model_dir(name), "results", "all_runs.csv")
    return os.path.exists(path) and run_id in set(pd.read_csv(path)['run_id'])


def save_run(name, run_id, row, y_test, pred, prob, test_idx, hist, client_table):
    folder = model_dir(name)
    for sub in ("results", "preds", "reports"):
        os.makedirs(os.path.join(folder, sub), exist_ok=True)
    np.savez_compressed(os.path.join(folder, "preds", f"{run_id}.npz"),
                        y_true=y_test, y_pred=pred, y_prob=prob, test_idx=test_idx)
    labs = list(range(len(LABELS)))
    rep = classification_report(y_test, pred, labels=labs, target_names=LABELS, digits=4,
                                output_dict=True, zero_division=0)
    pd.DataFrame(rep).transpose().to_csv(os.path.join(folder, "reports", f"{run_id}_per_class.csv"))
    pd.DataFrame(confusion_matrix(y_test, pred, labels=labs), index=LABELS, columns=LABELS).to_csv(
        os.path.join(folder, "reports", f"{run_id}_confusion.csv"))
    pd.DataFrame(hist).to_csv(os.path.join(folder, "reports", f"{run_id}_rounds.csv"), index=False)
    client_table.to_csv(os.path.join(folder, "reports", f"{run_id}_clients.csv"))
    path = os.path.join(folder, "results", "all_runs.csv")
    if os.path.exists(path):                       # append in the column order of the existing file
        cols = list(pd.read_csv(path, nrows=0).columns)
        if set(cols) != set(row):
            raise ValueError(f"{path} has other columns than this code writes; keep each model in its own folder")
        pd.DataFrame([row])[cols].to_csv(path, mode='a', header=False, index=False)
    else:
        pd.DataFrame([row]).to_csv(path, index=False)


# =============================== ONE RUN ===============================
def run_one(name, run_id, aggregator, seed, data, plat):
    global _rounds_this_session
    spec = MODELS[name]
    X_tr, y_tr, x_val, y_val, x_te, y_te, test_idx, n_features, n_classes = data
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    local = ModelSet(n_features, n_classes, spec["enc_att"])
    glob_m = ModelSet(n_features, n_classes, spec["enc_att"])
    train_client = make_trainer(local)

    n_params = local.n_params()
    print(f"  Model {name}: encoder attention={spec['enc_att']} | "
          f"params: encoder {local.enc.count_params():,}, decoder {local.dec.count_params():,}, "
          f"classifier {local.head.count_params():,} | sent per client and direction: {n_params:,}")
    assert n_params == spec["params"], f"{name} has {n_params} parameters; expected {spec['params']:,}"

    parts = (partition_iid(len(y_tr), seed, NUM_CLIENTS) if PARTITION == "iid"
             else partition_clients(y_tr, ALPHA, seed, NUM_CLIENTS))
    labeled = label_clients(parts, y_tr, LABELS_PER_CLASS_PER_CLIENT, seed)
    y_oh = tf.keras.utils.to_categorical(y_tr, n_classes).astype('float32')
    clients = [(X_tr[idx[~labeled[idx]]], X_tr[idx[labeled[idx]]], y_oh[idx[labeled[idx]]]) for idx in parts]
    active = [k for k, idx in enumerate(parts) if len(idx) >= MIN_CLIENT_FLOWS]
    client_table = pd.DataFrame([np.bincount(y_tr[idx], minlength=n_classes) for idx in parts],
                                columns=LABELS, index=[f"client_{k}" for k in range(NUM_CLIENTS)])
    client_table["labeled"] = [len(c[1]) for c in clients]
    client_table["active"] = [k in active for k in range(NUM_CLIENTS)]
    print(client_table.to_string())
    check_pairing(name, client_table, seed)

    bytes_per_round = len(active) * 2 * 4 * n_params            # float32, up + down, active clients
    global_w = local.get_w()
    meta = {"round_done": 0, "hist": [], "train_seconds": 0.0, "last_round_seconds": 0.0}
    if os.path.exists(ckpt_path(name, run_id)):
        meta, global_w = load_ckpt(name, run_id, local, len(global_w))
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
        glob_m.set_w(global_w)
        vm, _ = evaluate(glob_m.clf.predict(x_val, batch_size=2048, verbose=0), y_val)
        sec = time.time() - t0
        meta["hist"].append({"round": rnd, "val_acc": round(vm["acc"], 6), "val_macro_f1": round(vm["macro_f1"], 6),
                             "comm_MB_cum": round(rnd * bytes_per_round / 1e6, 3), "seconds": round(sec, 1)})
        meta.update(round_done=rnd, last_round_seconds=sec, train_seconds=meta["train_seconds"] + sec)
        print(f"  round {rnd}/{ROUNDS}: val acc {vm['acc']:.4f} | val macro-F1 {vm['macro_f1']:.4f} | {sec:.0f}s")
        save_ckpt(name, run_id, meta, global_w, local)
        _rounds_this_session += 1

    glob_m.set_w(global_w)
    prob = glob_m.clf.predict(x_te, batch_size=2048, verbose=0)
    m, pred = evaluate(prob, y_te)
    recall = confusion_matrix(y_te, pred, labels=list(range(n_classes))).diagonal() / np.bincount(y_te, minlength=n_classes)
    hist = pd.DataFrame(meta["hist"])
    r90 = hist.loc[hist.val_acc >= 0.90, "round"]
    row = dict(run_id=run_id, dataset=DATASET, setting="federated", method=spec["method"],
               model_config=spec["tag"], classifier_attention=False, head_dropout=HEAD_DROPOUT,
               att_dropout=ATT_DROPOUT, aggregator=aggregator, partition=PARTITION,
               alpha=(ALPHA if PARTITION == "dirichlet" else ""),
               clients=NUM_CLIENTS, active_clients=len(active), labels_per_client=LABELS_PER_CLIENT,
               labels_per_class_per_client=LABELS_PER_CLASS_PER_CLIENT, labels_total=int(labeled.sum()),
               seed=seed, rounds=ROUNDS, ae_epochs=AE_EPOCHS, stage15_epochs=STAGE15_EPOCHS,
               clf_epochs=CLF_EPOCHS, batch_ae=BATCH_AE, batch_clf=BATCH_CLF, lr=LR,
               **{k: round(float(v), 6) for k, v in m.items()},
               min_class_recall=round(float(recall.min()), 6),
               **{f"recall_{lab}": round(float(r), 6) for lab, r in zip(LABELS, recall)},
               rounds_to_90=int(r90.iloc[0]) if len(r90) else -1, params_sent=n_params,
               MB_per_round=round(bytes_per_round / 1e6, 3), MB_total=round(ROUNDS * bytes_per_round / 1e6, 1),
               train_seconds=round(meta["train_seconds"], 1), platform=plat, gpu=gpu_name(),
               tf_version=tf.__version__, timestamp=time.strftime("%Y-%m-%d %H:%M"))
    save_run(name, run_id, row, y_te, pred, prob, test_idx, meta["hist"], client_table)
    os.remove(ckpt_path(name, run_id))
    print(f"Done {run_id}: test acc {m['acc']:.4f} | macro-F1 {m['macro_f1']:.4f} | "
          f"per-class recall {np.round(recall, 3).tolist()}")


# =============================== SUMMARY ===============================
def load_model_runs(name, aggregator):
    path = os.path.join(model_dir(name), "results", "all_runs.csv")
    if not os.path.exists(path):
        return None
    runs = pd.read_csv(path)
    runs = runs[runs["aggregator"] == aggregator].copy()
    runs["seed"] = runs["seed"].astype(int)
    return runs.set_index("seed")


def summarize():
    rows = []
    for name in MODELS:
        for aggregator in AGGREGATORS:
            g = load_model_runs(name, aggregator)
            if g is None or not len(g):
                continue
            ms = lambda col: f"{g[col].mean():.4f} ± {g[col].std():.4f}"
            rows.append(dict(dataset=DATASET, model=name, aggregator=aggregator, seeds_done=f"{len(g)}/{len(SEEDS)}",
                             acc=ms("acc"), macro_f1=ms("macro_f1"), weighted_f1=ms("weighted_f1"),
                             min_class_recall=ms("min_class_recall"),
                             params_sent=int(g["params_sent"].iloc[0]), MB_per_round=round(g["MB_per_round"].mean(), 1),
                             train_minutes=round(g["train_seconds"].mean() / 60, 1)))
    if not rows:
        return
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT_ROOT, f"summary_{DATASET}_{SETTING_TAG}{FOLDER_SUFFIX}.csv"), index=False)
    pd.set_option("display.width", 250)
    print(f"\n{DATASET}: mean ± std over seeds (test set):")
    print(table.to_string(index=False))


# =============================== PAIRED TESTS ===============================
def macro_f1(y, p, C):
    """Same as sklearn f1_score(average='macro', zero_division=0) over the classes present in y or p."""
    cm = np.bincount(y * C + p, minlength=C * C).reshape(C, C)
    denom = cm.sum(1) + cm.sum(0)
    present = denom > 0
    return float((2 * np.diag(cm)[present] / denom[present]).mean())


def min_recall(y, p, C):
    t = np.bincount(y, minlength=C)
    return float((np.bincount(y[p == y], minlength=C) / np.maximum(t, 1))[t > 0].min())


def mcnemar_exact(y, pa, pb):
    ca, cb = pa == y, pb == y
    b, c = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    return b, c, (binomtest(c, b + c, 0.5).pvalue if b + c else 1.0)


def randomization_f1(y, pa, pb, C, rng):
    obs = macro_f1(y, pb, C) - macro_f1(y, pa, C)
    differ = np.flatnonzero(pa != pb)
    hits = 0
    for _ in range(N_PERM):
        sw = differ[rng.random(len(differ)) < 0.5]
        qa, qb = pa.copy(), pb.copy()
        qa[sw], qb[sw] = pb[sw], pa[sw]
        hits += abs(macro_f1(y, qb, C) - macro_f1(y, qa, C)) >= abs(obs) - 1e-12
    return obs, (hits + 1) / (N_PERM + 1)


def bootstrap_ci(y, pa, pb, C, rng):
    n = len(y)
    ca, cb = (pa == y).astype(float), (pb == y).astype(float)
    d_acc, d_f1 = np.empty(N_BOOT), np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        d_acc[i] = cb[idx].mean() - ca[idx].mean()
        d_f1[i] = macro_f1(y[idx], pb[idx], C) - macro_f1(y[idx], pa[idx], C)
    return np.percentile(d_acc, [2.5, 97.5]), np.percentile(d_f1, [2.5, 97.5])


def holm(pvals):
    p = np.asarray(pvals, float)
    adj, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def wilcoxon_p(d):
    d = np.asarray(d, float)
    if len(d) < 2 or np.all(d == 0):
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(wilcoxon(d, alternative="two-sided").pvalue)


def paired_t_p(d):
    d = np.asarray(d, float)
    if len(d) < 2 or np.std(d) == 0:
        return float("nan")
    return float(scipy.stats.ttest_1samp(d, 0.0).pvalue)


def fmt_p(p):
    return "n/a" if not np.isfinite(p) else ("< 0.001" if p < 0.001 else f"{p:.3f}")


def load_seed_outputs(name, run_id):
    folder = model_dir(name)
    clients = pd.read_csv(os.path.join(folder, "reports", f"{run_id}_clients.csv"), index_col=0)
    p = np.load(os.path.join(folder, "preds", f"{run_id}.npz"))
    return clients, p["y_true"].astype(int), p["y_pred"].astype(int), p["test_idx"]


def run_paired_tests():
    out = os.path.join(OUT_ROOT, f"paired_tests_{DATASET}_{SETTING_TAG}{FOLDER_SUFFIX}")
    C, rows, used = len(LABELS), [], []
    for aggregator in AGGREGATORS:
        for family, a, b in TEST_COMPARISONS:
            A, B = load_model_runs(a, aggregator), load_model_runs(b, aggregator)
            if A is None or B is None or not set(SEEDS) <= set(A.index) or not set(SEEDS) <= set(B.index):
                print(f"Paired test {b} vs {a} ({aggregator}) skipped: not all seeds are finished for both models")
                continue
            la, lb = model_label(a), model_label(b)
            print(f"\n[{DATASET}] {lb} vs {la} ({aggregator})")
            used.append(dict(family=family, aggregator=aggregator, baseline=la, compared=lb,
                             folder_A=model_dir(a), folder_B=model_dir(b), seeds=sorted(SEEDS)))
            for s in sorted(SEEDS):
                rid_a, rid_b = A.loc[s, "run_id"], B.loc[s, "run_id"]
                cl_a, y, pa, ti_a = load_seed_outputs(a, rid_a)
                cl_b, y_b, pb, ti_b = load_seed_outputs(b, rid_b)
                if not (cl_a.columns.equals(cl_b.columns) and cl_a.index.equals(cl_b.index)
                        and np.array_equal(cl_a.values.astype(str), cl_b.values.astype(str))):
                    raise ValueError(f"Seed {s}: client partition or labeled flows differ:\n  {rid_a}\n  {rid_b}")
                if not (np.array_equal(ti_a, ti_b) and np.array_equal(y, y_b)):
                    raise ValueError(f"Seed {s}: the two runs were not evaluated on the same test flows")
                for R, pred, rid in ((A, pa, rid_a), (B, pb, rid_b)):
                    acc, f1 = float((pred == y).mean()), macro_f1(y, pred, C)
                    if abs(acc - R.loc[s, "acc"]) > 1e-5 or abs(f1 - R.loc[s, "macro_f1"]) > 1e-5:
                        raise ValueError(f"{rid}: predictions give acc {acc:.6f} / macro-F1 {f1:.6f}, "
                                         f"all_runs.csv says {R.loc[s, 'acc']} / {R.loc[s, 'macro_f1']}")
                rng = np.random.default_rng([TEST_SEED, zlib.crc32(f"{DATASET}|{la}|{lb}|{s}".encode())])
                nb, nc, p_mc = mcnemar_exact(y, pa, pb)
                d_f1, p_f1 = randomization_f1(y, pa, pb, C, rng)
                ci_acc, ci_f1 = bootstrap_ci(y, pa, pb, C, rng)
                acc_a, acc_b = float((pa == y).mean()), float((pb == y).mean())
                rows.append(dict(family=family, aggregator=aggregator, dataset=DATASET, baseline=la, compared=lb,
                                 seed=s, run_A=rid_a, run_B=rid_b, n_test=len(y),
                                 acc_A=acc_a, acc_B=acc_b, delta_acc=acc_b - acc_a,
                                 delta_acc_ci_low=ci_acc[0], delta_acc_ci_high=ci_acc[1],
                                 only_A_correct=nb, only_B_correct=nc, p_acc_mcnemar=p_mc,
                                 f1_A=macro_f1(y, pa, C), f1_B=macro_f1(y, pb, C), delta_f1=d_f1,
                                 delta_f1_ci_low=ci_f1[0], delta_f1_ci_high=ci_f1[1], p_f1_randomization=p_f1,
                                 min_recall_A=min_recall(y, pa, C), min_recall_B=min_recall(y, pb, C)))
                print(f"  seed {s}: acc {acc_a:.4f} -> {acc_b:.4f} (McNemar p {fmt_p(p_mc)}) | macro-F1 "
                      f"{rows[-1]['f1_A']:.4f} -> {rows[-1]['f1_B']:.4f} (randomization p {fmt_p(p_f1)})")
    if not rows:
        return
    os.makedirs(out, exist_ok=True)
    per = pd.DataFrame(rows)
    for _, g in per.groupby(["aggregator", "family"]):
        adj = holm(np.concatenate([g["p_acc_mcnemar"], g["p_f1_randomization"]]))
        per.loc[g.index, "p_acc_holm"], per.loc[g.index, "p_f1_holm"] = adj[:len(g)], adj[len(g):]
    for metric in ("acc", "f1"):
        sig = per[f"p_{metric}_holm"] < SIG_LEVEL
        per[f"{metric}_result"] = np.where(~sig, "no significant difference",
                                           np.where(per[f"delta_{metric}"] > 0, per["compared"] + " better",
                                                    per["baseline"] + " better"))
    keys = ["aggregator", "family", "baseline", "compared"]
    summ = []
    for (agg, fam, a, b), g in per.groupby(keys, sort=False):
        n = len(g)
        summ.append(dict(dataset=DATASET, aggregator=agg, family=fam, baseline=a, compared=b, seeds=n,
                         acc_A=f"{g.acc_A.mean():.4f} ± {g.acc_A.std():.4f}", acc_B=f"{g.acc_B.mean():.4f} ± {g.acc_B.std():.4f}",
                         delta_acc_mean=g.delta_acc.mean(), delta_acc_std=g.delta_acc.std(),
                         f1_A=f"{g.f1_A.mean():.4f} ± {g.f1_A.std():.4f}", f1_B=f"{g.f1_B.mean():.4f} ± {g.f1_B.std():.4f}",
                         delta_f1_mean=g.delta_f1.mean(), delta_f1_std=g.delta_f1.std(),
                         seeds_B_higher_acc=f"{int((g.delta_acc > 0).sum())}/{n}",
                         seeds_B_higher_f1=f"{int((g.delta_f1 > 0).sum())}/{n}",
                         acc_significant_B_better=int((g.acc_result == b + " better").sum()),
                         acc_significant_A_better=int((g.acc_result == a + " better").sum()),
                         f1_significant_B_better=int((g.f1_result == b + " better").sum()),
                         f1_significant_A_better=int((g.f1_result == a + " better").sum()),
                         paired_t_p_acc=paired_t_p(g.delta_acc), paired_t_p_f1=paired_t_p(g.delta_f1),
                         wilcoxon_p_acc=wilcoxon_p(g.delta_acc), wilcoxon_p_f1=wilcoxon_p(g.delta_f1),
                         wilcoxon_smallest_possible_p=2 / 2 ** n))
    summ = pd.DataFrame(summ)
    per.to_csv(os.path.join(out, "per_seed.csv"), index=False)
    summ.to_csv(os.path.join(out, "summary.csv"), index=False)

    md = [f"# Paired tests: {DATASET}, {'IID' if PARTITION == 'iid' else f'Dirichlet alpha = {ALPHA}'}, "
          f"K = {NUM_CLIENTS}, {LABELS_PER_CLIENT} labels per client, {ROUNDS} rounds", "",
          f"McNemar's exact test (accuracy) and a paired randomization test with {N_PERM:,} swaps (macro-F1) per seed, "
          f"on the same frozen test set; 95% CIs from {N_BOOT:,} paired bootstrap resamples; Holm-Bonferroni "
          f"correction within each family; significance at {SIG_LEVEL}. Across seeds: paired t-test and Wilcoxon. "
          f"Differences are compared model (B) minus baseline (A).", ""]
    for (agg, fam, a, b), g in per.groupby(keys, sort=False):
        md += [f"## {b} vs {a} ({fam}, {agg})", "",
               "| Seed | Acc A | Acc B | Δ acc [95% CI] | McNemar p (Holm) | F1 A | F1 B | Δ macro-F1 [95% CI] | Randomization p (Holm) |",
               "|---|---|---|---|---|---|---|---|---|"]
        for r in g.itertuples():
            md.append(f"| {r.seed} | {r.acc_A:.4f} | {r.acc_B:.4f} | {r.delta_acc:+.4f} [{r.delta_acc_ci_low:+.4f}, "
                      f"{r.delta_acc_ci_high:+.4f}] | {fmt_p(r.p_acc_holm)} | {r.f1_A:.4f} | {r.f1_B:.4f} | "
                      f"{r.delta_f1:+.4f} [{r.delta_f1_ci_low:+.4f}, {r.delta_f1_ci_high:+.4f}] | {fmt_p(r.p_f1_holm)} |")
        s = summ[(summ.aggregator == agg) & (summ.family == fam) & (summ.baseline == a) & (summ.compared == b)].iloc[0]
        md += ["", f"Across seeds: Δ acc {s.delta_acc_mean:+.4f} ± {s.delta_acc_std:.4f} (paired t p = "
                   f"{fmt_p(s.paired_t_p_acc)}), Δ macro-F1 {s.delta_f1_mean:+.4f} ± {s.delta_f1_std:.4f} (paired t p = "
                   f"{fmt_p(s.paired_t_p_f1)}); {b} higher on {s.seeds_B_higher_acc} seeds (acc). Wilcoxon p = "
                   f"{fmt_p(s.wilcoxon_p_acc)} (acc); with {s.seeds} seeds the smallest possible p is "
                   f"{s.wilcoxon_smallest_possible_p:.3f}.", ""]
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(os.path.join(out, "protocol.json"), "w") as f:
        json.dump(dict(tests=dict(accuracy="McNemar exact (two-sided binomial on discordant flows)",
                                  macro_f1=f"paired approximate randomization, {N_PERM} swaps, two-sided",
                                  ci=f"paired bootstrap over test flows, {N_BOOT} resamples, percentile 95%",
                                  correction="Holm-Bonferroni within each family",
                                  across_seeds="paired t-test and Wilcoxon signed-rank"),
                       significance_level=SIG_LEVEL, test_seed=TEST_SEED, model="global model after the last round",
                       comparisons=used,
                       versions=dict(numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__),
                       created=time.strftime("%Y-%m-%d %H:%M")), f, indent=2, default=str)
    print(f"\n{DATASET} PAIRED TESTS (Holm-adjusted p):")
    print(per[["compared", "baseline", "seed", "acc_A", "acc_B", "p_acc_holm", "acc_result",
               "f1_A", "f1_B", "p_f1_holm", "f1_result"]].round(4).to_string(index=False))
    print(f"Saved per_seed.csv, summary.csv, report.md, protocol.json in {out}")


# =============================== MAIN ===============================
def run_dataset(plat):
    print("\n" + "#" * 80 + f"\n# {DATASET}, {SETTING_TAG}\n" + "#" * 80)
    import_previous()
    X, y, n_features = load_data()
    tr, va, te = get_split(y)
    if EXPECTED_SPLIT:
        assert (len(tr), len(va), len(te)) == tuple(EXPECTED_SPLIT), \
            f"Split {len(tr)}/{len(va)}/{len(te)} differs from Table II {EXPECTED_SPLIT}"
    data = (X[tr], y[tr], X[va], y[va], X[te], y[te], te, n_features, len(LABELS))
    print(f"Train {len(tr)} | Val {len(va)} | Test {len(te)}")
    print(f"Protocol: {PARTITION}{f' alpha {ALPHA}' if PARTITION == 'dirichlet' else ''} | K={NUM_CLIENTS} | "
          f"rounds {ROUNDS} | epochs: autoencoder {AE_EPOCHS}, classifier {CLF_EPOCHS} | batch {BATCH_AE}/{BATCH_CLF} | "
          f"labels {LABELS_PER_CLIENT}/client ({LABELS_PER_CLASS_PER_CLIENT}/class) | dropout: attention "
          f"{ATT_DROPOUT}, head {HEAD_DROPOUT} | seeds {SEEDS}")
    todo = []
    for name in MODELS_TO_RUN:
        runs = [(agg, s, run_id_for(name, agg, s)) for s in SEEDS for agg in AGGREGATORS]
        left = [(name, agg, s, rid) for agg, s, rid in runs if not already_done(name, rid)]
        todo += left
        print(f"  {name:10s} finished {len(runs) - len(left)}/{len(runs)}"
              + (f" | to run: seeds {[s for _, _, s, _ in left]}" if left else ""))
    order = sorted(todo, key=lambda t: (t[2], MODELS_TO_RUN.index(t[0])))   # seed first: paired sets finish early
    for name, aggregator, seed, rid in order:
        write_manifest(name)
        print(f"\n===== {rid} =====")
        run_one(name, rid, aggregator, seed, data, plat)
    summarize()
    if RUN_PAIRED_TESTS:
        run_paired_tests()


def main():
    plat = platform_name()
    print(f"IID experiment | datasets {DATASETS} | models {MODELS_TO_RUN} | {plat} | {gpu_name()}"
          + (" | QUICK TEST" if QUICK_TEST else ""))
    tf.config.experimental.enable_op_determinism()
    try:
        for ds in DATASETS:
            configure(ds)
            run_dataset(plat)
    except StopForNow:
        print("\n" + "=" * 70)
        print("STOPPED SAFELY (time budget). All progress is saved in:", OUT_ROOT)
        print('Kaggle: add this version\'s output as input, keep RESUME_FROM = ["/kaggle/input"], Save & Run All.')
        print("=" * 70)
        summarize()
        return
    print("\nALL DATASETS FINISHED.")


if __name__ == "__main__":
    main()
