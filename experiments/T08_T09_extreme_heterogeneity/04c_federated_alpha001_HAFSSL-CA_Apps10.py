# -*- coding: utf-8 -*-
# Table VIII, HAFSSL-CA on 10 Popular Apps (alpha = 0.01): hybrid attention in the encoder and in
# the classifier head ('clfatt'). Results: results/Apps10_FL-AECNN_HAFSSL_HAFSSL-CA.zip (clfatt folder).
# Converted from the Kaggle notebook of the reported runs (outputs removed).

import os, glob, json
import numpy as np
import pandas as pd

BASE = "/kaggle/input/notebooks/<user>/0-01-fl-aecnn-hafssl-hafssl-clfatt/"
pd.set_option("display.width", 200)

print("=== summary_Apps10_alpha001.csv ===")
p = os.path.join(BASE, "summary_Apps10_alpha001.csv")
print(pd.read_csv(p).to_string(index=False) if os.path.exists(p) else "not found")

for name in ["FL-AECNN", "HAFSSL", "HAFSSL_clfatt"]:
    folder = os.path.join(BASE, f"FED_Apps10_alpha001_{name}")
    print(f"\n=== {name} ===")
    f = os.path.join(folder, "results", "all_runs.csv")
    if os.path.exists(f):
        r = pd.read_csv(f)
        print(r[["run_id", "seed", "acc", "macro_f1", "min_class_recall", "rounds"]].to_string(index=False))
    else:
        print("no finished run yet")
    for c in glob.glob(os.path.join(folder, "checkpoints", "*.npz")):
        done = json.loads(str(np.load(c)["meta"]))["round_done"]
        print(f"unfinished: {os.path.basename(c)} -> {done}/100 rounds done")

# %%
# -*- coding: utf-8 -*-
"""
HAFSSL / FL-AECNN on 10 POPULAR APPS - extreme heterogeneity (Dirichlet alpha = 0.01, K = 10), R2 revision.
ONE NOTEBOOK FOR ALL THREE MODELS (same split, same client partitions, same labeled flows, same protocol):

  1. FL-AECNN       no attention                                        -> /kaggle/working/FED_Apps10_alpha001_FL-AECNN
  2. HAFSSL         hybrid attention in the encoder (the paper's model) -> /kaggle/working/FED_Apps10_alpha001_HAFSSL
  3. HAFSSL_clfatt  hybrid attention in the encoder AND the classifier  -> /kaggle/working/FED_Apps10_alpha001_HAFSSL_clfatt

MODELS
  encoder    Conv64-Conv64-MaxPool-[MHA->SE, 4 heads x key_dim 32, FFN 128]-Conv32-Conv32-Dense(39)
  decoder    Dense-Reshape-Conv32-Conv32-UpSampling-Conv64-Conv1
  classifier Dense-Reshape-Conv64-Conv64-MaxPool-[MHA->SE, 4 heads x key_dim 16, FFN 64]-Conv128-Conv128-MaxPool
             -Dense(128)-Softmax
  [...] = hybrid attention block, only where the model has attention. No head dropout in any model.
  Parameters sent per client and direction: FL-AECNN 385,234 | HAFSSL 435,862 | HAFSSL_clfatt 461,658

PROTOCOL (identical for the three models)
  every round, on each client:
    stage 1   : autoencoder on the client's UNLABELED flows, 3 epochs, batch 32, Adam (skipped if < 50 flows)
    stage 1.5 : encoder fine-tuned with the classifier head FROZEN, 2 epochs, batch 256, Adam lr 1e-4
    stage 2   : encoder + classifier, 3 epochs, batch 256, Adam
  data        : exact duplicate flows removed BEFORE splitting; one stratified train/val/test split (seed 2026)
  labels      : each client labels up to 200 flows of every class it holds (seeded); its other flows are unlabeled
  aggregation : weighted FedAvg n_k / n (Eq. 5); clients with < 50 flows are skipped
  rounds      : 100, no early stopping; seeds 0, 1, 2 (same partition and labeled flows for every model)
  test        : the GLOBAL model after the last round, once, on the frozen test set
  payload     : float32, up + down, active clients; encoder, decoder and classifier are sent every round

OUTPUT (in /kaggle/working)
  one folder per model : config.json, results/all_runs.csv, preds/, reports/ (per-class, confusion, rounds,
                         clients), splits/
  summary_Apps10_alpha001.csv : mean ± std over seeds for the three models
  paired_tests_Apps10_alpha001/ : McNemar (accuracy) + randomization test (macro-F1) per seed, Holm correction,
                                  written when all runs are finished (per_seed.csv, summary.csv, report.md)

HOW TO RUN ON KAGGLE
  1. GPU on. Paste this whole file into one cell. Save Version -> Save & Run All.
  2. A session stops safely before 12 hours and keeps all progress in its output. To continue:
     Add Input -> this notebook's output, set RESUME_FROM = ["/kaggle/input"], Save & Run All again.
     Finished runs are skipped and an interrupted run continues from its last saved round.
  3. Runs you already finished with the earlier notebooks (same code) are reused the same way: add those
     outputs as inputs too. Folders named exactly like the three above are found and checked automatically.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import json, time, shutil, zlib, warnings
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
DATA_PATH = "data/10PopularApps/pcapdroid_10class_each_normalized_cuttedfloefeature.csv"
DATASET   = "Apps10"
# Names in the order of the label codes 0..9 in the csv (must match the centralized Table II script)
LABELS    = ['TikTok', 'iQiyi Video', 'Jingdong Shopping', 'Snack Video', 'QQ Music',
             'QQ', 'Taobao Shopping', 'NetEase Cloud Music', 'Arena of Valor', 'WeChat']

# The three models: attention in the encoder / in the classifier, tag in the run ids, parameters sent
MODELS = {
    "FL-AECNN":      dict(method="FL-AECNN", enc_att=False, clf_att=False, tag="original", params=385234),
    "HAFSSL":        dict(method="HAFSSL",   enc_att=True,  clf_att=False, tag="standard", params=435862),
    "HAFSSL_clfatt": dict(method="HAFSSL",   enc_att=True,  clf_att=True,  tag="clfatt",   params=461658),
}
MODELS_TO_RUN = ["FL-AECNN", "HAFSSL", "HAFSSL_clfatt"]     # trained in this order

OUT_ROOT    = "/kaggle/working"
RESUME_FROM = ["/kaggle/input"]        # e.g. ["/kaggle/input"] to continue from earlier outputs (this notebook or the earlier ones)

EXPECTED_ROWS  = 170459                     # rows after removing duplicates (centralized Table II runs)
EXPECTED_SPLIT = (138071, 15342, 17046)     # train / val / test

AGGREGATORS = ["FedAvg"]
ALPHA        = 0.01
NUM_CLIENTS  = 10
ROUNDS       = 100
LOCAL_EPOCHS = 3               # stage 1 and stage 2
STAGE15_EPOCHS, STAGE15_LR = 2, 1e-4
BATCH_AE, BATCH_CLF = 32, 256 
LR = 1e-3
LABELS_PER_CLASS_PER_CLIENT = 200
MIN_CLIENT_FLOWS = 50
SEEDS = [0, 1, 2]

NUM_HEADS, KEY_DIM, SE_RATIO, LATENT_DIM, ATT_DROPOUT = 4, 32, 16, 39, 0.1
SPLIT_SEED        = 2026    # same as every other experiment = same test set
TIME_BUDGET_HOURS = 11.0
QUICK_TEST        = False   # True = 2 rounds, 1 epoch, seed 0, separate "_quicktest" folders (a few minutes)

# Paired tests, run automatically when every run is finished (protocol fixed here, before the results exist)
RUN_PAIRED_TESTS = True
TEST_COMPARISONS = [            # (family, baseline A, compared model B); Holm correction within each family
    ("main",     "FL-AECNN", "HAFSSL"),
    ("ablation", "FL-AECNN", "HAFSSL_clfatt"),
    ("ablation", "HAFSSL",   "HAFSSL_clfatt"),
]
SIG_LEVEL, N_PERM, N_BOOT, TEST_SEED = 0.05, 10_000, 10_000, 2026

if isinstance(RESUME_FROM, str):
    RESUME_FROM = [RESUME_FROM] if RESUME_FROM else []
assert set(MODELS_TO_RUN) <= set(MODELS), f"MODELS_TO_RUN must only use the names {list(MODELS)}"
if QUICK_TEST:
    ROUNDS, LOCAL_EPOCHS, STAGE15_EPOCHS, SEEDS = 2, 1, 1, [0]
FOLDER_SUFFIX = "_quicktest" if QUICK_TEST else ""
SESSION_START = time.time()
_STOP_AFTER_ROUNDS = int(os.environ.get("STOP_AFTER_ROUNDS", "0"))   # testing only
_rounds_this_session = 0

MANIFEST_KEYS = ["DATA_PATH", "DATASET", "LABELS", "AGGREGATORS", "ALPHA", "NUM_CLIENTS", "ROUNDS", "LOCAL_EPOCHS",
                 "STAGE15_EPOCHS", "STAGE15_LR", "BATCH_AE", "BATCH_CLF", "LR", "LABELS_PER_CLASS_PER_CLIENT",
                 "MIN_CLIENT_FLOWS", "SEEDS", "NUM_HEADS", "KEY_DIM", "SE_RATIO", "LATENT_DIM", "ATT_DROPOUT",
                 "SPLIT_SEED"]
MUST_MATCH = [k for k in MANIFEST_KEYS if k not in ("DATA_PATH", "AGGREGATORS", "SEEDS")]   # for reused runs


# =============================== FOLDERS AND NAMES ===============================
def model_dir(name):
    return os.path.join(OUT_ROOT, f"FED_{DATASET}_alpha001_{name}{FOLDER_SUFFIX}")


def run_id_for(name, aggregator, seed):
    spec = MODELS[name]
    tag = "" if spec["tag"] == "original" else f"_{spec['tag']}"
    return f"{DATASET}_{spec['method']}{tag}_{aggregator}_a{ALPHA}_K{NUM_CLIENTS}_s{seed}"


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
    for root in RESUME_FROM:
        if not os.path.isdir(root):
            raise FileNotFoundError(f"RESUME_FROM folder not found: {root}")
    for name in MODELS:
        dst = model_dir(name)
        if os.path.isdir(dst) and os.listdir(dst):
            print(f"{name}: already in {dst}, nothing copied")
            continue
        target = os.path.basename(dst)
        cands = sorted({d for root in RESUME_FROM for d, _, files in os.walk(root)
                        if os.path.basename(d) == target and "config.json" in files})
        if not cands:
            continue
        best = max(cands, key=lambda c: (finished_runs(c), checkpoint_round(c)))
        check_reused(best, name)
        shutil.copytree(best, dst)
        ck = checkpoint_round(best)
        print(f"{name}: reused {finished_runs(best)} finished run(s)"
              + (f" and a checkpoint at round {ck}" if ck else "") + f" from {best}")
        for c in cands:
            if c != best:
                print(f"   not used (less progress): {c}")


def write_manifest(name):
    folder = model_dir(name)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "config.json")
    if os.path.exists(path):              # keep the record of the session that started this model
        return
    spec = MODELS[name]
    cfg = {k: globals()[k] for k in MANIFEST_KEYS}
    cfg.update(MODEL=spec["method"], MODEL_TAG=spec["tag"], encoder_attention=spec["enc_att"],
               classifier_attention=spec["clf_att"], head_dropout=0.0, tf_version=tf.__version__,
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
    print(f"Features: {len(feat_cols)} | rows per class: {dict(zip(*np.unique(y, return_counts=True)))}")
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


def label_clients(parts, y_train, per_class, seed):
    """Each client labels up to `per_class` flows of every class it holds (original rule). Seeded."""
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
    """MHA -> SE, residual, FFN (same as the original SEBlock1D + hybrid_attention_block)."""
    a = se_block(layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=key_dim, dropout=ATT_DROPOUT)(x, x))
    x = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, a]))
    f = layers.Dense(key_dim * 4, activation='relu')(x)
    f = layers.Dense(x.shape[-1])(layers.Dropout(ATT_DROPOUT)(f))
    return layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, f]))


def build_models(n_features, n_classes, enc_attention, clf_attention):
    """enc_attention: hybrid attention block in the encoder; clf_attention: one more in the classifier head."""
    ie = layers.Input(shape=(n_features, 1))
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(ie)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D()(x)
    if enc_attention:
        x = hybrid_attention_block(x, KEY_DIM)                 # encoder attention (key_dim 32)
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
    if clf_attention:
        c = hybrid_attention_block(c, KEY_DIM // 2)            # classifier attention (key_dim 16)
    c = layers.Conv1D(128, 3, padding='same', activation='relu')(c)
    c = layers.Conv1D(128, 3, padding='same', activation='relu')(c)
    c = layers.MaxPooling1D(2)(c)
    c = layers.Dense(128, activation='relu')(layers.Flatten()(c))
    head = Model(ic, layers.Dense(n_classes, activation='softmax')(c), name='classifier_head')

    inp = layers.Input(shape=(n_features, 1))
    return enc, dec, head, Model(inp, dec(enc(inp))), Model(inp, head(enc(inp)))


class ModelSet:
    def __init__(self, n_features, n_classes, enc_attention, clf_attention):
        self.enc, self.dec, self.head, self.ae, self.clf = build_models(n_features, n_classes,
                                                                        enc_attention, clf_attention)
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
    m = {'acc': accuracy_score(y, pred), 'macro_precision': p, 'macro_recall': r, 'macro_f1': f1,
         'balanced_acc': balanced_accuracy_score(y, pred)}
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
    local = ModelSet(n_features, n_classes, spec["enc_att"], spec["clf_att"])
    glob = ModelSet(n_features, n_classes, spec["enc_att"], spec["clf_att"])
    train_client = make_trainer(local)

    n_params = local.n_params()
    print(f"  Model {name}: encoder attention={spec['enc_att']} | classifier attention={spec['clf_att']} | "
          f"params: encoder {local.enc.count_params():,}, decoder {local.dec.count_params():,}, "
          f"classifier {local.head.count_params():,}")
    print(f"  Parameters sent per client and direction: {n_params:,}")
    assert n_params == spec["params"], f"{name} has {n_params} parameters; expected {spec['params']:,}"

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
        glob.set_w(global_w)
        vm, _ = evaluate(glob.clf.predict(x_val, batch_size=2048, verbose=0), y_val)
        sec = time.time() - t0
        meta["hist"].append({"round": rnd, "val_acc": round(vm["acc"], 6), "val_macro_f1": round(vm["macro_f1"], 6),
                             "comm_MB_cum": round(rnd * bytes_per_round / 1e6, 3), "seconds": round(sec, 1)})
        meta.update(round_done=rnd, last_round_seconds=sec, train_seconds=meta["train_seconds"] + sec)
        print(f"  round {rnd}/{ROUNDS}: val acc {vm['acc']:.4f} | val macro-F1 {vm['macro_f1']:.4f} | {sec:.0f}s")
        save_ckpt(name, run_id, meta, global_w, local)
        _rounds_this_session += 1

    glob.set_w(global_w)
    prob = glob.clf.predict(x_te, batch_size=2048, verbose=0)
    m, pred = evaluate(prob, y_te)
    recall = confusion_matrix(y_te, pred, labels=list(range(n_classes))).diagonal() / np.bincount(y_te, minlength=n_classes)
    hist = pd.DataFrame(meta["hist"])
    r90 = hist.loc[hist.val_acc >= 0.90, "round"]
    row = dict(run_id=run_id, dataset=DATASET, setting="federated", method=spec["method"],
               model_config=spec["tag"], classifier_attention=spec["clf_att"], head_dropout=0.0,
               aggregator=aggregator, partition="dirichlet", alpha=ALPHA, clients=NUM_CLIENTS,
               active_clients=len(active), labels_per_class_per_client=LABELS_PER_CLASS_PER_CLIENT,
               labels_total=int(labeled.sum()), seed=seed, rounds=ROUNDS, local_epochs=LOCAL_EPOCHS,
               stage15_epochs=STAGE15_EPOCHS, batch_ae=BATCH_AE, batch_clf=BATCH_CLF, lr=LR,
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


# =============================== SUMMARY OF THE THREE MODELS ===============================
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
            rows.append(dict(model=name, aggregator=aggregator, seeds_done=f"{len(g)}/{len(SEEDS)}",
                             acc=ms("acc"), macro_f1=ms("macro_f1"), min_class_recall=ms("min_class_recall"),
                             params_sent=int(g["params_sent"].iloc[0]), MB_per_round=round(g["MB_per_round"].mean(), 1)))
    if not rows:
        return
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT_ROOT, f"summary_{DATASET}_alpha001{FOLDER_SUFFIX}.csv"), index=False)
    pd.set_option("display.width", 250)
    print("\nMean ± std over seeds (test set):")
    print(table.to_string(index=False))


# =============================== PAIRED TESTS ===============================
def macro_f1(y, p, C):
    """Same as sklearn f1_score(average='macro', zero_division=0) over the classes present in y or p."""
    cm = np.bincount(y * C + p, minlength=C * C).reshape(C, C)
    denom = cm.sum(1) + cm.sum(0)                     # = 2 TP + FP + FN per class
    present = denom > 0
    return float((2 * np.diag(cm)[present] / denom[present]).mean())


def min_recall(y, p, C):
    t = np.bincount(y, minlength=C)
    return float((np.bincount(y[p == y], minlength=C) / np.maximum(t, 1))[t > 0].min())


def mcnemar_exact(y, pa, pb):
    ca, cb = pa == y, pb == y
    b, c = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))      # b: only A right, c: only B right
    return b, c, (binomtest(c, b + c, 0.5).pvalue if b + c else 1.0)


def randomization_f1(y, pa, pb, C, rng):
    obs = macro_f1(y, pb, C) - macro_f1(y, pa, C)
    differ = np.flatnonzero(pa != pb)                          # swapping equal predictions changes nothing
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


def fmt_p(p):
    return "n/a" if not np.isfinite(p) else ("< 0.001" if p < 0.001 else f"{p:.3f}")


def load_seed_outputs(name, run_id):
    folder = model_dir(name)
    clients = pd.read_csv(os.path.join(folder, "reports", f"{run_id}_clients.csv"), index_col=0)
    p = np.load(os.path.join(folder, "preds", f"{run_id}.npz"))
    return clients, p["y_true"].astype(int), p["y_pred"].astype(int), p["test_idx"]


def run_paired_tests():
    out = os.path.join(OUT_ROOT, f"paired_tests_{DATASET}_alpha001{FOLDER_SUFFIX}")
    C, rows, used = len(LABELS), [], []
    for aggregator in AGGREGATORS:
        for family, a, b in TEST_COMPARISONS:
            A, B = load_model_runs(a, aggregator), load_model_runs(b, aggregator)
            if A is None or B is None or not set(SEEDS) <= set(A.index) or not set(SEEDS) <= set(B.index):
                print(f"Paired test {b} vs {a} ({aggregator}) skipped: not all seeds are finished for both models")
                continue
            la, lb = model_label(a), model_label(b)
            print(f"\n[{family}] {lb} vs {la} ({aggregator})")
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
                for R, pred, rid in ((A, pa, rid_a), (B, pb, rid_b)):      # saved predictions = reported numbers
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
    for _, g in per.groupby(["aggregator", "family"]):              # Holm over every test of the family
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
        summ.append(dict(aggregator=agg, family=fam, baseline=a, compared=b, seeds=n,
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
                         wilcoxon_p_acc=wilcoxon_p(g.delta_acc), wilcoxon_p_f1=wilcoxon_p(g.delta_f1),
                         wilcoxon_smallest_possible_p=2 / 2 ** n))
    summ = pd.DataFrame(summ)
    per.to_csv(os.path.join(out, "per_seed.csv"), index=False)
    summ.to_csv(os.path.join(out, "summary.csv"), index=False)

    md = [f"# Paired tests: {DATASET}, Dirichlet alpha = {ALPHA}, K = {NUM_CLIENTS}", "",
          f"McNemar's exact test (accuracy) and a paired randomization test with {N_PERM:,} swaps (macro-F1) per seed, "
          f"on the same frozen test set; 95% CIs from {N_BOOT:,} paired bootstrap resamples (per seed, not adjusted "
          f"for multiple testing); Holm-Bonferroni correction within each family; significance at {SIG_LEVEL}. "
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
        md += ["", f"Across seeds: Δ acc {s.delta_acc_mean:+.4f} ± {s.delta_acc_std:.4f}, Δ macro-F1 "
                   f"{s.delta_f1_mean:+.4f} ± {s.delta_f1_std:.4f}; {b} higher on {s.seeds_B_higher_acc} seeds (acc) and "
                   f"{s.seeds_B_higher_f1} (macro-F1). Wilcoxon signed-rank p = {fmt_p(s.wilcoxon_p_acc)} (acc), "
                   f"{fmt_p(s.wilcoxon_p_f1)} (macro-F1); with {s.seeds} seeds the smallest possible p is "
                   f"{s.wilcoxon_smallest_possible_p:.3f}.", ""]
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(os.path.join(out, "protocol.json"), "w") as f:
        json.dump(dict(tests=dict(accuracy="McNemar exact (two-sided binomial on discordant flows)",
                                  macro_f1=f"paired approximate randomization, {N_PERM} swaps, two-sided",
                                  ci=f"paired bootstrap over test flows, {N_BOOT} resamples, percentile 95%",
                                  correction="Holm-Bonferroni within each family", across_seeds="Wilcoxon signed-rank"),
                       significance_level=SIG_LEVEL, test_seed=TEST_SEED, model="global model after the last round",
                       comparisons=used,
                       versions=dict(numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__),
                       created=time.strftime("%Y-%m-%d %H:%M")), f, indent=2, default=str)
    print("\nPAIRED TESTS (Holm-adjusted p):")
    print(per[["family", "compared", "baseline", "seed", "acc_A", "acc_B", "p_acc_holm", "acc_result",
               "f1_A", "f1_B", "p_f1_holm", "f1_result"]].round(4).to_string(index=False))
    print(f"Saved per_seed.csv, summary.csv, report.md, protocol.json in {out}")


# =============================== MAIN ===============================
def main():
    plat = platform_name()
    print(f"{DATASET}, alpha {ALPHA}: models {MODELS_TO_RUN} | {plat} | {gpu_name()}"
          + (" | QUICK TEST" if QUICK_TEST else ""))
    import_previous()
    tf.config.experimental.enable_op_determinism()
    X, y, n_features = load_data()
    tr, va, te = get_split(y)
    if EXPECTED_SPLIT:
        assert (len(tr), len(va), len(te)) == tuple(EXPECTED_SPLIT), \
            f"Split {len(tr)}/{len(va)}/{len(te)} differs from Table II {EXPECTED_SPLIT}"
    data = (X[tr], y[tr], X[va], y[va], X[te], y[te], te, n_features, len(LABELS))
    print(f"Train {len(tr)} | Val {len(va)} | Test {len(te)}")
    print(f"Protocol: alpha {ALPHA} | K={NUM_CLIENTS} | rounds {ROUNDS} | local epochs {LOCAL_EPOCHS} "
          f"(+ stage 1.5: {STAGE15_EPOCHS}) | labels {LABELS_PER_CLASS_PER_CLIENT}/class/client | "
          f"aggregators {AGGREGATORS} | seeds {SEEDS}")

    todo = []
    for name in MODELS_TO_RUN:
        runs = [(agg, s, run_id_for(name, agg, s)) for s in SEEDS for agg in AGGREGATORS]
        left = [(name, agg, s, rid) for agg, s, rid in runs if not already_done(name, rid)]
        todo += left
        print(f"  {name:14s} finished {len(runs) - len(left)}/{len(runs)}"
              + (f" | to run: seeds {[s for _, _, s, _ in left]}" if left else ""))
    try:
        for name, aggregator, seed, rid in todo:
            write_manifest(name)
            print(f"\n===== {rid} =====")
            run_one(name, rid, aggregator, seed, data, plat)
    except StopForNow:
        print("\n" + "=" * 70)
        print("STOPPED SAFELY (time budget). All progress is saved in:", OUT_ROOT)
        print('Kaggle: add this version\'s output as input, set RESUME_FROM = ["/kaggle/input"], Save & Run All.')
        print("=" * 70)
        summarize()
        return
    print("\nALL RUNS FINISHED.")
    summarize()
    if RUN_PAIRED_TESTS:
        run_paired_tests()


if __name__ == "__main__":
    main()
