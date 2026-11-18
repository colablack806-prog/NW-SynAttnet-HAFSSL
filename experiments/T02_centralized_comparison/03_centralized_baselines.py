# -*- coding: utf-8 -*-
"""
Table II (centralized) baselines for the IEEE TAI R2 revision.
Methods: FLUIDS, VAE(-CNN), ByteSGAN, ADGCN.  Datasets: ISCX-VPN2016 (5 classes) and 10 Popular Apps.

PROTOCOL (identical to the Table II SynAttnNet / AECNN runs):
  * same de-duplicated data, same frozen train/val/test split (copied from REFERENCE_DIR),
  * same labeled flows per (label budget, seed): sample_labeled() of the SynAttnNet script, default_rng(seed),
  * label budgets 1000/2000/3000/4000 labeled flows PER CLASS, seeds 0,1,2,
  * unsupervised stages (AE / VAE / GAN discriminator / ADGCN-AE) use TRAINING flows only, never val/test,
  * model selection by the VALIDATION split only; the test split is touched once, at the end,
  * macro precision/recall/F1 computed from saved predictions.

Each baseline keeps its authors' architecture and hyper-parameters (from the original code). Deviations,
all needed to put every method on the same split and evaluation, are listed in DEVIATIONS below and in
the output folder (protocol_manifest.json).

ADGCN (Xu et al., Sci. Rep. 2025) is adapted to flow features: a class-agnostic AE (66-128-64-32) is
trained on training flows, the reconstructed flows form a k=10 nearest-neighbour graph, and a 4-layer
GCNII (alpha 0.2, lambda 0.5, hidden 64, lr 0.005, dropout 0.6) classifies the nodes. Like the original
it is TRANSDUCTIVE (test nodes are in the graph, without labels), which can only favour ADGCN.

Kaggle: one notebook per dataset (DATASET = "ISCX" or "Apps10"); QUICK_TEST first.
At the end: ONLY_SUMMARIZE = True with SUMMARY_DIRS = [this output, Table II SynAttnNet output,
AECNN output] builds Table II with mean +- std, McNemar and paired t-test (Holm).
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import glob, json, time, shutil, hashlib, math
import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, balanced_accuracy_score,
                             confusion_matrix, classification_report)

# =============================== CONFIG (10 Popular Apps, ready to run) ===============================
# Attach to this notebook (Add Input): 1) the 10-app-dataset, 2) the SynAttnNet Apps10 output,
# 3) the AECNN Apps10 output. Paths below are found automatically; you normally change nothing.
DATASET          = "Apps10"
INPUT_ROOT       = "/kaggle/input"
DATA_FILE        = "data/10PopularApps/pcapdroid_10class_each_normalized_cuttedfloefeature.csv"
REFERENCE_DIR    = ""              # SynAttnNet Apps10 output (folder with splits/, preds/, results/); "" = auto-find
AECNN_DIR        = ""              # AECNN Apps10 output (folder with preds/, results/); "" = auto-find
SPLIT_FILE       = ""              # optional explicit split .npz; "" = REFERENCE_DIR/splits
OUT_DIR          = ""              # "" = /kaggle/working/TableII_baselines_Apps10
METHODS          = ["FLUIDS", "VAE", "ByteSGAN", "ADGCN"]
LABELS_PER_CLASS = [1000, 2000, 3000, 4000]
SEEDS            = [0, 1, 2]
RESUME_FROM      = ""              # previous output of THIS notebook (Kaggle input) to continue after a stop
TIME_BUDGET_HOURS = 11.3
QUICK_TEST       = False          # True = 1 seed, 1000 labels, very short training (pipeline check only)

ONLY_SUMMARIZE   = False           # True = no training; rebuild Table II from the folders below
SUMMARY_DIRS     = []              # [] = this output + RESUME_FROM + SynAttnNet + AECNN (auto)
REFERENCE_METHOD = "SynAttnNet"

DATA_NAMES = {"ISCX": "ISCX_5class_each_normalized_cuttedfloefeature.csv",
              "Apps10": "pcapdroid_10class_each_normalized_cuttedfloefeature.csv"}
LABEL_NAMES = {"ISCX": ['chat', 'email', 'file', 'streaming', 'voip'],
               "Apps10": ['TikTok', 'iQiyi Video', 'Jingdong Shopping', 'Snack Video', 'QQ Music', 'QQ',
                          'Taobao Shopping', 'NetEase Cloud Music', 'Arena of Valor', 'WeChat']}

# ----- original hyper-parameters (from the authors' code / papers) -----
HP = {
    "FLUIDS":   dict(latent=39, ae_epochs=100, clf_epochs=100, batch=128, lr=1e-3, patience=10),
    "VAE":      dict(latent=20, vae_epochs=20, vae_batch=32, vae_lr=8e-6, clf_epochs=100, batch=128,
                     lr=1e-3, patience=10),
    "ByteSGAN": dict(latent=100, steps=5000, batch=128, lr=2e-4, beta_1=0.1, dropout=0.5, eval_every=100),
    "ADGCN":    dict(ae_dims=(128, 64, 32), ae_epochs=10, ae_batch=256, ae_lr=1e-3, k=10, layers=4,
                     alpha=0.2, lamda=0.5, hidden=64, lr=0.005, dropout=0.6, wd_conv=0.01, wd_other=5e-4,
                     max_epochs=3000, patience=200),
}
DEVIATIONS = [
    "All methods: data, de-duplication, train/val/test split and labeled flows are those of the Table II "
    "SynAttnNet runs; unsupervised stages see training flows only (originals pre-trained on all data, incl. test).",
    "All methods: model selected on the validation split (early stopping / best validation checkpoint); "
    "originals used the final epoch or monitored the test set.",
    "All methods: macro precision/recall/F1 from saved predictions (ADGCN code used weighted averaging).",
    "ByteSGAN: labeled mini-batches drawn from the shared labeled set (original: 1000/class drawn with replacement).",
    "ADGCN: flow features instead of 784 payload bytes; class-agnostic AE (66-128-64-32, 10 epochs) instead of "
    "per-class AEs (which need the class of every flow, including test flows); symmetric k=10 Euclidean kNN graph "
    "(the heat-kernel ranking is identical; edge weights are unused in the authors' code); 4 GCNII layers as in "
    "the paper; validation accuracy for early stopping (patience 200); transductive as in the original.",
]

if not OUT_DIR:
    OUT_DIR = f"/kaggle/working/TableII_baselines_{DATASET}"
if QUICK_TEST:
    SEEDS, LABELS_PER_CLASS = [0], [min(LABELS_PER_CLASS)]
    OUT_DIR = OUT_DIR + "_quicktest"
    HP["FLUIDS"].update(ae_epochs=2, clf_epochs=2)
    HP["VAE"].update(vae_epochs=1, clf_epochs=2)
    HP["ByteSGAN"].update(steps=200)
    HP["ADGCN"].update(ae_epochs=1, max_epochs=60, patience=20)
SESSION_START = time.time()

for g in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass


class StopForNow(Exception):
    pass


# =============================== AUTO-FIND INPUT FOLDERS ===============================
def _auto_dir(kind):
    """kind = 'synattnnet' (needs splits/<DATASET>_split.npz) or 'aecnn' (needs results/all_runs.csv)."""
    if kind == "synattnnet":
        hits = glob.glob(os.path.join(INPUT_ROOT, "**", "splits", f"{DATASET}_split.npz"), recursive=True)
        dirs = [os.path.dirname(os.path.dirname(h)) for h in hits]
        dirs = [d for d in dirs if "aecnn" not in d.lower() and "baseline" not in d.lower()
                and "quicktest" not in d.lower()]
    else:
        hits = glob.glob(os.path.join(INPUT_ROOT, "**", "results", "all_runs.csv"), recursive=True)
        dirs = []
        for h in hits:
            d = os.path.dirname(os.path.dirname(h))
            if "aecnn" not in d.lower() or "quicktest" in d.lower():
                continue
            try:
                r = pd.read_csv(h, usecols=["dataset", "method"])
                if (r["dataset"] == DATASET).any() and r["method"].astype(str).str.contains("AECNN").any():
                    dirs.append(d)
            except Exception:
                pass
    return sorted(dirs)[0] if dirs else ""


def resolve_inputs():
    global REFERENCE_DIR, AECNN_DIR
    if not (REFERENCE_DIR and os.path.isdir(REFERENCE_DIR)):
        REFERENCE_DIR = _auto_dir("synattnnet")
    if not (AECNN_DIR and os.path.isdir(AECNN_DIR)):
        AECNN_DIR = _auto_dir("aecnn")
    print("SynAttnNet folder:", REFERENCE_DIR or "NOT FOUND (split will be rebuilt with the same rule)")
    print("AECNN folder     :", AECNN_DIR or "NOT FOUND (AECNN will be missing from Table II)")


# =============================== DATA ===============================
def find_data_file():
    if DATA_FILE and os.path.isfile(DATA_FILE):
        return DATA_FILE
    name = DATA_NAMES[DATASET]
    hits = glob.glob(os.path.join(INPUT_ROOT, "**", name), recursive=True) + glob.glob(f"./**/{name}", recursive=True)
    if not hits:
        raise FileNotFoundError(f"{name} not found. Set DATA_FILE to the CSV path.")
    return hits[0]


def load_data():
    """IDENTICAL to load_data() in 01_centralized_synattnnet_R2.py and the AECNN script."""
    path = find_data_file()
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c != 'label']
    n0 = len(df)
    df = df.drop_duplicates(subset=feat_cols).reset_index(drop=True)
    X = df[feat_cols].values.astype(np.float32)
    y = df['label'].values.astype(int)
    print(f"Data: {path}")
    print(f"Rows {n0} -> {len(y)} after removing {n0 - len(y)} exact duplicates | features {X.shape[1]} | "
          f"min {X.min():.3f} max {X.max():.3f}")
    if X.shape[1] != 66:
        print(f"NOTE: {X.shape[1]} features (the baselines were written for 66).")
    return X, y


def fingerprint(X, y):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X).tobytes())
    h.update(np.ascontiguousarray(y).tobytes())
    return h.hexdigest()


def find_split_file():
    if SPLIT_FILE:
        return SPLIT_FILE
    if not REFERENCE_DIR:
        return None
    hits = sorted(glob.glob(os.path.join(REFERENCE_DIR, "**", "*split*.npz"), recursive=True))
    hits = [h for h in hits if "selection" not in os.path.basename(h)]
    if not hits:
        raise FileNotFoundError(f"No *split*.npz found under {REFERENCE_DIR}")
    print("Split file:", hits[0])
    return hits[0]


def make_split_like_synattnnet(y):
    """Fallback when no reference split is attached: IDENTICAL to get_split() in
    01_centralized_synattnnet_R2.py (same data + SPLIT_SEED 2026 = same train/val/test)."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    trval, test = train_test_split(idx, test_size=0.10, stratify=y, random_state=2026)
    train, val = train_test_split(trval, test_size=0.10, stratify=y[trval], random_state=2026)
    path = os.path.join(OUT_DIR, "splits", f"{DATASET}_split.npz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, train=train, val=val, test=test, n_rows=len(y))
    print("No REFERENCE_DIR: split rebuilt with the SynAttnNet rule (seed 2026) and saved to", path)
    return path


def load_split(n_rows, data_sha, y=None):
    path = find_split_file() or make_split_like_synattnnet(y)
    with np.load(path, allow_pickle=False) as s:
        keys = set(s.files)
        tr, va, te = (s[k] for k in ("train", "val" if "val" in keys else "valid", "test"))
        if "n_rows" in keys and int(s["n_rows"]) != n_rows:
            raise ValueError(f"Split has n_rows={int(s['n_rows'])} but data has {n_rows} after de-duplication. "
                             "The preprocessing differs from the Table II script; do not run.")
        if "data_sha256" in keys and str(s["data_sha256"]) != data_sha:
            print("WARNING: split data_sha256 differs from this script's fingerprint (hash method may differ). "
                  "Row count matches; test labels are checked against the reference predictions below.")
    allidx = np.concatenate([tr, va, te])
    assert len(allidx) == n_rows and np.array_equal(np.sort(allidx), np.arange(n_rows)), \
        "Split indices do not cover the data exactly once."
    return tr.astype(int), va.astype(int), te.astype(int), path


def check_reference_test(y, te):
    """Every reference prediction file must have the same test labels in the same order."""
    dirs = [d for d in (REFERENCE_DIR, AECNN_DIR) if d]
    if not dirs:
        return "no reference preds checked"
    files = sorted(f for d in dirs for f in glob.glob(os.path.join(d, "**", "preds", "*.npz"), recursive=True))
    n = 0
    for f in files:
        with np.load(f, allow_pickle=False) as p:
            if "y_true" not in p.files:
                continue
            if "test_idx" in p.files and not np.array_equal(p["test_idx"], te):
                raise ValueError(f"Reference test indices differ: {f}")
            if not np.array_equal(p["y_true"], y[te]):
                raise ValueError(f"Reference test labels differ: {f}")
            n += 1
    print(f"Test set identical to {n} reference prediction file(s).")
    return f"test set identical to {n} reference prediction files"


def find_reference_labels(budget, seed, y, tr):
    """Return train-relative labeled indices saved by the reference run, or None."""
    if not REFERENCE_DIR:
        return None, None
    cands = glob.glob(os.path.join(REFERENCE_DIR, "**", "splits", "*.npz"), recursive=True)
    tags_b = (f"L{budget}", f"_{budget}_", f"lab{budget}", f"{budget}pc")
    for f in sorted(cands):
        b = os.path.basename(f)
        if not any(t in b for t in tags_b) or not (f"_s{seed}." in b or f"_s{seed}_" in b or f"seed{seed}" in b):
            continue
        with np.load(f, allow_pickle=False) as d:
            for k in d.files:
                if "lab" not in k.lower() or d[k].dtype.kind not in "iu" or d[k].ndim != 1:
                    continue
                idx = d[k].astype(int)
                n_cls = len(np.unique(y))
                if len(idx) != budget * n_cls:
                    continue
                if idx.max() < len(tr) and np.all(np.bincount(y[tr][idx], minlength=n_cls) == budget):
                    return np.sort(idx), f"{os.path.basename(f)}:{k} (train-relative)"
                pos = {v: i for i, v in enumerate(tr)}
                if all(v in pos for v in idx) and np.all(np.bincount(y[idx], minlength=n_cls) == budget):
                    return np.sort(np.array([pos[v] for v in idx])), f"{os.path.basename(f)}:{k} (absolute)"
    return None, None


def select_labels(y_tr, budget, seed):
    """IDENTICAL to sample_labeled() in the SynAttnNet / AECNN scripts: same seed + budget = same flows."""
    rng = np.random.default_rng(seed)
    chosen = []
    for c in np.unique(y_tr):
        pool = np.where(y_tr == c)[0]
        k = min(budget, len(pool))
        if k < budget:
            print(f"  Warning: class {c} has only {len(pool)} training flows; using all of them.")
        chosen.extend(rng.choice(pool, k, replace=False))
    return np.array(chosen)


# =============================== METRICS / SAVING ===============================
def evaluate(prob, y_true):
    pred = prob.argmax(1)
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="macro", zero_division=0)
    m = dict(acc=accuracy_score(y_true, pred), macro_precision=p, macro_recall=r, macro_f1=f1,
             balanced_acc=balanced_accuracy_score(y_true, pred))
    assert abs(m["macro_recall"] - m["balanced_acc"]) < 1e-9
    return m, pred


def results_path():
    return os.path.join(OUT_DIR, "results", "all_runs.csv")


def already_done(run_id):
    return os.path.exists(results_path()) and run_id in set(pd.read_csv(results_path())["run_id"])


def save_run(run_id, row, y_te, prob, te, hist=None):
    for sub in ("results", "preds", "reports"):
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)
    m, pred = evaluate(prob, y_te)
    names = LABEL_NAMES[DATASET]
    labs = list(range(len(names)))
    rec = confusion_matrix(y_te, pred, labels=labs).diagonal() / np.maximum(np.bincount(y_te, minlength=len(labs)), 1)
    np.savez_compressed(os.path.join(OUT_DIR, "preds", f"{run_id}.npz"),
                        y_true=y_te, y_pred=pred, y_prob=prob.astype(np.float32), test_idx=te)
    rep = classification_report(y_te, pred, labels=labs, target_names=names, digits=4, output_dict=True,
                                zero_division=0)
    pd.DataFrame(rep).transpose().to_csv(os.path.join(OUT_DIR, "reports", f"{run_id}_per_class.csv"))
    pd.DataFrame(confusion_matrix(y_te, pred, labels=labs), index=names, columns=names).to_csv(
        os.path.join(OUT_DIR, "reports", f"{run_id}_confusion.csv"))
    if hist is not None:
        pd.DataFrame(hist).to_csv(os.path.join(OUT_DIR, "reports", f"{run_id}_history.csv"), index=False)
    row.update({k: round(float(v), 6) for k, v in m.items()})
    row.update(min_class_recall=round(float(rec.min()), 6), classes_below_50=int((rec < 0.5).sum()))
    row.update({f"recall_{n}": round(float(r), 6) for n, r in zip(names, rec)})
    row.update(platform="kaggle" if os.path.exists("/kaggle") else "pc", tf_version=tf.__version__,
               gpu=(tf.config.list_physical_devices('GPU') or ["CPU"])[0].__str__()[:60],
               timestamp=time.strftime("%Y-%m-%d %H:%M"))
    pd.DataFrame([row]).to_csv(results_path(), mode="a", header=not os.path.exists(results_path()), index=False)
    print(f"  -> test acc {m['acc']:.4f} | macro-F1 {m['macro_f1']:.4f} | min class recall {rec.min():.3f}")


def early_stop(patience):
    return [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)]


def n_params(*models):
    return int(sum(m.count_params() for m in models))


# =============================== FLUIDS ===============================
class FLUIDS:
    """Dense AE (n-150-100-50-39) pre-trained on unlabeled flows; frozen encoder + MLP (150-100-50)."""
    def __init__(self, n_features, n_classes, seed, X_tr, X_va):
        hp = HP["FLUIDS"]
        tf.keras.utils.set_random_seed(seed)
        ie = layers.Input(shape=(n_features,))
        x = layers.Dense(150, activation="relu")(ie)
        x = layers.Dense(100, activation="relu")(x)
        x = layers.Dense(50, activation="relu")(x)
        self.enc = Model(ie, layers.Dense(hp["latent"], activation="relu")(x))
        idd = layers.Input(shape=(hp["latent"],))
        d = layers.Dense(50, activation="relu")(idd)
        d = layers.Dense(100, activation="relu")(d)
        d = layers.Dense(150, activation="relu")(d)
        self.dec = Model(idd, layers.Dense(n_features, activation="relu")(d))
        ae = Model(ie, self.dec(self.enc(ie)))
        ae.compile(optimizer=tf.keras.optimizers.Adam(hp["lr"]), loss="mse")
        ae.fit(X_tr, X_tr, epochs=hp["ae_epochs"], batch_size=hp["batch"], validation_data=(X_va, X_va),
               callbacks=early_stop(hp["patience"]), verbose=0)
        self.n_classes = n_classes

    def run(self, X_lab, y_lab, X_va, y_va, X_te, seed):
        hp = HP["FLUIDS"]
        tf.keras.utils.set_random_seed(seed)
        Z = lambda a: self.enc.predict(a, batch_size=4096, verbose=0)
        ic = layers.Input(shape=(HP["FLUIDS"]["latent"],))
        c = layers.Dense(150, activation="relu")(ic)
        c = layers.Dense(100, activation="relu")(c)
        c = layers.Dense(50, activation="relu")(c)
        mlp = Model(ic, layers.Dense(self.n_classes, activation="softmax")(c))
        mlp.compile(optimizer=tf.keras.optimizers.Adam(hp["lr"]), loss="sparse_categorical_crossentropy")
        h = mlp.fit(Z(X_lab), y_lab, epochs=hp["clf_epochs"], batch_size=hp["batch"],
                    validation_data=(Z(X_va), y_va), callbacks=early_stop(hp["patience"]), verbose=0)
        return mlp.predict(Z(X_te), batch_size=4096, verbose=0), n_params(self.enc, self.dec, mlp), h.history


# =============================== VAE-CNN ===============================
class VAE_CNN:
    """VAE (n-512-2x20) pre-trained on unlabeled flows; encoder output [mu, logvar] (40-d) -> 1-D CNN."""
    def __init__(self, n_features, n_classes, seed, X_tr, X_va):
        hp = HP["VAE"]
        tf.keras.utils.set_random_seed(seed)
        L = hp["latent"]
        self.enc = tf.keras.Sequential([layers.Input(shape=(n_features,)), layers.Dense(512, activation="relu"),
                                        layers.Dense(2 * L)])
        self.dec = tf.keras.Sequential([layers.Input(shape=(L,)), layers.Dense(512, activation="relu"),
                                        layers.Dense(n_features)])
        opt = tf.keras.optimizers.Adam(hp["vae_lr"])
        vars_ = self.enc.trainable_variables + self.dec.trainable_variables

        def loss_fn(x):
            mean, logvar = tf.split(self.enc(x), 2, axis=1)
            z = tf.random.normal(tf.shape(mean)) * tf.exp(0.5 * logvar) + mean
            rec = tf.reduce_mean(tf.square(x - self.dec(z)))
            kl = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mean) - tf.exp(logvar))
            return rec + kl

        @tf.function
        def step(x):
            with tf.GradientTape() as tape:
                loss = loss_fn(x)
            opt.apply_gradients(zip(tape.gradient(loss, vars_), vars_))

        ds = tf.data.Dataset.from_tensor_slices(X_tr).shuffle(len(X_tr), seed=seed).batch(hp["vae_batch"])
        for ep in range(hp["vae_epochs"]):
            for xb in ds:
                step(xb)
        self.n_classes = n_classes

    def run(self, X_lab, y_lab, X_va, y_va, X_te, seed):
        hp = HP["VAE"]
        tf.keras.utils.set_random_seed(seed)
        Z = lambda a: self.enc.predict(a, batch_size=4096, verbose=0)
        ic = layers.Input(shape=(2 * hp["latent"],))
        x = layers.Reshape((33, 32))(layers.Dense(33 * 32)(ic))
        x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
        x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
        x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dense(128, activation="relu")(layers.Flatten()(x))
        cnn = Model(ic, layers.Dense(self.n_classes, activation="softmax")(x))
        cnn.compile(optimizer=tf.keras.optimizers.Adam(hp["lr"]), loss="sparse_categorical_crossentropy")
        h = cnn.fit(Z(X_lab), y_lab, epochs=hp["clf_epochs"], batch_size=hp["batch"],
                    validation_data=(Z(X_va), y_va), callbacks=early_stop(hp["patience"]), verbose=0)
        return cnn.predict(Z(X_te), batch_size=4096, verbose=0), n_params(self.enc, self.dec, cnn), h.history


# =============================== ByteSGAN ===============================
def run_bytesgan(X_tr, X_lab, y_lab, X_va, y_va, X_te, n_classes, seed):
    """Semi-supervised GAN (Salimans et al. style) exactly as the authors' code: shared 1-D CNN trunk,
    supervised softmax head, unsupervised head D(x) = Z/(Z+1) with Z = sum(exp(logits))."""
    hp = HP["ByteSGAN"]
    tf.keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(30_000 + seed)
    n = X_tr.shape[1]
    ii = layers.Input(shape=(n, 1))
    f = layers.Conv1D(64, 3, padding="same", activation="relu")(ii)
    f = layers.Conv1D(64, 3, padding="same", activation="relu")(f)
    f = layers.MaxPooling1D(2)(f)
    f = layers.Conv1D(128, 3, padding="same", activation="relu")(f)
    f = layers.Conv1D(128, 3, padding="same", activation="relu")(f)
    f = layers.MaxPooling1D(2)(f)
    f = layers.Dense(128, activation="relu")(layers.Flatten()(f))
    f = layers.Dropout(hp["dropout"])(f)
    trunk = Model(ii, layers.Dense(n_classes)(f))           # logits

    h = int(math.ceil(n / 2))
    zi = layers.Input(shape=(hp["latent"],))
    g = layers.LeakyReLU(0.2)(layers.Dense(128 * h)(zi))
    g = layers.BatchNormalization(momentum=0.8)(layers.Reshape((h, 128))(g))
    g = layers.Conv1D(1, 3, padding="same", activation="tanh")(layers.UpSampling1D()(g))
    g = layers.Cropping1D((0, 2 * h - n))(g) if 2 * h != n else g
    gen = Model(zi, g)

    mk = lambda: tf.keras.optimizers.Adam(hp["lr"], beta_1=hp["beta_1"])
    opt_c, opt_d, opt_g = mk(), mk(), mk()
    tv, gv = trunk.trainable_variables, gen.trainable_variables
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # D(x) = S/(S+1) with S = sum exp(logits)  <=>  sigmoid(logsumexp(logits))  (numerically stable form)
    d_logit = lambda logits: tf.reduce_logsumexp(logits, axis=-1, keepdims=True)

    @tf.function
    def c_step(x, y):
        with tf.GradientTape() as t:
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, trunk(x, training=True),
                                                                                  from_logits=True))
        opt_c.apply_gradients(zip(t.gradient(loss, tv), tv))
        return loss

    @tf.function
    def d_step(x, target):
        with tf.GradientTape() as t:
            loss = bce(target, d_logit(trunk(x, training=True)))
        opt_d.apply_gradients(zip(t.gradient(loss, tv), tv))
        return loss

    @tf.function
    def g_step(z):
        with tf.GradientTape() as t:
            loss = bce(tf.ones((tf.shape(z)[0], 1)), d_logit(trunk(gen(z, training=True), training=True)))
        opt_g.apply_gradients(zip(t.gradient(loss, gv), gv))
        return loss

    X3, L3, V3, T3 = X_tr[..., None], X_lab[..., None], X_va[..., None], X_te[..., None]
    half = hp["batch"] // 2
    best_acc, best_w, hist = -1.0, None, []
    for step in range(hp["steps"]):
        b = rng.integers(0, len(L3), half)
        c_step(tf.constant(L3[b]), tf.constant(y_lab[b]))
        d_step(tf.constant(X3[rng.integers(0, len(X3), half)]), tf.ones((half, 1)))
        fake = gen(tf.random.normal((half, hp["latent"])), training=False)
        d_step(fake, tf.zeros((half, 1)))
        g_step(tf.random.normal((hp["batch"], hp["latent"])))
        if (step + 1) % hp["eval_every"] == 0 or step + 1 == hp["steps"]:
            va = float(np.mean(trunk.predict(V3, batch_size=4096, verbose=0).argmax(1) == y_va))
            hist.append({"step": step + 1, "val_acc": round(va, 6)})
            if va > best_acc:
                best_acc, best_w = va, trunk.get_weights()
    trunk.set_weights(best_w)
    prob = tf.nn.softmax(trunk.predict(T3, batch_size=4096, verbose=0)).numpy()
    return prob, n_params(trunk, gen), hist


# =============================== ADGCN ===============================
class ADGCN:
    """AE reconstruction of every flow -> symmetric kNN graph (k=10) over all flows -> 4-layer GCNII."""
    def __init__(self, X, tr_idx, va_idx, n_classes, seed):
        hp = HP["ADGCN"]
        tf.keras.utils.set_random_seed(seed)
        n = X.shape[1]
        in01 = X.min() >= 0 and X.max() <= 1
        ie = layers.Input(shape=(n,))
        e = ie
        for d in hp["ae_dims"]:
            e = layers.Dense(d, activation="relu")(e)
        x = e
        for d in list(hp["ae_dims"][:-1])[::-1]:
            x = layers.Dense(d, activation="relu")(x)
        out = layers.Dense(n, activation="sigmoid" if in01 else None)(x)
        ae = Model(ie, out)
        ae.compile(optimizer=tf.keras.optimizers.Adam(hp["ae_lr"]),
                   loss="binary_crossentropy" if in01 else "mse")
        ae.fit(X[tr_idx], X[tr_idx], epochs=hp["ae_epochs"], batch_size=hp["ae_batch"],
               validation_data=(X[va_idx], X[va_idx]), verbose=0)
        self.ae_params = ae.count_params()
        self.Xr = ae.predict(X, batch_size=4096, verbose=0).astype(np.float32)
        t0 = time.time()
        self.P = self.build_graph(self.Xr, hp["k"])
        print(f"  ADGCN graph: {self.Xr.shape[0]} nodes, {self.P.indices.shape[0]} non-zeros "
              f"({time.time() - t0:.0f}s)")
        self.n_classes = n_classes

    @staticmethod
    def build_graph(Xr, k):
        N = len(Xr)
        Xt = tf.constant(Xr)
        sq = tf.reduce_sum(Xt * Xt, axis=1)
        nbr = np.zeros((N, k), dtype=np.int64)
        chunk = 1024
        for s in range(0, N, chunk):
            a = Xt[s:s + chunk]
            d = tf.reduce_sum(a * a, 1, keepdims=True) - 2.0 * tf.matmul(a, Xt, transpose_b=True) + sq[None, :]
            idx = tf.math.top_k(-d, k=k + 1).indices.numpy()
            for i, row in enumerate(idx):
                row = row[row != s + i][:k]
                nbr[s + i] = row
        rows = np.repeat(np.arange(N), k)
        A = sp.coo_matrix((np.ones(N * k, np.float32), (rows, nbr.ravel())), shape=(N, N)).tocsr()
        A = A.maximum(A.T)
        A.setdiag(1.0)
        deg = np.asarray(A.sum(1)).ravel()
        dinv = sp.diags(1.0 / np.sqrt(deg))
        P = (dinv @ A @ dinv).tocoo()
        st = tf.SparseTensor(np.stack([P.row, P.col], 1).astype(np.int64), P.data.astype(np.float32), P.shape)
        return tf.sparse.reorder(st)

    def run(self, y, lab_abs, va_idx, te_idx, seed):
        hp = HP["ADGCN"]
        tf.keras.utils.set_random_seed(seed)
        lin_in = layers.Dense(hp["hidden"])
        convs = [layers.Dense(hp["hidden"], use_bias=False) for _ in range(hp["layers"])]
        lin_out = layers.Dense(self.n_classes)
        X = tf.constant(self.Xr)
        lin_in.build((None, X.shape[1]))
        for c in convs:
            c.build((None, hp["hidden"]))
        lin_out.build((None, hp["hidden"]))
        conv_vars = [v for c in convs for v in c.trainable_variables]
        other_vars = lin_in.trainable_variables + lin_out.trainable_variables
        all_vars = conv_vars + other_vars
        opt = tf.keras.optimizers.Adam(hp["lr"])
        betas = [math.log(hp["lamda"] / (l + 1) + 1) for l in range(hp["layers"])]
        P, a, dr = self.P, hp["alpha"], hp["dropout"]

        def forward(training):
            h = tf.nn.dropout(X, dr) if training else X
            h0 = tf.nn.relu(lin_in(h))
            h = h0
            for c, b in zip(convs, betas):
                h = tf.nn.dropout(h, dr) if training else h
                sup = (1 - a) * tf.sparse.sparse_dense_matmul(P, h) + a * h0
                h = tf.nn.relu(b * c(sup) + (1 - b) * sup)
            h = tf.nn.dropout(h, dr) if training else h
            return lin_out(h)

        lab_t = tf.constant(lab_abs)
        y_lab = tf.constant(y[lab_abs])

        @tf.function
        def train_step():
            with tf.GradientTape() as t:
                logits = tf.gather(forward(True), lab_t)
                loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_lab, logits,
                                                                                      from_logits=True))
                loss += 0.5 * hp["wd_conv"] * tf.add_n([tf.reduce_sum(v * v) for v in conv_vars])
                loss += 0.5 * hp["wd_other"] * tf.add_n([tf.reduce_sum(v * v) for v in other_vars])
            opt.apply_gradients(zip(t.gradient(loss, all_vars), all_vars))
            return loss

        @tf.function
        def predict():
            return tf.nn.softmax(forward(False))

        best, bad, best_w, hist = -1.0, 0, None, []
        for ep in range(1, hp["max_epochs"] + 1):
            loss = float(train_step())
            prob = predict().numpy()
            va = float(np.mean(prob[va_idx].argmax(1) == y[va_idx]))
            if ep % 50 == 0:
                hist.append({"epoch": ep, "loss": round(loss, 5), "val_acc": round(va, 6)})
            if va > best:
                best, bad, best_w = va, 0, [v.numpy() for v in all_vars]
            else:
                bad += 1
                if bad >= hp["patience"]:
                    break
        for v, w in zip(all_vars, best_w):
            v.assign(w)
        prob = predict().numpy()[te_idx]
        params = self.ae_params + int(sum(np.prod(v.shape) for v in all_vars))
        return prob, params, hist


# =============================== DRIVER ===============================
def manifest(data_sha, split_path, test_check):
    cfg = dict(dataset=DATASET, methods=METHODS, labels_per_class=LABELS_PER_CLASS, seeds=SEEDS,
               data_sha256=data_sha, split_file=os.path.basename(split_path), hyperparameters=HP,
               selection="validation split only; test evaluated once", metrics="macro, from saved predictions",
               deviations=DEVIATIONS, quick_test=QUICK_TEST)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "protocol_manifest.json")
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        keep = ("dataset", "data_sha256", "split_file", "hyperparameters", "quick_test")
        if any(old.get(k) != json.loads(json.dumps(cfg))[k] for k in keep):
            raise ValueError("Settings changed since the saved runs. Use a new OUT_DIR.")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print("Reference test check:", test_check)


def check_time(estimate_s):
    if time.time() - SESSION_START + estimate_s > TIME_BUDGET_HOURS * 3600:
        raise StopForNow()


def summary_dirs():
    return SUMMARY_DIRS or [d for d in (OUT_DIR, RESUME_FROM, REFERENCE_DIR, AECNN_DIR) if d and os.path.isdir(d)]


def main():
    resolve_inputs()
    if RESUME_FROM and not (os.path.isdir(OUT_DIR) and os.listdir(OUT_DIR)):
        shutil.copytree(RESUME_FROM, OUT_DIR, dirs_exist_ok=True)
        print("Copied previous progress from", RESUME_FROM)
    X, y = load_data()
    data_sha = fingerprint(X, y)
    names = LABEL_NAMES[DATASET]
    assert np.array_equal(np.unique(y), np.arange(len(names))), f"Labels must be 0..{len(names) - 1}"
    tr, va, te, split_path = load_split(len(y), data_sha, y)
    test_check = check_reference_test(y, te)
    manifest(data_sha, split_path, test_check)
    X_tr, y_tr, X_va, y_va, X_te, y_te = X[tr], y[tr], X[va], y[va], X[te], y[te]
    C, F = len(names), X.shape[1]
    print(f"{DATASET}: train {len(tr)} | val {len(va)} | test {len(te)} | classes {C} | features {F}")
    print("Training flows per class:", np.bincount(y_tr, minlength=C).tolist())
    budgets = [b for b in LABELS_PER_CLASS if np.bincount(y_tr, minlength=C).min() >= b]
    if budgets != LABELS_PER_CLASS:
        print("SKIPPED budgets (a class has too few training flows):", sorted(set(LABELS_PER_CLASS) - set(budgets)))

    durations = {}
    try:
        for seed in SEEDS:
            for method in METHODS:
                todo = [b for b in budgets if not already_done(f"{DATASET}_{method}_L{b}_s{seed}")]
                if not todo:
                    print(f"Skip (done): {method} seed {seed}")
                    continue
                check_time(1.3 * durations.get(method, 0) * (len(todo) + 1))
                t_pre = time.time()
                print(f"\n===== {DATASET} | {method} | seed {seed} | budgets {todo} =====")
                pre = None
                if method == "FLUIDS":
                    pre = FLUIDS(F, C, seed, X_tr, X_va)
                elif method == "VAE":
                    pre = VAE_CNN(F, C, seed, X_tr, X_va)
                elif method == "ADGCN":
                    pre = ADGCN(X, tr, va, C, seed)
                pre_s = time.time() - t_pre
                for b in todo:
                    run_id = f"{DATASET}_{method}_L{b}_s{seed}"
                    check_time(1.3 * durations.get(method, 0))
                    lab, src = find_reference_labels(b, seed, y, tr)
                    if lab is None:
                        lab, src = select_labels(y_tr, b, seed), "same rule as SynAttnNet/AECNN: default_rng(seed)"
                    print(f"  {run_id}: {len(lab)} labeled flows ({src})")
                    t0 = time.time()
                    if method == "ByteSGAN":
                        prob, npar, hist = run_bytesgan(X_tr, X_tr[lab], y_tr[lab], X_va, y_va, X_te, C, seed)
                    elif method == "ADGCN":
                        prob, npar, hist = pre.run(y, tr[lab], va, te, seed)
                    else:
                        prob, npar, hist = pre.run(X_tr[lab], y_tr[lab], X_va, y_va, X_te, seed)
                    sec = time.time() - t0 + pre_s / len(todo)
                    durations[method] = max(durations.get(method, 0), sec)
                    row = dict(run_id=run_id, dataset=DATASET, setting="centralized", method=method,
                               labels_per_class=b, labels_total=int(len(lab)),
                               labeled_share=round(len(lab) / len(tr), 5), seed=seed, n_params=npar,
                               label_source=src, data_sha256=data_sha, split_file=os.path.basename(split_path),
                               train_seconds=round(sec, 1))
                    save_run(run_id, row, y_te, prob, te, hist)
                del pre
                tf.keras.backend.clear_session()
    except StopForNow:
        print("\n" + "=" * 70 + "\nSTOPPED SAFELY (time budget). Results so far are in " + OUT_DIR +
              "\nAdd this output as input, set RESUME_FROM to it, Save & Run All.\n" + "=" * 70)
        return
    print("\nALL RUNS FINISHED.")
    summarize(summary_dirs())


# =============================== SUMMARY (Table II) ===============================
ALIASES = {"labels_per_class": ["labels_per_class", "labeled_per_class", "label_budget", "labels", "labelnum",
                                "labels_per_class_global"],
           "macro_precision": ["macro_precision", "precision"], "macro_recall": ["macro_recall", "recall"],
           "macro_f1": ["macro_f1", "f1"], "acc": ["acc", "accuracy"]}


def load_all(dirs):
    frames = []
    for d in dirs:
        for f in glob.glob(os.path.join(d, "**", "results", "*.csv"), recursive=True):
            df = pd.read_csv(f)
            for std, al in ALIASES.items():
                if std not in df.columns:
                    hit = next((a for a in al if a in df.columns), None)
                    if hit:
                        df[std] = df[hit]
            need = {"run_id", "method", "seed", "acc", "macro_f1", "labels_per_class"}
            if not need.issubset(df.columns):
                print(f"Skipped {f}: missing {need - set(df.columns)}")
                continue
            if "setting" in df.columns:
                df = df[df["setting"].fillna("centralized") == "centralized"]
            df["_preds"] = [os.path.join(os.path.dirname(os.path.dirname(f)), "preds", f"{r}.npz") for r in df.run_id]
            if "dataset" not in df.columns:
                df["dataset"] = DATASET
            frames.append(df)
    if not frames:
        raise ValueError("No usable results found in SUMMARY_DIRS.")
    return pd.concat(frames, ignore_index=True).drop_duplicates("run_id")


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = run
    return adj


def summarize(dirs):
    from scipy.stats import ttest_rel, binomtest
    df = load_all(dirs)
    ref = REFERENCE_METHOD or next((m for m in ["SynAttnNet", "Proposed", "HAFSSL"] if m in set(df.method)), None)
    cols = ["acc", "macro_precision", "macro_recall", "macro_f1"]
    tab = (df.groupby(["dataset", "labels_per_class", "method"])[cols].agg(["mean", "std", "count"]))
    out_dir = os.path.join(OUT_DIR, "summary")
    os.makedirs(out_dir, exist_ok=True)
    tab.round(4).to_csv(os.path.join(out_dir, "tableII_mean_std.csv"))
    print("\nTable II (mean, std, n seeds):\n", tab.round(4).to_string())
    if ref is None:
        print("No reference method found; set REFERENCE_METHOD to run the tests.")
        return
    rows = []
    for (ds, b), g in df.groupby(["dataset", "labels_per_class"]):
        r = g[g.method == ref].set_index("seed")
        for m in sorted(set(g.method) - {ref}):
            o = g[g.method == m].set_index("seed")
            seeds = sorted(set(r.index) & set(o.index))
            if len(seeds) < 2:
                continue
            d = r.loc[seeds, "acc"].values - o.loc[seeds, "acc"].values
            p_t = ttest_rel(r.loc[seeds, "acc"], o.loc[seeds, "acc"]).pvalue
            mc = []
            for s in seeds:
                try:
                    A, B = np.load(r.loc[s, "_preds"]), np.load(o.loc[s, "_preds"])
                    assert np.array_equal(A["y_true"], B["y_true"])
                    a, bb = A["y_pred"] == A["y_true"], B["y_pred"] == B["y_true"]
                    n01, n10 = int(np.sum(a & ~bb)), int(np.sum(~a & bb))
                    mc.append((binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0, np.sign(n01 - n10)))
                except Exception as e:
                    mc.append((np.nan, 0))
                    print(f"McNemar skipped for {m} seed {s}: {e}")
            rows.append(dict(dataset=ds, labels_per_class=b, comparison=f"{ref} vs {m}", n_seeds=len(seeds),
                             mean_diff_pp=round(100 * d.mean(), 3), ttest_p=p_t,
                             mcnemar_p_max=np.nanmax([x[0] for x in mc]),
                             mcnemar_same_sign=all(x[1] == np.sign(d.mean()) for x in mc)))
    if not rows:
        print("No paired comparisons (need >= 2 common seeds).")
        return
    res = pd.DataFrame(rows)
    res["ttest_p_holm"] = holm(res["ttest_p"].fillna(1.0))
    res["mcnemar_p_holm_max"] = holm(res["mcnemar_p_max"].fillna(1.0))
    res["significant"] = (res.ttest_p_holm < 0.05) & (res.mcnemar_p_holm_max < 0.05) & res.mcnemar_same_sign
    res.round(5).to_csv(os.path.join(out_dir, "tableII_tests.csv"), index=False)
    print("\nPaired tests (reference minus baseline, accuracy; Holm over all rows):\n", res.round(4).to_string())
    print("\nSaved:", out_dir)


if __name__ == "__main__":
    if ONLY_SUMMARIZE:
        resolve_inputs()
        summarize(summary_dirs())
    else:
        main()
