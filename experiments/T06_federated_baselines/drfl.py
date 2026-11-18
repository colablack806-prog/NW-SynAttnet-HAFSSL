# -*- coding: utf-8 -*-
"""
Run once with SUPERVISED_FEDAVG_CONTROL = False (DRFL, results FED_ISCX_drfl) and once with True
(control: plain FedAvg on the same labeled flows, no DRFL mechanism, results FED_ISCX_drfl_control).

DRFL baseline for the R2 revision (Reviewer 1, comment 2): head-to-head with HAFSSL and FL-AECNN.

Method: RFNTC from S. Shi, Y. Guo, D. Wang, Y. Zhu, Z. Han, "Distributionally Robust Federated Learning for
Network Traffic Classification With Noisy Labels," IEEE Trans. Mobile Comput., 23(5):6212-6226, 2024.
(Reviewer 1 calls it DRFL; the paper's algorithm is RFNTC.)

SAME AS TABLE VII (03_federated_table7_R2.py), so the comparison is paired:
  data file, duplicate removal, frozen split (SPLIT_SEED 2026), Dirichlet alpha 0.5, K = 10 clients (all take part
  every round), the client partition and the labeled flows of each seed (same functions; checked against the
  partition files saved by Table VII), 50 rounds, 5 local epochs, batch 256, Adam 1e-3, FedAvg, the global model
  after the last round tested once on the frozen test set.

RFNTC as in the paper (Sections III-V), every round, on every client:
  1. split its labeled flows with the received global model (small-loss criterion): loss <= zeta -> "certain";
     zeta = mean loss of the client's flows (the paper's default; Fig. 6)
  2. send the count, mean and covariance of its certain flows' logits; the server forms the global Gaussian N(m, S)
  3. normalize the uncertain flows in logit space with the Euler-Maruyama OU scheme of Eqs. (18)/(27):
         z <- z - tau (z - m) dt + sqrt(2 dt / tau) S^(1/2) N(0, I)      (T steps)
  4. train on certain + normalized uncertain flows (Eq. 8); FedAvg with weights d_i / D (Eq. 9)

Choices where the paper is not specific, or the benchmark differs (state these in the paper):
  - unknown-class branch (class M+1 with confidence threshold delta) is OFF by default: this benchmark has clean
    labels and no unknown application, so every uncertain flow is "uncertain known". UNKNOWN_CLASS = True turns it
    on (delta = 0.5, the best value in Fig. 7; Eq. (23) writes a MAX, which would send almost every uncertain
    flow to "unknown", so a fixed delta is used as in Fig. 7)
  - global distribution: the exact mean and covariance of all clients' certain logits (pooled from each client's
    count, mean and covariance) instead of the FMCMCDR Bayesian estimator; these are the two moments Eq. (27) uses
  - OU parameters are not reported: tau = 1, dt = 0.1, T = 10
  - WARMUP_ROUNDS = 5 rounds of plain FedAvg before the first small-loss split (an untrained model cannot rank losses)
  - classifier: ARCH "papercnn" = the paper's CNN in 1-D for 66 flow features: Conv(9) 32 -> Conv(9) 64 -> FC 128 +
    dropout 0.5 -> FC output (the paper does not give the hidden size or the dropout rate);
    ARCH "aecnn" = the FL-AECNN encoder + classifier, i.e. the baselines' backbone without the autoencoder stage
  - DRFL is fully supervised: each client trains on its LABELED flows only (the method has no unlabeled-data stage);
    clients without labeled flows do not train and get weight 0
  - labels are clean, as for HAFSSL and FL-AECNN. DRFL was designed for noisy labels, so this setting does not use
    its main strength; say so when you report it

KAGGLE
  1. Add the ISCX csv as input (the script also searches /kaggle/input for the file name).
  2. Add the Table VII output as input; set PARTITION_CHECK_DIR (.../T7_gpu0/splits) and COMPARE_DIRS (.../T7_gpu0).
  3. QUICK_TEST = True -> Save & Run All (a few minutes). Then QUICK_TEST = False -> Save & Run All.
  4. If it stops ("STOPPED SAFELY"), add this notebook's output as input, set RESUME_FROM, run again.
Outputs: OUT_DIR/results/all_runs.csv, preds/, reports/, splits/, config_drfl.json,
         drfl_comparison_summary.csv, drfl_comparison_tests.csv, drfl_comparison_mcnemar_per_seed.csv
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
ARCHS     = ["papercnn", "aecnn"]          # "papercnn" = DRFL's own CNN, "aecnn" = same backbone as FL-AECNN
SEEDS     = [0, 1, 2]
OUT_DIR   = "/kaggle/working/FED_ISCX_drfl"

# Table VII setting (do not change: it makes the comparison paired)
ALPHA             = 0.5
NUM_CLIENTS       = 10
ROUNDS            = 50
LOCAL_EPOCHS      = 5
BATCH             = 256
LR                = 1e-3
LABELED_PER_CLASS = 1000
SPLIT_SEED        = 2026
COLLAPSE_RECALL   = 0.10

# RFNTC settings (see the docstring)
ZETA_MULT     = 1.0      # small-loss threshold = ZETA_MULT x mean loss
UNKNOWN_CLASS = False    # True = extra class M+1 for uncertain flows with confidence < DELTA
DELTA         = 0.5
OU_TAU, OU_DT, OU_T = 1.0, 0.1, 10
WARMUP_ROUNDS = 5
SUPERVISED_FEDAVG_CONTROL = False   # True = CONTROL: same script, same labeled flows, but plain FedAvg in every round
                                    # (no small-loss split, no normalization) -> shows what DRFL itself adds
COV_RIDGE     = 1e-4
LATENT_DIM    = 39

PARTITION_CHECK_DIR = ""   # e.g. the unzipped T07 results: .../T7_gpu0/splits
COMPARE_DIRS        = []   # e.g. [".../T7_gpu0", ".../FED_ISCX_drfl"] (unzipped results)
#   ^ your finished DRFL run (it must be added as an input: Add Input -> Your Work -> fed-iscx-drfl).
#     If the path is wrong, the script stops at once and lists the result folders it can see.
COMPARE_AGGREGATOR  = "FedAvg"

RESUME_FROM       = ""       # Kaggle: previous version's output folder, to continue after a stop
TIME_BUDGET_HOURS = 11.0
QUICK_TEST        = False    # True = 2 rounds, 1 local epoch, 1 warm-up round, seed 0
ONLY_COMPARE      = False    # True = no training; only the comparison with COMPARE_DIRS

if QUICK_TEST:
    ROUNDS, LOCAL_EPOCHS, SEEDS, WARMUP_ROUNDS = 2, 1, [0], 1
    OUT_DIR = OUT_DIR + "_quicktest"
if SUPERVISED_FEDAVG_CONTROL:
    WARMUP_ROUNDS = ROUNDS                  # every round is plain FedAvg: the DRFL mechanism is never used
    OUT_DIR = OUT_DIR + "_control"
METHOD_PREFIX = "SupFedAvg" if SUPERVISED_FEDAVG_CONTROL else "DRFL"
N_CLASSES = len(LABELS)
N_OUT = N_CLASSES + (1 if UNKNOWN_CLASS else 0)
SAFE_LABELS = [l.replace(" ", "_") for l in LABELS]
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


def data_path():
    if os.path.exists(DATA_PATH):
        return DATA_PATH
    name = os.path.basename(DATA_PATH)
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    if not hits:
        raise FileNotFoundError(f"{name} not found. Add it as input or fix DATA_PATH.")
    print("Data file found at", hits[0])
    return hits[0]


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


def save_config(path_used):
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = dict(script="03_federated_drfl_R2.py", method="RFNTC (DRFL), Shi et al., IEEE TMC 2024",
               dataset=DATASET, data_path=path_used, labels=LABELS, archs=ARCHS, seeds=SEEDS, alpha=ALPHA,
               clients=NUM_CLIENTS, rounds=ROUNDS, local_epochs=LOCAL_EPOCHS, batch=BATCH, lr=LR,
               labeled_per_class=LABELED_PER_CLASS, split_seed=SPLIT_SEED, zeta_mult=ZETA_MULT,
               unknown_class=UNKNOWN_CLASS, delta=DELTA, ou_tau=OU_TAU, ou_dt=OU_DT, ou_T=OU_T,
               warmup_rounds=WARMUP_ROUNDS, supervised_fedavg_control=SUPERVISED_FEDAVG_CONTROL,
               global_distribution="exact pooled mean and covariance of the clients' certain logits",
               quick_test=QUICK_TEST, tf_version=tf.__version__)
    with open(os.path.join(OUT_DIR, "config_drfl.json"), "w") as f:
        json.dump(cfg, f, indent=2)


# =============================== DATA (identical to Table VII) ===============================
def load_data():
    path = data_path()
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c != 'label']
    n_before = len(df)
    df = df.drop_duplicates(subset=feat_cols).reset_index(drop=True)
    print(f"Rows: {n_before} -> {len(df)} after removing {n_before - len(df)} exact duplicates")
    X = df[feat_cols].values.astype('float32').reshape(len(df), len(feat_cols), 1)
    y = df['label'].values.astype(int)
    return X, y, len(feat_cols), path


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


def sample_labeled(y_train, per_class, seed):
    """Same function as Table VII and the centralized runs: same seed and budget = same labeled flows."""
    rng = np.random.default_rng(seed)
    chosen = []
    for c in np.unique(y_train):
        pool = np.where(y_train == c)[0]
        chosen.extend(rng.choice(pool, min(per_class, len(pool)), replace=False))
    return np.array(chosen)


def partition_clients(y_train, kind, alpha, seed, n_clients, min_size=10):
    """Identical to 03_federated_hafssl_R2.py and 03_federated_table7_R2.py (same seed = same partition)."""
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


def partition_file(folder, seed):
    return os.path.join(folder, f"{DATASET}_partition_a{ALPHA}_K{NUM_CLIENTS}_L{LABELED_PER_CLASS}_s{seed}.npz")


def save_and_check_partition(parts, labeled_idx, seed):
    """Saves this run's clients and labeled flows, and stops if they differ from Table VII's saved ones."""
    arrays = {f"client_{k}": p for k, p in enumerate(parts)}
    arrays["labeled"] = np.sort(labeled_idx)
    own = partition_file(os.path.join(OUT_DIR, "splits"), seed)
    targets = [own] + ([partition_file(PARTITION_CHECK_DIR, seed)] if PARTITION_CHECK_DIR else [])
    for path in targets:
        if os.path.exists(path):
            old = np.load(path)
            for key, arr in arrays.items():
                if key not in old.files or not np.array_equal(old[key], arr):
                    raise RuntimeError(f"Seed {seed}: clients or labeled flows differ from {path} ({key}). "
                                       "Check that the data file and split are the same as in Table VII.")
            if path != own:
                print(f"  Pairing check passed: same clients and labeled flows as {path}")
        elif path != own:
            print(f"  WARNING: Table VII partition file not found, pairing not checked: {path}")
    if not os.path.exists(own):
        os.makedirs(os.path.dirname(own), exist_ok=True)
        np.savez_compressed(own, **arrays)


# =============================== MODEL ===============================
def build_model(arch, n_features):
    """Returns pre-softmax logits (RFNTC works in this space)."""
    inp = layers.Input(shape=(n_features, 1))
    if arch == "papercnn":                      # Shi et al., Sec. VII-A, made 1-D for 66 flow features
        x = layers.Conv1D(32, 9, padding='same', activation='relu')(inp)
        x = layers.Conv1D(64, 9, padding='same', activation='relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
    elif arch == "aecnn":                       # FL-AECNN encoder + classifier head (no decoder, no AE stage)
        x = layers.Conv1D(64, 3, padding='same', activation='relu')(inp)
        x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
        x = layers.MaxPooling1D()(x)
        x = layers.Conv1D(32, 3, padding='same', activation='relu')(x)
        x = layers.Conv1D(32, 3, padding='same', activation='relu')(x)
        steps, ch = x.shape[1], x.shape[2]
        x = layers.Dense(LATENT_DIM, activation='relu')(layers.Flatten()(x))
        x = layers.Reshape((steps, ch))(layers.Dense(steps * ch)(x))
        x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
        x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
        x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dense(128, activation='relu')(layers.Flatten()(x))
    else:
        raise ValueError(arch)
    return Model(inp, layers.Dense(N_OUT)(x), name=f"drfl_{arch}")


def logits_of(model, x):
    return model.predict(x, batch_size=2048, verbose=0).astype(np.float64)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# =============================== RFNTC: CLIENT SPLIT, GLOBAL GAUSSIAN, LOCAL TRAINING ===============================
def client_partition(model, x, y):
    """kind 0 = certain, 1 = uncertain (known), 2 = uncertain unknown (only if UNKNOWN_CLASS).
    Also returns what the client sends: count, mean and covariance of its certain flows' logits."""
    z = logits_of(model, x)
    zs = z - z.max(axis=1, keepdims=True)
    logp = zs - np.log(np.exp(zs).sum(axis=1, keepdims=True))
    loss = -logp[np.arange(len(y)), y]
    certain = loss <= ZETA_MULT * loss.mean()
    kind = np.where(certain, 0, 1)
    if UNKNOWN_CLASS:
        conf = softmax(z)[:, :N_CLASSES].max(axis=1)
        kind = np.where(certain, 0, np.where(conf >= DELTA, 1, 2))
    zc = z[certain]
    mean = zc.mean(axis=0) if len(zc) else np.zeros(N_OUT)
    cov = np.cov(zc.T, bias=True) if len(zc) > 1 else np.zeros((N_OUT, N_OUT))
    return kind, len(zc), mean, np.atleast_2d(cov)


def global_gaussian(stats_list):
    """Mean and covariance of the union of all clients' certain logits, from (count, mean, cov) per client."""
    n = np.array([s[0] for s in stats_list], dtype=np.float64)
    if n.sum() == 0:
        return np.zeros(N_OUT, np.float32), np.eye(N_OUT, dtype=np.float32)
    w = n / n.sum()
    m = sum(wi * s[1] for wi, s in zip(w, stats_list))
    second = sum(wi * (s[2] + np.outer(s[1], s[1])) for wi, s in zip(w, stats_list))
    cov = second - np.outer(m, m) + COV_RIDGE * np.eye(N_OUT)
    vals, vecs = np.linalg.eigh((cov + cov.T) / 2)
    root = vecs @ np.diag(np.sqrt(np.clip(vals, 0, None)))          # root @ root.T = cov
    return m.astype(np.float32), root.astype(np.float32)


def reset_optimizer(opt):
    for v in opt.variables:
        if "learning_rate" not in v.path:
            v.assign(tf.zeros_like(v))


def make_trainer(model):
    tv = model.trainable_variables
    opt = tf.keras.optimizers.Adam(LR)
    opt.build(tv)
    c_noise = float(np.sqrt(2.0 * OU_DT / OU_TAU))

    @tf.function(reduce_retracing=True)
    def step(x, y, normalize, noise, m, root):
        with tf.GradientTape() as tape:
            z = model(x, training=True)
            zt = z
            for t in range(OU_T):                                   # Euler-Maruyama, Eq. (18) / (27)
                zt = zt - OU_TAU * (zt - m) * OU_DT + c_noise * tf.matmul(noise[t], root, transpose_b=True)
            zu = tf.where(normalize[:, None], zt, z)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=zu))
        opt.apply_gradients(zip(tape.gradient(loss, tv), tv))
        return loss

    def train_client(x, y_int, kind, m, root, rng):
        reset_optimizer(opt)
        y_used = y_int.copy()
        if UNKNOWN_CLASS:
            y_used[kind == 2] = N_CLASSES                            # uncertain unknown -> class M+1
        y_oh = tf.keras.utils.to_categorical(y_used, N_OUT).astype('float32')
        norm = kind == 1                                             # only uncertain-known flows are normalized
        m_t, r_t = tf.constant(m), tf.constant(root)
        steps = 0
        for _ in range(LOCAL_EPOCHS):
            order = rng.permutation(len(x))
            for i in range(0, len(order), BATCH):
                b = order[i:i + BATCH]
                noise = rng.standard_normal((OU_T, len(b), N_OUT)).astype('float32')
                step(tf.constant(x[b]), tf.constant(y_oh[b]), tf.constant(norm[b]), tf.constant(noise), m_t, r_t)
                steps += 1
        return steps

    return train_client


# =============================== CHECKPOINTS ===============================
class StopForNow(Exception):
    pass


def ckpt_path(run_id):
    return os.path.join(OUT_DIR, "checkpoints", f"{run_id}.npz")


def save_ckpt(run_id, meta, global_w, model):
    os.makedirs(os.path.dirname(ckpt_path(run_id)), exist_ok=True)
    arr = {f"g__{i}": w for i, w in enumerate(global_w)}
    arr.update({f"l__{i}": np.asarray(v.numpy()) for i, v in enumerate(model.variables)})   # incl. dropout state
    arr["meta"] = np.array(json.dumps(meta))
    tmp = ckpt_path(run_id) + ".tmp.npz"
    np.savez(tmp, **arr)
    os.replace(tmp, ckpt_path(run_id))


def load_ckpt(run_id, model, n_global):
    d = np.load(ckpt_path(run_id), allow_pickle=False)
    for i, v in enumerate(model.variables):
        v.assign(d[f"l__{i}"])
    return json.loads(str(d["meta"])), [d[f"g__{i}"] for i in range(n_global)]


def check_time(last_round_seconds):
    if _STOP_AFTER_ROUNDS and _rounds_this_session >= _STOP_AFTER_ROUNDS:
        raise StopForNow()
    if time.time() - SESSION_START + 1.3 * last_round_seconds > TIME_BUDGET_HOURS * 3600:
        raise StopForNow()


# =============================== EVALUATION ===============================
def evaluate_logits(z, y):
    pred = z[:, :N_CLASSES].argmax(axis=1)                          # the test set has no unknown traffic
    p, r, f1, _ = precision_recall_fscore_support(y, pred, labels=list(range(N_CLASSES)),
                                                  average='macro', zero_division=0)
    m = {'acc': accuracy_score(y, pred), 'macro_precision': p, 'macro_recall': r, 'macro_f1': f1,
         'balanced_acc': balanced_accuracy_score(y, pred)}
    rec = confusion_matrix(y, pred, labels=list(range(N_CLASSES))).diagonal() / np.bincount(y, minlength=N_CLASSES)
    return m, pred, rec


def results_csv(folder=None):
    return os.path.join(folder or OUT_DIR, "results", "all_runs.csv")


def already_done(run_id):
    return os.path.exists(results_csv()) and run_id in set(pd.read_csv(results_csv())['run_id'])


def save_run(run_id, row, y_test, pred, prob, test_idx, meta, client_classes):
    for sub in ("results", "preds", "reports"):
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)
    np.savez_compressed(os.path.join(OUT_DIR, "preds", f"{run_id}.npz"),
                        y_true=y_test, y_pred=pred, y_prob=prob, test_idx=test_idx)
    rep = classification_report(y_test, pred, labels=list(range(N_CLASSES)), target_names=LABELS,
                                digits=4, output_dict=True, zero_division=0)
    pd.DataFrame(rep).transpose().to_csv(os.path.join(OUT_DIR, "reports", f"{run_id}_per_class.csv"))
    pd.DataFrame(confusion_matrix(y_test, pred, labels=list(range(N_CLASSES))),
                 index=LABELS, columns=LABELS).to_csv(os.path.join(OUT_DIR, "reports", f"{run_id}_confusion.csv"))
    pd.DataFrame(meta["hist"]).to_csv(os.path.join(OUT_DIR, "reports", f"{run_id}_rounds.csv"), index=False)
    client_classes.to_csv(os.path.join(OUT_DIR, "reports", f"{run_id}_client_classes.csv"))
    path = results_csv()
    pd.DataFrame([row]).to_csv(path, mode='a', header=not os.path.exists(path), index=False)


# =============================== ONE RUN ===============================
def run_one(run_id, arch, seed, data, plat):
    global _rounds_this_session
    X_tr, y_tr, x_val, y_val, x_te, y_te, test_idx, n_features = data
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = build_model(arch, n_features)
    train_client = make_trainer(model)

    parts = partition_clients(y_tr, "dirichlet", ALPHA, seed, NUM_CLIENTS)
    labeled_idx = sample_labeled(y_tr, LABELED_PER_CLASS, seed)
    save_and_check_partition(parts, labeled_idx, seed)
    labeled = np.zeros(len(y_tr), dtype=bool)
    labeled[labeled_idx] = True
    clients = [(X_tr[idx[labeled[idx]]], y_tr[idx[labeled[idx]]]) for idx in parts]   # DRFL: labeled flows only
    client_classes = pd.DataFrame([np.bincount(y_tr[idx], minlength=N_CLASSES) for idx in parts],
                                  columns=LABELS, index=[f"client_{k}" for k in range(NUM_CLIENTS)])
    client_classes["labeled"] = [len(c[1]) for c in clients]
    d_k = [len(c[1]) for c in clients]
    active = [k for k in range(NUM_CLIENTS) if d_k[k] > 0]

    n_params = int(model.count_params())
    stats_bytes = 4 * (2 * (N_OUT + N_OUT * N_OUT) + 1)          # up: count, mean, cov; down: mean, cov
    bytes_model = len(active) * 2 * 4 * n_params                 # float32 weights, up + down
    global_w = model.get_weights()
    meta = {"round_done": 0, "hist": [], "train_seconds": 0.0, "last_round_seconds": 0.0, "bytes": 0}
    if os.path.exists(ckpt_path(run_id)):
        meta, global_w = load_ckpt(run_id, model, len(global_w))
        print(f"  Resuming from checkpoint: {meta['round_done']} rounds done")

    for rnd in range(meta["round_done"] + 1, ROUNDS + 1):
        check_time(meta["last_round_seconds"])
        t0 = time.time()
        warm = rnd <= WARMUP_ROUNDS
        kinds, stats_list = {}, []
        for k in active:                                         # 1-2: split + certain-logit moments
            if warm:
                kinds[k] = np.zeros(d_k[k], dtype=int)
            else:
                model.set_weights(global_w)
                kinds[k], n_c, mean, cov = client_partition(model, *clients[k])
                stats_list.append((n_c, mean, cov))
        if warm:
            m, root = np.zeros(N_OUT, np.float32), np.zeros((N_OUT, N_OUT), np.float32)
        else:
            m, root = global_gaussian(stats_list)
        client_ws, n_k = [], []
        for k in active:                                         # 3-4: normalize + local training
            tf.keras.utils.set_random_seed(seed * 100000 + rnd * 100 + k)
            rng = np.random.default_rng(seed * 100000 + rnd * 100 + k)
            model.set_weights(global_w)
            train_client(clients[k][0], clients[k][1], kinds[k], m, root, rng)
            client_ws.append(model.get_weights())
            n_k.append(d_k[k])
        p = np.asarray(n_k, dtype=np.float64) / np.sum(n_k)        # FedAvg, d_i / D
        global_w = [sum(pk * w[i] for pk, w in zip(p, client_ws)).astype(np.float32) for i in range(len(global_w))]
        meta["bytes"] += bytes_model + (0 if warm else len(active) * stats_bytes)

        model.set_weights(global_w)
        vm, _, vrec = evaluate_logits(logits_of(model, x_val), y_val)
        allk = np.concatenate([kinds[k] for k in active])
        sec = time.time() - t0
        meta["hist"].append({"round": rnd, "warmup": warm, "val_acc": round(vm["acc"], 6),
                             "val_macro_f1": round(vm["macro_f1"], 6), "val_min_recall": round(float(vrec.min()), 6),
                             "share_certain": round(float(np.mean(allk == 0)), 4),
                             "share_uncertain": round(float(np.mean(allk == 1)), 4),
                             "share_unknown": round(float(np.mean(allk == 2)), 4),
                             "comm_MB_cum": round(meta["bytes"] / 1e6, 3), "seconds": round(sec, 1)})
        meta.update(round_done=rnd, last_round_seconds=sec, train_seconds=meta["train_seconds"] + sec)
        print(f"  round {rnd}/{ROUNDS}{' (warm-up)' if warm else ''}: val acc {vm['acc']:.4f} | "
              f"val macro-F1 {vm['macro_f1']:.4f} | certain {np.mean(allk == 0):.2f} "
              f"uncertain {np.mean(allk == 1):.2f} unknown {np.mean(allk == 2):.2f} | {sec:.0f}s")
        save_ckpt(run_id, meta, global_w, model)
        _rounds_this_session += 1

    model.set_weights(global_w)
    z_te = logits_of(model, x_te)
    m_te, pred, rec = evaluate_logits(z_te, y_te)
    prob = softmax(z_te[:, :N_CLASSES])
    unknown_share = float(np.mean(z_te.argmax(axis=1) == N_CLASSES)) if UNKNOWN_CLASS else 0.0
    hist = pd.DataFrame(meta["hist"])
    r90 = hist.loc[hist.val_acc >= 0.90, "round"]
    last10 = hist.tail(10)
    row = dict(run_id=run_id, dataset=DATASET, setting="federated", method=f"{METHOD_PREFIX}-{arch}", variant=METHOD_PREFIX,
               arch=arch, aggregator="FedAvg", partition="dirichlet", alpha=ALPHA, clients=NUM_CLIENTS,
               labels_per_class=LABELED_PER_CLASS, labels_total=int(labeled.sum()), seed=seed, rounds=ROUNDS,
               local_epochs=LOCAL_EPOCHS, batch=BATCH, lr=LR,
               **{k: round(float(v), 6) for k, v in m_te.items()},
               min_class_recall=round(float(rec.min()), 6), collapse=bool(rec.min() < COLLAPSE_RECALL),
               **{f"recall_{l}": round(float(r), 6) for l, r in zip(SAFE_LABELS, rec)},
               val_acc_last10=round(float(last10.val_acc.mean()), 6),
               val_f1_last10=round(float(last10.val_macro_f1.mean()), 6),
               rounds_to_90=int(r90.iloc[0]) if len(r90) else -1, unknown_share=round(unknown_share, 6),
               share_certain_last=float(hist.share_certain.iloc[-1]),
               share_uncertain_last=float(hist.share_uncertain.iloc[-1]),
               share_unknown_last=float(hist.share_unknown.iloc[-1]),
               params_sent=n_params, MB_total=round(meta["bytes"] / 1e6, 1),
               MB_per_round=round(meta["bytes"] / 1e6 / ROUNDS, 3),
               smallest_client=int(min(len(i) for i in parts)), largest_client=int(max(len(i) for i in parts)),
               active_clients=len(active), zeta_mult=ZETA_MULT, unknown_class=UNKNOWN_CLASS, delta=DELTA,
               ou_tau=OU_TAU, ou_dt=OU_DT, ou_T=OU_T, warmup_rounds=WARMUP_ROUNDS,
               train_seconds=round(meta["train_seconds"], 1), platform=plat, gpu=gpu_name(),
               tf_version=tf.__version__, timestamp=time.strftime("%Y-%m-%d %H:%M"))
    save_run(run_id, row, y_te, pred, prob, test_idx, meta, client_classes)
    os.remove(ckpt_path(run_id))
    print(f"Done {run_id}: test acc {m_te['acc']:.4f} | macro-F1 {m_te['macro_f1']:.4f} | lowest class recall "
          f"{rec.min():.3f}" + ("  <-- COLLAPSE" if rec.min() < COLLAPSE_RECALL else ""))


# =============================== COMPARISON WITH HAFSSL / FL-AECNN ===============================
def mcnemar_exact(y, pred_a, pred_b):
    a_ok, b_ok = pred_a == y, pred_b == y
    n01, n10 = int(np.sum(a_ok & ~b_ok)), int(np.sum(~a_ok & b_ok))
    if n01 + n10 == 0:
        return n01, n10, 1.0
    return n01, n10, float(stats.binomtest(n01, n01 + n10, 0.5).pvalue)


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def compare():
    """Table rows and paired tests: HAFSSL / FL-AECNN vs DRFL and the control, and control vs DRFL."""
    runs = []
    for folder in [OUT_DIR] + list(COMPARE_DIRS):
        if os.path.exists(results_csv(folder)):
            r = pd.read_csv(results_csv(folder))
            r["folder"] = folder
            runs.append(r)
    if not runs:
        print("No results to compare.")
        return
    res = pd.concat(runs, ignore_index=True).drop_duplicates("run_id", keep="first")
    res = res[(res.aggregator == COMPARE_AGGREGATOR) & (res.partition == "dirichlet") & np.isclose(res.alpha, ALPHA)
              & (res.clients == NUM_CLIENTS) & (res.labels_per_class == LABELED_PER_CLASS)
              & (res.rounds == ROUNDS) & (res.local_epochs == LOCAL_EPOCHS)].copy()
    res["seed"] = res.seed.astype(int)
    methods = set(res.method)
    baselines = [m for m in ("HAFSSL", "FL-AECNN") if m in methods]
    others = sorted(m for m in methods if str(m).startswith(("DRFL", "SupFedAvg")))
    pairs = [(a, b) for a in baselines for b in others]
    pairs += [(f"SupFedAvg-{x}", f"DRFL-{x}") for x in ("papercnn", "aecnn")
              if {f"SupFedAvg-{x}", f"DRFL-{x}"} <= methods]          # what the DRFL mechanism adds
    if not pairs:
        print(f"Comparison skipped: found methods {sorted(methods)} (need HAFSSL / FL-AECNN, DRFL or SupFedAvg).")
        return

    order = baselines + [m for m in ("SupFedAvg-papercnn", "DRFL-papercnn", "SupFedAvg-aecnn", "DRFL-aecnn") if m in others]
    tab = (res.groupby("method").agg(n_seeds=("seed", "nunique"), acc_mean=("acc", "mean"), acc_std=("acc", "std"),
                                     f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std"),
                                     min_recall_mean=("min_class_recall", "mean"),
                                     min_recall_worst=("min_class_recall", "min"),
                                     collapse_runs=("collapse", "sum"), params_sent=("params_sent", "first"))
           .reindex(order).reset_index())
    tab.to_csv(os.path.join(OUT_DIR, "drfl_comparison_summary.csv"), index=False)
    pct = lambda m, s: f"{100 * m:.2f} ± {100 * s:.2f}"
    show = tab.assign(acc=[pct(a, b) for a, b in zip(tab.acc_mean, tab.acc_std)],
                      macro_f1=[pct(a, b) for a, b in zip(tab.f1_mean, tab.f1_std)],
                      lowest_recall_worst_seed=(100 * tab.min_recall_worst).round(1))
    print("\nComparison (test set, mean ± std over seeds, %):")
    print(show[["method", "n_seeds", "acc", "macro_f1", "lowest_recall_worst_seed", "collapse_runs",
                "params_sent"]].to_string(index=False))

    per_seed = []
    cols = LABELS + ["labeled"]
    for a, b in pairs:
        ra, rb = res[res.method == a].set_index("seed"), res[res.method == b].set_index("seed")
        for s in sorted(set(ra.index) & set(rb.index)):
            fa, fb = ra.loc[s, "folder"], rb.loc[s, "folder"]
            pa = np.load(os.path.join(fa, "preds", f"{ra.loc[s, 'run_id']}.npz"))
            pb = np.load(os.path.join(fb, "preds", f"{rb.loc[s, 'run_id']}.npz"))
            if not (np.array_equal(pa["test_idx"], pb["test_idx"]) and np.array_equal(pa["y_true"], pb["y_true"])):
                raise RuntimeError(f"Seed {s}: {a} and {b} were not tested on the same flows.")
            ca = pd.read_csv(os.path.join(fa, "reports", f"{ra.loc[s, 'run_id']}_client_classes.csv"), index_col=0)
            cb = pd.read_csv(os.path.join(fb, "reports", f"{rb.loc[s, 'run_id']}_client_classes.csv"), index_col=0)
            if not np.array_equal(ca[cols].values, cb[cols].values):
                raise RuntimeError(f"Seed {s}: {a} and {b} used different clients or labeled flows.")
            n01, n10, pv = mcnemar_exact(pa["y_true"], pa["y_pred"], pb["y_pred"])
            per_seed.append(dict(reference=a, compared=b, seed=s, acc_reference=ra.loc[s, "acc"],
                                 acc_compared=rb.loc[s, "acc"], diff=ra.loc[s, "acc"] - rb.loc[s, "acc"],
                                 only_reference_correct=n01, only_compared_correct=n10, mcnemar_p=pv))
    if not per_seed:
        print("No paired seeds found.")
        return
    ps = pd.DataFrame(per_seed)
    ps["mcnemar_p_holm"] = holm(ps.mcnemar_p)                       # family: every per-seed McNemar test
    ps.to_csv(os.path.join(OUT_DIR, "drfl_comparison_mcnemar_per_seed.csv"), index=False)
    tests = []
    for (a, b), g in ps.groupby(["reference", "compared"], sort=False):
        d = g["diff"].values
        tp = float(stats.ttest_rel(g.acc_reference, g.acc_compared).pvalue) if len(d) >= 2 and np.std(d) > 0 else np.nan
        half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)) if len(d) >= 2 else np.nan
        tests.append(dict(reference=a, compared=b, n_seeds=len(d), mean_diff_pp=100 * d.mean(),
                          ci95_low_pp=100 * (d.mean() - half), ci95_high_pp=100 * (d.mean() + half),
                          per_seed_pp=np.round(100 * d, 2).tolist(), ttest_p=tp,
                          mcnemar_all_seeds_sig=bool((g.mcnemar_p_holm < 0.05).all()
                                                     and (np.sign(d) == np.sign(d.mean())).all())))
    tt = pd.DataFrame(tests)
    ok = tt.ttest_p.notna()
    tt["ttest_p_holm"] = np.nan
    if ok.any():
        tt.loc[ok, "ttest_p_holm"] = holm(tt.loc[ok, "ttest_p"])     # family: all comparisons in this table
    tt["reference_better"] = (tt.ttest_p_holm < 0.05) & tt.mcnemar_all_seeds_sig & (tt.mean_diff_pp > 0)
    tt["compared_better"] = (tt.ttest_p_holm < 0.05) & tt.mcnemar_all_seeds_sig & (tt.mean_diff_pp < 0)
    tt.to_csv(os.path.join(OUT_DIR, "drfl_comparison_tests.csv"), index=False)
    print("\nReference minus compared, accuracy (percentage points), paired by seed:")
    print(tt.round(4).to_string(index=False))
    print("\nreference_better / compared_better = paired t-test (Holm) AND McNemar (Holm) significant in every seed, same sign.")


# =============================== MAIN ===============================
def check_input_paths():
    """Stops before any training if an input folder is wrong, and lists the result folders that exist."""
    wanted = ([PARTITION_CHECK_DIR] if PARTITION_CHECK_DIR else []) + list(COMPARE_DIRS)
    missing = [d for d in wanted if not os.path.isdir(d)]
    if not missing:
        return
    found = sorted({os.path.dirname(os.path.dirname(f))
                    for f in glob.glob("/kaggle/input/**/results/all_runs.csv", recursive=True)})
    raise FileNotFoundError("These input folders do not exist:\n  " + "\n  ".join(missing)
                            + "\nResult folders found under /kaggle/input (copy the right path from here):\n  "
                            + ("\n  ".join(found) or "(none: add the notebook or dataset as an input first)"))


def main():
    check_input_paths()
    if ONLY_COMPARE:
        os.makedirs(OUT_DIR, exist_ok=True)
        compare()
        return
    plat = platform_name()
    restore_previous_output()
    tf.config.experimental.enable_op_determinism()
    X, y, n_features, path_used = load_data()
    save_config(path_used)
    tr, va, te = get_split(y)
    if PARTITION_CHECK_DIR:                                       # same frozen split as Table VII
        t7 = os.path.join(PARTITION_CHECK_DIR, f"{DATASET}_split.npz")
        if os.path.exists(t7):
            s = np.load(t7)
            if not all(np.array_equal(s[k], v) for k, v in (("train", tr), ("val", va), ("test", te))):
                raise RuntimeError(f"The data split differs from Table VII's ({t7}).")
            print("Split check passed: same train/val/test split as Table VII")
        else:
            print(f"  WARNING: Table VII split file not found, split not checked: {t7}")
    data = (X[tr], y[tr], X[va], y[va], X[te], y[te], te, n_features)
    plan = [(seed, arch) for seed in SEEDS for arch in ARCHS]
    print(f"{'CONTROL: supervised FedAvg (no DRFL mechanism)' if SUPERVISED_FEDAVG_CONTROL else 'DRFL (RFNTC)'} | {DATASET} | train {len(tr)} | val {len(va)} | test {len(te)} | {plat} | {gpu_name()}")
    print(f"archs {ARCHS} | seeds {SEEDS} | alpha {ALPHA} | K {NUM_CLIENTS} | rounds {ROUNDS} (warm-up {WARMUP_ROUNDS}) | "
          f"local epochs {LOCAL_EPOCHS} | labels/class {LABELED_PER_CLASS} | zeta x{ZETA_MULT} | "
          f"OU tau {OU_TAU}, dt {OU_DT}, T {OU_T} | unknown class {UNKNOWN_CLASS} | QUICK_TEST {QUICK_TEST}")
    try:
        for seed, arch in plan:
            run_id = f"{DATASET}_{METHOD_PREFIX}-{arch}_FedAvg_a{ALPHA}_K{NUM_CLIENTS}_L{LABELED_PER_CLASS}_s{seed}"
            if already_done(run_id):
                print("Skip (already done):", run_id)
                continue
            print(f"\n===== {run_id} =====")
            run_one(run_id, arch, seed, data, plat)
    except StopForNow:
        print("\n" + "=" * 70)
        print("STOPPED SAFELY (time budget). All progress is saved in:", OUT_DIR)
        print("Kaggle: add this version's output as input, set RESUME_FROM, Save & Run All.")
        print("=" * 70)
        return
    print(f"\nALL {METHOD_PREFIX} RUNS FINISHED.")
    compare()


if __name__ == "__main__":
    main()
