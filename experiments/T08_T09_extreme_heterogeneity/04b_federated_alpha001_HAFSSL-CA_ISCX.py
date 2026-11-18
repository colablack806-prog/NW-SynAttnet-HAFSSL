# -*- coding: utf-8 -*-
# In the paper, MODEL = "HAFSSL" of this script is HAFSSL-CA (Table VIII, ISCX-VPN2016, 461,013 parameters):
# hybrid attention in the encoder (key_dim 32) and in the classifier head (key_dim 16), plus Dropout(0.3)
# in the head. Results: results/ISCX_HAFSSL-CA.zip. MODEL = "FL-AECNN" gives the FL-AECNN row.
# Converted from the Kaggle notebook of the reported runs (outputs removed).

# -*- coding: utf-8 -*-
"""
HAFSSL / FL-AECNN - extreme heterogeneity (Dirichlet alpha = 0.01, K = 10), Table VIII, R2 revision.

SAME MODEL AND LOCAL TRAINING AS THE ORIGINAL alpha=0.01 SCRIPT:
  model (HAFSSL) : encoder Conv64-Conv64-MaxPool-[MHA->SE, key_dim 32]-Conv32-Conv32-Dense(39)
                   classifier Dense-Reshape-Conv64-Conv64-MaxPool-[MHA->SE, key_dim 16]-Conv128-Conv128
                              -MaxPool-Dense(128)-Dropout(0.3)-Softmax
  model (FL-AECNN): same layers WITHOUT any attention block and without Dropout(0.3)  [10]
  every round, on each client:
    stage 1   : autoencoder on the client's UNLABELED flows, 3 epochs, batch 32, Adam (skipped if < 50 flows)
    stage 1.5 : encoder fine-tuned with the classifier head FROZEN, 2 epochs, batch 256, Adam lr 1e-4
    stage 2   : encoder + classifier, 3 epochs, batch 256, Adam
  labels       : 200 labeled flows per class on each client (1000 per client, as before), for the classes it holds
  aggregation  : weighted FedAvg n_k / n (Eq. 5); clients with < 50 flows are skipped, as before
  rounds       : 100

FIXES REQUIRED BY THE REVIEWERS (Reviewer 1, Comments 1 and 7):
  - seeds 0, 1, 2 (model, client partition, labeled flows) -> mean +- std over 3 independent runs
  - HAFSSL and FL-AECNN get EXACTLY the same partition and labeled flows for each seed
    (no more 1,000 vs 3,000 labels); run them in two notebooks with MODEL = "HAFSSL" / "FL-AECNN"
  - the GLOBAL model is evaluated on a frozen test set (not the average of local client models)
  - duplicates removed; same split as all other experiments; the test set is never used in training
  - macro metrics + per-class recall + confusion matrix saved for every run (degenerate classes visible)
  - checkpoint after every round; stops safely before Kaggle's 12-hour limit and resumes
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
MODEL     = "HAFSSL"          # "HAFSSL" or "FL-AECNN"  (run each in its own notebook)

DATA_PATH = "data/ISCX-VPN2016/ISCX_5class_each_normalized_cuttedfloefeature.csv"
DATASET   = "ISCX"
LABELS    = ['chat', 'email', 'file', 'streaming', 'voip']
OUT_DIR   = f"/kaggle/working/FED_ISCX_alpha001_{MODEL}"

AGGREGATORS = ["FedAvg"]       # original setting; add "FedNova" to also test gradient correction
ALPHA        = 0.01
NUM_CLIENTS  = 10
ROUNDS       = 100
LOCAL_EPOCHS = 3               # stage 1 and stage 2
STAGE15_EPOCHS, STAGE15_LR = 2, 1e-4
BATCH_AE, BATCH_CLF = 32, 256  # original: autoencoder used the Keras default batch (32)
LR = 1e-3
LABELS_PER_CLASS_PER_CLIENT = 200   # = original labelnum 1000 per client / 5 classes
MIN_CLIENT_FLOWS = 50               # clients with fewer flows are skipped (as in the original)
SEEDS = [0, 1, 2]

NUM_HEADS, KEY_DIM, SE_RATIO, LATENT_DIM, ATT_DROPOUT = 4, 32, 16, 39, 0.1
SPLIT_SEED        = 2026    # same as every other experiment = same test set
RESUME_FROM       = ""      # Kaggle: previous version's output folder, to continue after a stop
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


# =============================== DATA ===============================
def load_data():
    df = pd.read_csv(DATA_PATH)
    feat_cols = [c for c in df.columns if c != 'label']
    n_before = len(df)
    df = df.drop_duplicates(subset=feat_cols).reset_index(drop=True)
    print(f"Rows: {n_before} -> {len(df)} after removing {n_before - len(df)} exact duplicates")
    X = df[feat_cols].values.astype('float32').reshape(len(df), len(feat_cols), 1)
    return X, df['label'].values.astype(int), len(feat_cols)


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


def build_models(n_features, n_classes, use_attention):
    ie = layers.Input(shape=(n_features, 1))
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(ie)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D()(x)
    if use_attention:
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
    if use_attention:
        c = hybrid_attention_block(c, KEY_DIM // 2)            # classifier attention (key_dim 16)
    c = layers.Conv1D(128, 3, padding='same', activation='relu')(c)
    c = layers.Conv1D(128, 3, padding='same', activation='relu')(c)
    c = layers.MaxPooling1D(2)(c)
    c = layers.Dense(128, activation='relu')(layers.Flatten()(c))
    if use_attention:
        c = layers.Dropout(0.3)(c)
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

    n_params = local.n_params()
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
    row = dict(run_id=run_id, dataset=DATASET, setting="federated", method=MODEL, aggregator=aggregator,
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
    tf.config.experimental.enable_op_determinism()
    X, y, n_features = load_data()
    tr, va, te = get_split(y)
    data = (X[tr], y[tr], X[va], y[va], X[te], y[te], te, n_features, len(LABELS))
    print(f"Train {len(tr)} | Val {len(va)} | Test {len(te)} | {plat} | {gpu_name()}")
    print(f"MODEL {MODEL} | alpha {ALPHA} | K={NUM_CLIENTS} | rounds {ROUNDS} | local epochs {LOCAL_EPOCHS} "
          f"(+ stage 1.5: {STAGE15_EPOCHS}) | labels {LABELS_PER_CLASS_PER_CLIENT}/class/client | "
          f"aggregators {AGGREGATORS} | seeds {SEEDS}")
    try:
        for seed in SEEDS:
            for aggregator in AGGREGATORS:
                run_id = f"{DATASET}_{MODEL}_{aggregator}_a{ALPHA}_K{NUM_CLIENTS}_s{seed}"
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
