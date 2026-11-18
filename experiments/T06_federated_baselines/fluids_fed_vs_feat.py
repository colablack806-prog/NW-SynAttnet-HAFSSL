# -*- coding: utf-8 -*-
"""
NEW FEDERATED BASELINES (Reviewer 1, Comment 2): FLUIDS-fed and FEAT, plus CNN-FedAvg, under EXACTLY the
Table VII protocol: same de-duplicated ISCX data, frozen split (SPLIT_SEED 2026), Dirichlet alpha 0.5, K = 10,
the same client partition and the same 1,000 labeled flows per class for seeds 0, 1, 2, 50 rounds,
5 local epochs, batch 256, Adam 1e-3, global model evaluated on the same test set.
3 methods x 3 seeds = 9 runs, about 1-1.5 h on one T4.

FLUIDS-fed (Aouedi et al., CCNC 2022 [7]; architecture and training from the FLUIDS code in [10]'s repository):
  * model: Dense autoencoder 66-150-100-50-39 (decoder mirrored), MLP head 39-150-100-50-C
  * federated stage: every client trains the autoencoder on ALL its local flows (labels unused), FedAvg n_k/n,
    50 rounds x 5 local epochs, batch 256, Adam 1e-3 (same schedule as HAFSSL's AE stage)
  * server stage (FLUIDS design): the server holds the labeled flows (the same 5,000 labeled flows HAFSSL uses,
    pooled) and trains the MLP head on the FROZEN federated encoder, 100 epochs, batch 128, Adam 1e-3
    (settings of the FLUIDS code in [10]).
  * note for the paper: FLUIDS assumes labeled data at the server, i.e. its classifier is trained centrally and
    is not affected by client label skew. Only autoencoder weights are exchanged during the rounds.

FEAT (Guo & Wang, IEEE IoT-J 2023 [6]), re-implemented from the paper (no public code; authors contacted):
  * every client trains the local CNN and uploads its update dw_i = w_i - w_global (all clients, every round)
  * Thompson sampling: g = lambda*K candidates drawn one by one by argmax of theta_c ~ Beta(A_c+1, B_c+1);
    d of them chosen at random
  * reward R_i = -|| dw_i - mean_{j in S} dw_j ||^2   (Eq. 17); duels between all pairs in S: winner A += eta,
    loser B += eta
  * FedAvg of the selected clients only, weights N_i / sum_{j in S} N_j  (Eq. 2, normalised over S)
  * local model: the paper's CNN (Conv1D 32/64/128, kernel 3, each + max-pooling; FC 256, FC 256; softmax),
    supervised on the client's labeled flows (FEAT is supervised and does not use unlabeled flows)
  FEAT adaptations (state them in the paper):
    1. No multitask heads: FEAT's auxiliary targets (flow duration, bandwidth) come from flow-duration features
       that are removed in our 66-feature input.
    2. K = 10 (paper: 50). Selection ratio d/K = 0.3, the largest the paper tests on ISCX (Fig. 5b): d = 3;
       lambda = 0.6 as in the paper: g = 6.
    3. eta = 1 per duel (the paper's "eta_t = 0.01" is not separated from the model learning rate; with 0.01
       the Beta posteriors stay almost uniform over 50 rounds, i.e. random selection, which handicaps FEAT).
    4. Optimizer Adam 1e-3, batch 256, 5 local epochs: identical to HAFSSL and FL-AECNN.

CNN-FedAvg: FEAT's CNN with all clients aggregated (the paper's "Random"/FedAvg reference); shows whether
FEAT's client selection itself helps.

Kaggle: Accelerator GPU T4 (x1). Add as Input (1) the ISCX dataset and (2) your Table VII output (the notebook
version with T7_gpu0 / T7_gpu1). The cell checks that every run uses the identical Table VII partition and
labeled flows, and runs paired tests against HAFSSL and FL-AECNN (FedAvg).
Paste this whole file into ONE cell -> Save Version -> Save & Run All.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import json, time, glob, shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, balanced_accuracy_score,
                             confusion_matrix, classification_report)
from scipy import stats

# =============================== CONFIG ===============================
DATA_PATH = "data/ISCX-VPN2016/ISCX_5class_each_normalized_cuttedfloefeature.csv"
DATASET   = "ISCX"
LABELS    = ['chat', 'email', 'file', 'streaming', 'voip']
OUT_DIR   = "/kaggle/working/FED_ISCX_baselines"
T7_SEARCH = "/kaggle/input"          # where the Table VII output (T7_gpu0 / T7_gpu1) is searched for

METHODS = ["FLUIDS-fed", "FEAT", "CNN-FedAvg"]
SEEDS   = [0, 1, 2]

# Fixed protocol (identical to Table VII)
ALPHA, NUM_CLIENTS, ROUNDS, LOCAL_EPOCHS, BATCH, LR = 0.5, 10, 50, 5, 256, 1e-3
LABELED_PER_CLASS, SPLIT_SEED, COLLAPSE_RECALL = 1000, 2026, 0.10

# FEAT settings (see adaptations above)
FEAT_D, FEAT_LAMBDA, FEAT_ETA = 3, 0.6, 1.0
# FLUIDS server stage (settings of the FLUIDS code in [10])
FLUIDS_HEAD_EPOCHS, FLUIDS_HEAD_BATCH = 100, 128

QUICK_TEST = False     # True = 2 rounds, 1 epoch, seed 0
if QUICK_TEST:
    ROUNDS, LOCAL_EPOCHS, SEEDS, FLUIDS_HEAD_EPOCHS = 2, 1, [0], 2
    OUT_DIR += "_quicktest"
SAFE_LABELS = [l.replace(" ", "_") for l in LABELS]


# =============================== DATA (identical to all R2 scripts) ===============================
def load_data():
    df = pd.read_csv(DATA_PATH)
    feat_cols = [c for c in df.columns if c != 'label']
    n_before = len(df)
    df = df.drop_duplicates(subset=feat_cols).reset_index(drop=True)
    print(f"Rows: {n_before} -> {len(df)} after removing {n_before - len(df)} exact duplicates")
    X = df[feat_cols].values.astype('float32').reshape(len(df), len(feat_cols), 1)
    return X, df['label'].values.astype(int), len(feat_cols)


def get_split(y):
    idx = np.arange(len(y))
    trval, test = train_test_split(idx, test_size=0.10, stratify=y, random_state=SPLIT_SEED)
    train, val = train_test_split(trval, test_size=0.10, stratify=y[trval], random_state=SPLIT_SEED)
    return train, val, test


def sample_labeled(y_train, per_class, seed):
    rng = np.random.default_rng(seed)
    chosen = []
    for c in np.unique(y_train):
        pool = np.where(y_train == c)[0]
        chosen.extend(rng.choice(pool, min(per_class, len(pool)), replace=False))
    return np.array(chosen)


def partition_clients(y_train, kind, alpha, seed, n_clients, min_size=10):
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


def find_t7_dirs():
    found = []
    for name in ("T7_gpu0", "T7_gpu1", "FED_ISCX_table7"):
        found += [d for d in glob.glob(os.path.join(T7_SEARCH, "**", name), recursive=True) if os.path.isdir(d)]
    return sorted(set(found))


def check_against_table7(parts, labeled_idx, seed, t7_dirs):
    """Proves FEAT uses exactly the Table VII clients and labeled flows for this seed."""
    for d in t7_dirs:
        f = os.path.join(d, "splits", f"{DATASET}_partition_a{ALPHA}_K{NUM_CLIENTS}_L{LABELED_PER_CLASS}_s{seed}.npz")
        if os.path.exists(f):
            old = np.load(f)
            ok = all(np.array_equal(old[f"client_{k}"], p) for k, p in enumerate(parts)) and \
                 np.array_equal(old["labeled"], np.sort(labeled_idx))
            if not ok:
                raise RuntimeError(f"Seed {seed}: partition differs from Table VII ({f}). Check DATA_PATH.")
            return f
    return None


# =============================== FEAT CNN (paper, Sec. VI-A) ===============================
def build_feat_cnn(n_features, n_classes):
    i = layers.Input(shape=(n_features, 1))
    x = i
    for ch in (32, 64, 128):
        x = layers.Conv1D(ch, 3, padding='same', activation='relu')(x)
        x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    return Model(i, layers.Dense(n_classes, activation='softmax')(x), name="FEAT_CNN")


# =============================== FEAT client selection (Algorithm 1) ===============================
def feat_select(A, B, eligible, rng):
    """Thompson sampling over eligible clients -> g candidates -> d random. Returns sorted list S."""
    pool = list(eligible)
    g = max(FEAT_D, int(round(FEAT_LAMBDA * NUM_CLIENTS)))
    cand = []
    for _ in range(min(g, len(pool))):
        theta = {c: rng.beta(A[c] + 1.0, B[c] + 1.0) for c in pool}
        p = max(theta, key=theta.get)
        cand.append(p)
        pool.remove(p)
    return sorted(rng.choice(cand, size=min(FEAT_D, len(cand)), replace=False).tolist())


def feat_duel(A, B, S, deltas):
    """R_i = -||dw_i - mean_S dw||^2 ; every pair duels; winner A += eta, loser B += eta."""
    flat = {i: np.concatenate([d.ravel() for d in deltas[i]]) for i in S}
    mean = np.mean([flat[i] for i in S], axis=0)
    R = {i: -float(np.sum((flat[i] - mean) ** 2)) for i in S}
    for s in S:
        for s2 in S:
            if s != s2 and R[s] > R[s2]:
                A[s] += FEAT_ETA
                B[s2] += FEAT_ETA
    return R


# =============================== evaluation / saving ===============================
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
    p = results_csv()
    return os.path.exists(p) and run_id in set(pd.read_csv(p)['run_id'])


def reset_optimizer(opt):
    for v in opt.variables:
        if "learning_rate" not in v.path:
            v.assign(tf.zeros_like(v))



# =============================== FLUIDS-fed ([7], architecture from [10]'s FLUIDS code) ===============================
def build_fluids(n_features, n_classes, latent_dim=39):
    ie = layers.Input(shape=(n_features,))
    x = layers.Dense(150, activation="relu")(ie)
    x = layers.Dense(100, activation="relu")(x)
    x = layers.Dense(50, activation="relu")(x)
    enc = Model(ie, layers.Dense(latent_dim, activation="relu")(x), name="fluids_encoder")
    idd = layers.Input(shape=(latent_dim,))
    d = layers.Dense(50, activation="relu")(idd)
    d = layers.Dense(100, activation="relu")(d)
    d = layers.Dense(150, activation="relu")(d)
    dec = Model(idd, layers.Dense(n_features, activation="relu")(d), name="fluids_decoder")
    ic = layers.Input(shape=(latent_dim,))
    c = layers.Dense(150, activation="relu")(ic)
    c = layers.Dense(100, activation="relu")(c)
    c = layers.Dense(50, activation="relu")(c)
    head = Model(ic, layers.Dense(n_classes, activation="softmax")(c), name="fluids_mlp")
    inp = layers.Input(shape=(n_features,))
    return enc, dec, head, Model(inp, dec(enc(inp))), Model(inp, head(enc(inp)))


def run_fluids(run_id, seed, data, t7_dirs):
    X_tr, y_tr, x_val, y_val, x_te, y_te, test_idx, n_features, n_classes = data
    flat = lambda a: a.reshape(len(a), -1)                    # FLUIDS takes the flat 66-feature vector
    X_tr, x_val, x_te = flat(X_tr), flat(x_val), flat(x_te)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    enc, dec, head, ae, clf = build_fluids(n_features, n_classes)
    ae_w0 = enc.get_weights() + dec.get_weights()
    n_enc = len(enc.get_weights())
    opt = tf.keras.optimizers.Adam(LR)
    opt.build(ae.trainable_variables)

    @tf.function(reduce_retracing=True)
    def ae_step(x):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(x - ae(x, training=True)))
        opt.apply_gradients(zip(tape.gradient(loss, ae.trainable_variables), ae.trainable_variables))
        return loss

    parts = partition_clients(y_tr, "dirichlet", ALPHA, seed, NUM_CLIENTS)
    labeled_idx = sample_labeled(y_tr, LABELED_PER_CLASS, seed)
    checked = check_against_table7(parts, labeled_idx, seed, t7_dirs)
    n_k = [len(p) for p in parts]
    print(f"  partition {'= Table VII (' + os.path.basename(checked) + ')' if checked else 'not checked (no T7 found)'}"
          f" | flows per client {n_k}")
    p_k = np.array(n_k, dtype=np.float64) / np.sum(n_k)
    ae_params = enc.count_params() + dec.count_params()
    bytes_per_round = NUM_CLIENTS * 2 * 4 * ae_params           # only autoencoder weights are exchanged
    global_w, hist, t_start = ae_w0, [], time.time()

    def set_ae(w):
        enc.set_weights(w[:n_enc]); dec.set_weights(w[n_enc:])

    for rnd in range(1, ROUNDS + 1):                            # federated autoencoder stage
        t0 = time.time()
        client_ws, losses = [], []
        for k, idx in enumerate(parts):
            tf.keras.utils.set_random_seed(seed * 100000 + rnd * 100 + k)
            rng = np.random.default_rng(seed * 100000 + rnd * 100 + k)
            set_ae(global_w)
            reset_optimizer(opt)
            x = X_tr[idx]
            tot = tf.constant(0.0); n = 0
            for _ in range(LOCAL_EPOCHS):
                order = rng.permutation(len(x))
                for i in range(0, len(order), BATCH):
                    tot += ae_step(tf.constant(x[order[i:i + BATCH]])); n += 1
            client_ws.append(enc.get_weights() + dec.get_weights())
            losses.append(float(tot) / max(n, 1))
        global_w = [sum(pk * w[i] for pk, w in zip(p_k, client_ws)).astype(np.float32) for i in range(len(global_w))]
        set_ae(global_w)
        val_mse = float(np.mean(np.square(x_val - ae.predict(x_val, batch_size=2048, verbose=0))))
        sec = time.time() - t0
        hist.append({"round": rnd, "val_recon_mse": round(val_mse, 6),
                     "ae_loss": round(float(np.average(losses, weights=n_k)), 6),
                     "comm_MB_cum": round(rnd * bytes_per_round / 1e6, 3), "seconds": round(sec, 1)})
        print(f"  round {rnd}/{ROUNDS}: val reconstruction MSE {val_mse:.5f} | {sec:.0f}s")

    # server stage: MLP head on the frozen federated encoder, trained on the pooled labeled flows
    set_ae(global_w)
    enc.trainable = False
    tf.keras.utils.set_random_seed(seed * 100000 + 99999)
    y_lab = tf.keras.utils.to_categorical(y_tr[labeled_idx], n_classes).astype("float32")
    clf.compile(optimizer=tf.keras.optimizers.Adam(LR), loss="categorical_crossentropy", metrics=["accuracy"])
    h_fit = clf.fit(X_tr[labeled_idx], y_lab, epochs=FLUIDS_HEAD_EPOCHS, batch_size=FLUIDS_HEAD_BATCH,
                    shuffle=True, verbose=0,
                    validation_data=(x_val, tf.keras.utils.to_categorical(y_val, n_classes)))
    for e, (a, va) in enumerate(zip(h_fit.history["accuracy"], h_fit.history["val_accuracy"]), 1):
        if e % 20 == 0 or e == FLUIDS_HEAD_EPOCHS:
            print(f"  server head epoch {e}/{FLUIDS_HEAD_EPOCHS}: train acc {a:.4f} | val acc {va:.4f}")
    vm, _, _ = evaluate(clf.predict(x_val, batch_size=2048, verbose=0), y_val, n_classes)
    prob = clf.predict(x_te, batch_size=2048, verbose=0)
    m, pred, rec = evaluate(prob, y_te, n_classes)
    row = dict(run_id=run_id, dataset=DATASET, setting="federated", method="FLUIDS-fed", variant="FLUIDS-Dense",
               aggregator="FedAvg (AE) + server head", partition="dirichlet", alpha=ALPHA, clients=NUM_CLIENTS,
               labels_per_class=LABELED_PER_CLASS, labels_total=len(labeled_idx), seed=seed, rounds=ROUNDS,
               local_epochs=LOCAL_EPOCHS, batch=BATCH, lr=LR, server_head_epochs=FLUIDS_HEAD_EPOCHS,
               **{k: round(float(v), 6) for k, v in m.items()},
               min_class_recall=round(float(rec.min()), 6), collapse=bool(rec.min() < COLLAPSE_RECALL),
               **{f"recall_{l}": round(float(r), 6) for l, r in zip(SAFE_LABELS, rec)},
               val_acc_final=round(float(vm["acc"]), 6), val_acc_last10=np.nan, val_f1_last10=np.nan,
               rounds_to_90=-1,                                   # classifier exists only after the rounds
               params_sent=ae_params, params_inference=enc.count_params() + head.count_params(),
               MB_per_round=round(bytes_per_round / 1e6, 3), MB_total=round(ROUNDS * bytes_per_round / 1e6, 1),
               MB_to_90=-1, labeled_MB_to_server=round(len(labeled_idx) * n_features * 4 / 1e6, 3),
               partition_checked=bool(checked), train_seconds=round(time.time() - t_start, 1),
               tf_version=tf.__version__, timestamp=time.strftime("%Y-%m-%d %H:%M"))
    save_outputs(run_id, row, y_te, pred, prob, test_idx, pd.DataFrame(hist), n_classes)
    print(f"Done {run_id}: test acc {m['acc']:.4f} | macro-F1 {m['macro_f1']:.4f} | lowest class recall {rec.min():.3f}")


def save_outputs(run_id, row, y_te, pred, prob, test_idx, h, n_classes):
    for sub in ("results", "preds", "reports"):
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)
    np.savez_compressed(os.path.join(OUT_DIR, "preds", f"{run_id}.npz"),
                        y_true=y_te, y_pred=pred, y_prob=prob, test_idx=test_idx)
    pd.DataFrame(classification_report(y_te, pred, labels=list(range(n_classes)), target_names=LABELS, digits=4,
                                       output_dict=True, zero_division=0)).T.to_csv(
        os.path.join(OUT_DIR, "reports", f"{run_id}_per_class.csv"))
    pd.DataFrame(confusion_matrix(y_te, pred, labels=list(range(n_classes))), index=LABELS, columns=LABELS).to_csv(
        os.path.join(OUT_DIR, "reports", f"{run_id}_confusion.csv"))
    h.to_csv(os.path.join(OUT_DIR, "reports", f"{run_id}_rounds.csv"), index=False)
    p = results_csv()                     # rewrite the whole file: methods have different columns
    df = pd.concat([pd.read_csv(p), pd.DataFrame([row])], ignore_index=True) if os.path.exists(p) else pd.DataFrame([row])
    df.to_csv(p, index=False)

# =============================== ONE RUN ===============================
def run_one(run_id, method, seed, data, t7_dirs):
    X_tr, y_tr, x_val, y_val, x_te, y_te, test_idx, n_features, n_classes = data
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = build_feat_cnn(n_features, n_classes)
    opt = tf.keras.optimizers.Adam(LR)
    opt.build(model.trainable_variables)
    cce = tf.keras.losses.CategoricalCrossentropy()

    @tf.function(reduce_retracing=True)
    def step(x, y):
        with tf.GradientTape() as tape:
            loss = cce(y, model(x, training=True))
        opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
        return loss

    parts = partition_clients(y_tr, "dirichlet", ALPHA, seed, NUM_CLIENTS)
    labeled_idx = sample_labeled(y_tr, LABELED_PER_CLASS, seed)
    checked = check_against_table7(parts, labeled_idx, seed, t7_dirs)
    labeled = np.zeros(len(y_tr), dtype=bool)
    labeled[labeled_idx] = True
    y_oh = tf.keras.utils.to_categorical(y_tr, n_classes).astype('float32')
    clients = [(X_tr[idx[labeled[idx]]], y_oh[idx[labeled[idx]]]) for idx in parts]
    n_lab = [len(c[0]) for c in clients]
    eligible = [k for k in range(NUM_CLIENTS) if n_lab[k] > 0]
    print(f"  partition {'= Table VII (' + os.path.basename(checked) + ')' if checked else 'not checked (no T7 found)'}"
          f" | labeled per client {n_lab}")

    n_params = model.count_params()
    bytes_per_round = NUM_CLIENTS * 2 * 4 * n_params + (NUM_CLIENTS * 4 if method == "FEAT" else 0)  # + rewards
    sel_rng = np.random.default_rng(seed * 1000 + 7)
    A, B = np.zeros(NUM_CLIENTS), np.zeros(NUM_CLIENTS)
    global_w = model.get_weights()
    hist, t_start = [], time.time()
    for rnd in range(1, ROUNDS + 1):
        t0 = time.time()
        client_ws, deltas, losses = {}, {}, {}
        for k in range(NUM_CLIENTS):                  # FEAT: every client trains and uploads (Alg. 2, lines 5-9)
            if n_lab[k] == 0:
                continue
            tf.keras.utils.set_random_seed(seed * 100000 + rnd * 100 + k)
            rng = np.random.default_rng(seed * 100000 + rnd * 100 + k)
            model.set_weights(global_w)
            reset_optimizer(opt)
            x, y = clients[k]
            tot = tf.constant(0.0); n = 0
            for _ in range(LOCAL_EPOCHS):
                order = rng.permutation(len(x))
                for i in range(0, len(order), BATCH):
                    tot += step(tf.constant(x[order[i:i + BATCH]]), tf.constant(y[order[i:i + BATCH]]))
                    n += 1
            w = model.get_weights()
            client_ws[k] = w
            deltas[k] = [a - b for a, b in zip(w, global_w)]
            losses[k] = float(tot) / max(n, 1)
        if method == "FEAT":
            S = feat_select(A, B, eligible, sel_rng)
            R = feat_duel(A, B, S, deltas)
        else:
            S, R = eligible, {}
        p = np.array([n_lab[k] for k in S], dtype=np.float64)
        p /= p.sum()
        global_w = [sum(pk * client_ws[k][i] for pk, k in zip(p, S)).astype(np.float32) for i in range(len(global_w))]
        model.set_weights(global_w)
        vm, _, vrec = evaluate(model.predict(x_val, batch_size=2048, verbose=0), y_val, n_classes)
        sec = time.time() - t0
        hist.append({"round": rnd, "val_acc": round(vm["acc"], 6), "val_macro_f1": round(vm["macro_f1"], 6),
                     "val_min_recall": round(float(vrec.min()), 6),
                     **{f"val_recall_{l}": round(float(r), 6) for l, r in zip(SAFE_LABELS, vrec)},
                     "selected": " ".join(map(str, S)),
                     "clf_loss": round(float(np.average([losses[k] for k in losses],
                                                        weights=[n_lab[k] for k in losses])), 6),
                     "comm_MB_cum": round(rnd * bytes_per_round / 1e6, 3), "seconds": round(sec, 1)})
        print(f"  round {rnd}/{ROUNDS}: val acc {vm['acc']:.4f} | val macro-F1 {vm['macro_f1']:.4f} | "
              f"min recall {vrec.min():.3f} | selected {S} | {sec:.0f}s")

    prob = model.predict(x_te, batch_size=2048, verbose=0)
    m, pred, rec = evaluate(prob, y_te, n_classes)
    h = pd.DataFrame(hist)
    r90 = h.loc[h.val_acc >= 0.90, "round"]
    sel_counts = np.bincount(np.concatenate([np.array(s.split(), dtype=int) for s in h.selected]),
                             minlength=NUM_CLIENTS)
    row = dict(run_id=run_id, dataset=DATASET, setting="federated", method=method, variant="FEAT-CNN",
               aggregator=("FEAT-selection" if method == "FEAT" else "FedAvg"), partition="dirichlet", alpha=ALPHA,
               clients=NUM_CLIENTS, labels_per_class=LABELED_PER_CLASS, labels_total=int(labeled.sum()), seed=seed,
               rounds=ROUNDS, local_epochs=LOCAL_EPOCHS, batch=BATCH, lr=LR,
               feat_d=FEAT_D if method == "FEAT" else NUM_CLIENTS, feat_lambda=FEAT_LAMBDA, feat_eta=FEAT_ETA,
               **{k: round(float(v), 6) for k, v in m.items()},
               min_class_recall=round(float(rec.min()), 6), collapse=bool(rec.min() < COLLAPSE_RECALL),
               **{f"recall_{l}": round(float(r), 6) for l, r in zip(SAFE_LABELS, rec)},
               val_acc_last10=round(float(h.tail(10).val_acc.mean()), 6),
               val_f1_last10=round(float(h.tail(10).val_macro_f1.mean()), 6),
               rounds_to_90=int(r90.iloc[0]) if len(r90) else -1,
               params_sent=n_params, MB_per_round=round(bytes_per_round / 1e6, 3),
               MB_total=round(ROUNDS * bytes_per_round / 1e6, 1),
               MB_to_90=round(int(r90.iloc[0]) * bytes_per_round / 1e6, 1) if len(r90) else -1,
               times_selected=" ".join(map(str, sel_counts)), beta_A=" ".join(f"{a:g}" for a in A),
               beta_B=" ".join(f"{b:g}" for b in B), partition_checked=bool(checked),
               train_seconds=round(time.time() - t_start, 1), tf_version=tf.__version__,
               timestamp=time.strftime("%Y-%m-%d %H:%M"))
    save_outputs(run_id, row, y_te, pred, prob, test_idx, h, n_classes)
    print(f"Done {run_id}: test acc {m['acc']:.4f} | macro-F1 {m['macro_f1']:.4f} | lowest class recall {rec.min():.3f}"
          f" | times selected per client {sel_counts.tolist()}")


# =============================== SUMMARY + paired tests vs HAFSSL / FL-AECNN (FedAvg) ===============================
def mcnemar_exact(y, a, b):
    n01, n10 = int(np.sum((a == y) & (b != y))), int(np.sum((a != y) & (b == y)))
    return n01, n10, (1.0 if n01 + n10 == 0 else float(stats.binomtest(n01, n01 + n10, 0.5).pvalue))


def holm(pv):
    p = np.asarray(pv, float); order = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def summarize(t7_dirs):
    res = pd.read_csv(results_csv())
    print("\nNew federated baselines (test set, mean ± std over seeds):")
    print(res.groupby("method").agg(n=("seed", "nunique"), acc=("acc", "mean"), acc_sd=("acc", "std"),
                                    f1=("macro_f1", "mean"), f1_sd=("macro_f1", "std"),
                                    worst_recall=("min_class_recall", "min"),
                                    last10=("val_acc_last10", "mean")).round(4).to_string())
    print("(FLUIDS-fed: last10 = NaN, its classifier is trained by the server only after the 50 rounds)")
    ref_rows, ref_preds = [], {}
    for d in t7_dirs:
        if os.path.exists(results_csv(d)):
            ref_rows.append(pd.read_csv(results_csv(d)))
        for f in glob.glob(os.path.join(d, "preds", "*.npz")):
            ref_preds[os.path.basename(f)[:-4]] = f
    if not ref_rows:
        print("\nTable VII output not found in", T7_SEARCH, "-> no paired tests. Add it as Input and rerun.")
        return
    ref = pd.concat(ref_rows).drop_duplicates("run_id")
    ref = ref[(ref.aggregator == "FedAvg") & ref.method.isin(["HAFSSL", "FL-AECNN"])]
    pairs = []
    for meth in METHODS:
        for refm in ["HAFSSL", "FL-AECNN"]:
            for s in SEEDS:
                a = res[(res.method == meth) & (res.seed == s)]
                b = ref[(ref.method == refm) & (ref.seed == s)]
                if a.empty or b.empty or b.run_id.iloc[0] not in ref_preds:
                    continue
                da = np.load(os.path.join(OUT_DIR, "preds", a.run_id.iloc[0] + ".npz"))
                db = np.load(ref_preds[b.run_id.iloc[0]])
                assert np.array_equal(da["test_idx"], db["test_idx"]), "Different test sets!"
                n01, n10, p = mcnemar_exact(da["y_true"], db["y_pred"], da["y_pred"])   # reference vs baseline
                pairs.append(dict(baseline=meth, reference=f"{refm} (FedAvg)", seed=s, acc_ref=b.acc.iloc[0],
                                  acc_baseline=a.acc.iloc[0], diff_pp=100 * (b.acc.iloc[0] - a.acc.iloc[0]),
                                  only_ref_correct=n01, only_baseline_correct=n10, mcnemar_p=p))
    if not pairs:
        print("No pairs found."); return
    pr = pd.DataFrame(pairs)
    pr["mcnemar_p_holm"] = holm(pr.mcnemar_p)
    pr.to_csv(os.path.join(OUT_DIR, "baselines_mcnemar_per_seed.csv"), index=False)
    tt = []
    for (bm, rm), g in pr.groupby(["baseline", "reference"]):
        d = g.diff_pp.values
        tp = float(stats.ttest_rel(g.acc_ref, g.acc_baseline).pvalue) if len(d) >= 2 and np.std(d) > 0 else np.nan
        tt.append(dict(baseline=bm, reference=rm, n_seeds=len(d), mean_diff_pp=d.mean(), ttest_p=tp,
                       mcnemar_all_seeds_sig=bool((g.mcnemar_p_holm < 0.05).all() and (np.sign(d) == np.sign(d.mean())).all())))
    tt = pd.DataFrame(tt)
    ok = tt.ttest_p.notna()
    tt["ttest_p_holm"] = np.nan
    tt.loc[ok, "ttest_p_holm"] = holm(tt.loc[ok, "ttest_p"])
    tt["reference_better"] = (tt.ttest_p_holm < 0.05) & tt.mcnemar_all_seeds_sig & (tt.mean_diff_pp > 0)
    tt.to_csv(os.path.join(OUT_DIR, "baselines_paired_tests.csv"), index=False)
    pd.set_option("display.width", 220)
    print("\nPer seed (reference minus baseline, pp):\n" + pr.round(4).to_string(index=False))
    print("\nPaired over seeds:\n" + tt.round(4).to_string(index=False))


# =============================== MAIN ===============================
def main():
    tf.config.experimental.enable_op_determinism()
    os.makedirs(OUT_DIR, exist_ok=True)
    X, y, n_features = load_data()
    tr, va, te = get_split(y)
    data = (X[tr], y[tr], X[va], y[va], X[te], y[te], te, n_features, len(LABELS))
    t7_dirs = find_t7_dirs()
    print(f"NEW FEDERATED BASELINES | {DATASET} | train {len(tr)} | val {len(va)} | test {len(te)} | "
          f"GPU {tf.config.list_physical_devices('GPU') != []}")
    print(f"methods {METHODS} | seeds {SEEDS} | alpha {ALPHA} | K {NUM_CLIENTS} | rounds {ROUNDS} | "
          f"local epochs {LOCAL_EPOCHS} | labels/class {LABELED_PER_CLASS} | FEAT d={FEAT_D} lambda={FEAT_LAMBDA} "
          f"eta={FEAT_ETA} | FLUIDS head {FLUIDS_HEAD_EPOCHS} ep | QUICK_TEST {QUICK_TEST}")
    print("Table VII folders found:", t7_dirs if t7_dirs else "none (partition check and paired tests skipped)")
    with open(os.path.join(OUT_DIR, "config_baselines.json"), "w") as f:
        json.dump(dict(methods=METHODS, seeds=SEEDS, alpha=ALPHA, clients=NUM_CLIENTS, rounds=ROUNDS,
                       local_epochs=LOCAL_EPOCHS, batch=BATCH, lr=LR, labeled_per_class=LABELED_PER_CLASS,
                       feat_d=FEAT_D, feat_lambda=FEAT_LAMBDA, feat_eta=FEAT_ETA, fluids_head_epochs=FLUIDS_HEAD_EPOCHS,
                       fluids_head_batch=FLUIDS_HEAD_BATCH, split_seed=SPLIT_SEED,
                       quick_test=QUICK_TEST, data_path=DATA_PATH), f, indent=2)
    for seed in SEEDS:
        for method in METHODS:
            run_id = f"{DATASET}_{method}_a{ALPHA}_K{NUM_CLIENTS}_L{LABELED_PER_CLASS}_s{seed}"
            if already_done(run_id):
                print("Skip (already done):", run_id); continue
            print(f"\n===== {run_id} =====")
            if method == "FLUIDS-fed":
                run_fluids(run_id, seed, data, t7_dirs)
            else:
                run_one(run_id, method, seed, data, t7_dirs)
    summarize(t7_dirs)


main()
