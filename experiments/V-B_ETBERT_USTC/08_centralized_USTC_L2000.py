# -*- coding: utf-8 -*-
"""
SynAttnNet, AECNN and AECNN-ft on USTC-TFC2016 - CENTRALIZED, protocol of Table II (R2 revision).
Purpose: the like-for-like centralized comparison for ET-BERT (Reviewer 1, Comment 2).

MODELS AND TRAINING: copied from the Table II scripts (01_centralized_synattnnet_R2.py and the AECNN script):
  * autoencoder 50 epochs on ALL training flows (labels not used), then classifier 50 epochs on the labeled
    flows; batch 128; Adam, learning rate 1e-3; the final model is evaluated once on the test set.
  * SynAttnNet : hybrid attention block (MHA -> SE), encoder fine-tuned in the classifier stage
  * AECNN      : same network without the attention block, encoder FROZEN in the classifier stage, as in [10]
  * AECNN-ft   : AECNN with the encoder fine-tuned = SynAttnNet without the attention block
    (isolates the attention block, as in the centralized ablation of Section V-D)

DATA, SPLIT AND LABELS (standalone: no federated run is needed):
  * USTC .npy payload strings -> first 128 two-byte tokens, each divided by 65535, zero padded to 128 values;
    rows with ambiguous labels and exact duplicates removed (93,968 flows)
  * the frozen split of the paper, split seed 2026 (76,113 / 8,458 / 9,397); the sizes are checked
  * labeled flows: 100 per class (2,000), drawn from the training split with seed s, exactly as in Table II
    (sample_labeled). The ET-BERT script draws them with the same function, so for a given seed all models use
    the SAME 2,000 labeled flows. They are saved to splits/USTC_labeled_s{seed}.npz.

KAGGLE SETUP (GPU T4, one notebook):
  Add as Input: only the USTC dataset (6 .npy files).
  Paste the whole file into ONE cell. QUICK_TEST = True first (1 epoch, seed 0), then False -> Save & Run All.
  Expected time on a T4: about 30 minutes per seed for the three models (about 1.5 hours for seeds 0-2).
  If the session stops, run again: finished runs are skipped.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import glob, json, time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             balanced_accuracy_score, confusion_matrix, classification_report)
from scipy import stats

# =============================== CONFIG ===============================
BASE_PATH  = "data/USTC-TFC2016"   # the 6 .npy files
REF_SEARCH = "/kaggle/input"          # where the USTC dataset is searched for if BASE_PATH is wrong
OUT_DIR    = "/kaggle/working/CEN_USTC_L2000"
DATASET    = "USTC"
LABELS     = ['BitTorrent', 'Facetime', 'FTP', 'Gmail', 'MySQL', 'Outlook', 'Skype', 'SMB', 'Weibo',
              'WorldOfWarcraft', 'Cridex', 'Geodo', 'Htbot', 'Miuref', 'Neris', 'Nsis-ay', 'Shifu',
              'Tinba', 'Virut', 'Zeus']
METHODS    = ["SynAttnNet", "AECNN", "AECNN-ft"]
SEEDS      = [0, 1, 2]
LABELS_PER_CLASS = 100                 # 2,000 labeled flows in total, the label budget of the USTC runs
EXPECTED_SIZES   = (93968, 76113, 8458, 9397)   # flows after de-duplication, train, validation, test

# Table II settings - do not change
LATENT_DIM, NUM_HEADS, KEY_DIM, SE_RATIO = 39, 4, 32, 16
EPOCHS_AE, EPOCHS_CLF, BATCH, LR = 50, 50, 128, 1e-3
# Data protocol of the USTC runs - do not change
SPLIT_SEED, MAX_LEN = 2026, 128
DETERMINISTIC = True
QUICK_TEST    = False                  # True = 1 epoch, seed 0

if QUICK_TEST:
    EPOCHS_AE = EPOCHS_CLF = 1
    SEEDS = [0]
    OUT_DIR = OUT_DIR + "_quicktest"


# =============================== DATA (identical to the USTC federated and ET-BERT scripts) ===============================
def autodetect_base_path():
    global BASE_PATH
    if os.path.exists(os.path.join(BASE_PATH, "x_payload_train.npy")):
        return BASE_PATH
    hits = glob.glob(os.path.join(REF_SEARCH, "**", "x_payload_train.npy"), recursive=True)
    if not hits:
        raise FileNotFoundError("x_payload_train.npy not found: attach the USTC dataset or set BASE_PATH.")
    BASE_PATH = os.path.dirname(hits[0]) + os.sep
    print("BASE_PATH auto-detected:", BASE_PATH)
    return BASE_PATH


def hex_to_tokens(hex_str, max_len=MAX_LEN):
    ids = [int(t, 16) for t in str(hex_str).strip().split() if len(t) == 4][:max_len]
    ids += [0] * max(0, max_len - len(ids))
    return np.array(ids, dtype=np.float32) / 65535.0


def load_data():
    parts_x, parts_y = [], []
    for name in ("train", "valid", "test"):
        parts_x.append(np.load(os.path.join(BASE_PATH, f"x_payload_{name}.npy"), allow_pickle=True))
        parts_y.append(np.load(os.path.join(BASE_PATH, f"y_{name}.npy"), allow_pickle=True))
    X = np.stack([hex_to_tokens(s) for s in np.concatenate(parts_x)])
    y = np.concatenate(parts_y).astype(int)
    n0 = len(y)
    key = pd.util.hash_pandas_object(pd.DataFrame(X), index=False).values
    labels_per_key = pd.Series(y).groupby(key).nunique()
    ambiguous = np.isin(key, labels_per_key[labels_per_key > 1].index.values)
    keep = (~ambiguous) & (~pd.Series(key).duplicated().values)
    print(f"Rows: {n0} -> {int(keep.sum())} after removing {int(ambiguous.sum())} rows with ambiguous labels "
          f"and {int((~ambiguous).sum() - keep.sum())} exact duplicates")
    X, y = X[keep], y[keep]
    return X.reshape(len(X), MAX_LEN, 1), y


def get_split(y):
    idx = np.arange(len(y))
    trval, test = train_test_split(idx, test_size=0.10, stratify=y, random_state=SPLIT_SEED)
    train, val = train_test_split(trval, test_size=0.10, stratify=y[trval], random_state=SPLIT_SEED)
    return train, val, test


def check_sizes(y, tr, va, te):
    got = (len(y), len(tr), len(va), len(te))
    if got != EXPECTED_SIZES:
        raise RuntimeError(f"Data sizes {got} differ from the paper {EXPECTED_SIZES}: check BASE_PATH and the files.")


def sample_labeled(y_train, per_class, seed):
    """Table II rule: 'per_class' labeled flows per class from the TRAINING split; same seed = same flows.
    The ET-BERT script uses this same function, so all models get the same labeled flows."""
    rng = np.random.default_rng(seed)
    chosen = []
    for c in np.unique(y_train):
        pool = np.where(y_train == c)[0]
        chosen.extend(rng.choice(pool, min(per_class, len(pool)), replace=False))
    return np.array(chosen)


# =============================== MODELS (copied from the Table II scripts) ===============================
def se_block(x, ratio=SE_RATIO):
    ch = x.shape[-1]
    s = layers.GlobalAveragePooling1D()(x)
    s = layers.Dense(ch // ratio, activation='relu')(s)
    s = layers.Dense(ch, activation='sigmoid')(s)
    s = layers.Reshape((1, ch))(s)
    return layers.Multiply()([x, s])


def hybrid_attention_block(x):
    a = layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=KEY_DIM, dropout=0.1)(x, x)   # MHA first
    a = se_block(a)                                                                       # then SE
    x = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, a]))
    f = layers.Dense(KEY_DIM * 4, activation='relu')(x)
    f = layers.Dense(x.shape[-1])(layers.Dropout(0.1)(f))
    return layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, f]))


def build_models(n_features, n_classes, attention):
    """attention=True: SynAttnNet (Table II); attention=False: AECNN network of [10] (Table II)."""
    ie = layers.Input(shape=(n_features, 1))
    x = layers.Conv1D(64, 3, padding='same', activation='relu', name='conv_1')(ie)
    x = layers.Conv1D(64, 3, padding='same', activation='relu', name='conv_2')(x)
    x = layers.MaxPooling1D(name='maxpool_1')(x)              # 128 -> 64 positions on USTC
    if attention:
        x = hybrid_attention_block(x)
    x = layers.Conv1D(32, 3, padding='same', activation='relu', name='conv_3')(x)
    x = layers.Conv1D(32, 3, padding='same', activation='relu', name='conv_4')(x)
    steps, ch = x.shape[1], x.shape[2]
    encoder = Model(ie, layers.Dense(LATENT_DIM, activation='relu')(layers.Flatten()(x)), name='encoder')

    idd = layers.Input(shape=(LATENT_DIM,))
    d = layers.Reshape((steps, ch))(layers.Dense(steps * ch)(idd))
    d = layers.Conv1D(32, 3, padding='same', activation='relu')(d)
    d = layers.Conv1D(32, 3, padding='same', activation='relu')(d)
    d = layers.UpSampling1D()(d)
    d = layers.Conv1D(64, 3, padding='same', activation='relu')(d)
    decoder = Model(idd, layers.Conv1D(1, 3, padding='same', activation='relu')(d), name='decoder')

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
    ae = Model(inp, decoder(encoder(inp)), name='autoencoder')
    clf = Model(inp, head(encoder(inp)), name='classifier')       # same encoder object as the autoencoder
    return encoder, decoder, head, ae, clf


# =============================== EVALUATION (identical to Table II) ===============================
def gpu_name():
    gpus = tf.config.list_physical_devices('GPU')
    return tf.config.experimental.get_device_details(gpus[0]).get('device_name', 'GPU') if gpus else "CPU only"


def evaluate(clf, x_test, y_test, n_classes):
    prob = clf.predict(x_test, batch_size=1024, verbose=0)
    pred = prob.argmax(axis=1)
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, labels=list(range(n_classes)), average='macro',
                                                  zero_division=0)
    m = {'acc': accuracy_score(y_test, pred), 'macro_precision': p, 'macro_recall': r, 'macro_f1': f1,
         'balanced_acc': balanced_accuracy_score(y_test, pred)}
    rec = confusion_matrix(y_test, pred, labels=list(range(n_classes))).diagonal() / np.maximum(
        np.bincount(y_test, minlength=n_classes), 1)
    return m, pred, prob, rec


def results_csv():
    return os.path.join(OUT_DIR, "results", "all_runs.csv")


def already_done(run_id):
    return os.path.exists(results_csv()) and run_id in set(pd.read_csv(results_csv())['run_id'])


def save_run(run_id, row, y_test, pred, prob, test_idx, lab_idx, histories):
    for sub in ("results", "preds", "reports"):
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)
    np.savez_compressed(os.path.join(OUT_DIR, "preds", f"{run_id}.npz"),
                        y_true=y_test, y_pred=pred, y_prob=prob, test_idx=test_idx, labeled_idx=lab_idx)
    rep = classification_report(y_test, pred, labels=list(range(len(LABELS))), target_names=LABELS, digits=4,
                                output_dict=True, zero_division=0)
    pd.DataFrame(rep).transpose().to_csv(os.path.join(OUT_DIR, "reports", f"{run_id}_per_class.csv"))
    pd.DataFrame(confusion_matrix(y_test, pred, labels=list(range(len(LABELS)))), index=LABELS,
                 columns=LABELS).to_csv(os.path.join(OUT_DIR, "reports", f"{run_id}_confusion.csv"))
    with open(os.path.join(OUT_DIR, "reports", f"{run_id}_history.json"), "w") as fh:
        json.dump(histories, fh)
    pd.DataFrame([row]).to_csv(results_csv(), mode='a', header=not os.path.exists(results_csv()), index=False)


def holm(pv):
    p = np.asarray(pv, float); order = np.argsort(p); adj = np.empty_like(p); run_max = 0.0
    for rank, i in enumerate(order):
        run_max = max(run_max, (len(p) - rank) * p[i]); adj[i] = min(1.0, run_max)
    return adj


def summarize():
    res = pd.read_csv(results_csv())
    print("\nUSTC-TFC2016, centralized, 2,000 labeled flows (test set, mean ± std over seeds):")
    print(res.groupby('method')[['acc', 'macro_f1', 'min_class_recall']].agg(['mean', 'std']).round(4).to_string())
    rows = []
    for other in ("AECNN", "AECNN-ft"):
        for s in sorted(res.seed.unique()):
            fa = os.path.join(OUT_DIR, "preds", f"{DATASET}_SynAttnNet_L2000_s{s}.npz")
            fb = os.path.join(OUT_DIR, "preds", f"{DATASET}_{other}_L2000_s{s}.npz")
            if not (os.path.exists(fa) and os.path.exists(fb)):
                continue
            a, b = np.load(fa), np.load(fb)
            y = a["y_true"]
            n01 = int(np.sum((a["y_pred"] == y) & (b["y_pred"] != y)))
            n10 = int(np.sum((a["y_pred"] != y) & (b["y_pred"] == y)))
            p = 1.0 if n01 + n10 == 0 else float(stats.binomtest(n01, n01 + n10, 0.5).pvalue)
            rows.append(dict(comparison=f"SynAttnNet vs {other}", seed=s, acc_synattnnet=float(np.mean(a["y_pred"] == y)),
                             acc_other=float(np.mean(b["y_pred"] == y)), mcnemar_p=p))
    if rows:
        pr = pd.DataFrame(rows)
        pr["mcnemar_p_holm"] = holm(pr.mcnemar_p)
        pr["diff_pp"] = 100 * (pr.acc_synattnnet - pr.acc_other)
        pr.to_csv(os.path.join(OUT_DIR, "paired_mcnemar_per_seed.csv"), index=False)
        tt = []
        for comp, g in pr.groupby("comparison"):
            tp = float(stats.ttest_rel(g.acc_synattnnet, g.acc_other).pvalue) if len(g) >= 2 and g.diff_pp.std() > 0 else np.nan
            tt.append(dict(comparison=comp, n_seeds=len(g), mean_diff_pp=g.diff_pp.mean(),
                           synattnnet_higher_in=int((g.diff_pp > 0).sum()), ttest_p=tp,
                           mcnemar_sig_seeds=int((g.mcnemar_p_holm < 0.05).sum())))
        tt = pd.DataFrame(tt)
        tt.to_csv(os.path.join(OUT_DIR, "paired_tests.csv"), index=False)
        pd.set_option("display.width", 200)
        print("\nPer seed:\n" + pr.round(4).to_string(index=False))
        print("\nPaired over seeds:\n" + tt.round(4).to_string(index=False))


# =============================== MAIN ===============================
def main():
    if DETERMINISTIC:
        tf.config.experimental.enable_op_determinism()
    autodetect_base_path()
    X, y = load_data()
    tr, va, te = get_split(y)
    check_sizes(y, tr, va, te)
    os.makedirs(os.path.join(OUT_DIR, "splits"), exist_ok=True)
    np.savez(os.path.join(OUT_DIR, "splits", f"{DATASET}_split.npz"), train=tr, val=va, test=te, n_rows=len(y))
    x_tr, x_val, x_te = X[tr], X[va], X[te]
    y_tr, y_val, y_te = y[tr], y[va], y[te]
    n_classes = len(LABELS)
    assert len(np.unique(y)) == n_classes, f"{len(np.unique(y))} classes found, LABELS has {n_classes}"
    y_val_oh = tf.keras.utils.to_categorical(y_val, n_classes)
    print(f"Train {len(tr)} | Val {len(va)} | Test {len(te)} | sizes as in the paper | "
          f"{gpu_name()} | methods {METHODS} | seeds {SEEDS} | QUICK_TEST {QUICK_TEST}")
    with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
        json.dump(dict(methods=METHODS, seeds=SEEDS, latent=LATENT_DIM, heads=NUM_HEADS, key_dim=KEY_DIM,
                       se_ratio=SE_RATIO, epochs_ae=EPOCHS_AE, epochs_clf=EPOCHS_CLF, batch=BATCH, lr=LR,
                       split_seed=SPLIT_SEED, max_len=MAX_LEN, labels_per_class=LABELS_PER_CLASS,
                       labels="sample_labeled(y_train, 100, seed), as in Table II"), f, indent=2)

    for seed in SEEDS:
        sel = sample_labeled(y_tr, LABELS_PER_CLASS, seed)
        np.savez(os.path.join(OUT_DIR, "splits", f"{DATASET}_labeled_s{seed}.npz"),
                 train_positions=sel, global_indices=tr[sel], labels=y_tr[sel])
        per_class = np.bincount(y_tr[sel], minlength=n_classes)
        print(f"\nSeed {seed}: {len(sel)} labeled flows, per class {per_class.min()}-{per_class.max()}")
        assert len(sel) == 2000 and per_class.min() == 100 == per_class.max(), "expected 100 labeled flows per class"
        for method in METHODS:
            run_id = f"{DATASET}_{method}_L2000_s{seed}"
            if already_done(run_id):
                print("Skip (already done):", run_id)
                continue
            print(f"\n===== {run_id} =====")
            attention = method == "SynAttnNet"
            freeze = method == "AECNN"
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(seed)
            encoder, decoder, head, ae, clf = build_models(MAX_LEN, n_classes, attention)
            t0, hist = time.time(), {}

            # Stage 1: unsupervised autoencoder on ALL training flows (labels not used)
            ae.compile(optimizer=tf.keras.optimizers.Adam(LR), loss='mse')
            h = ae.fit(x_tr, x_tr, epochs=EPOCHS_AE, batch_size=BATCH, shuffle=True,
                       validation_data=(x_val, x_val), verbose=2)
            hist['ae'] = {k: [float(v) for v in vals] for k, vals in h.history.items()}

            # Stage 2: classifier on the 2,000 labeled flows (encoder frozen for AECNN, fine-tuned otherwise)
            encoder.trainable = not freeze
            clf.compile(optimizer=tf.keras.optimizers.Adam(LR), loss='categorical_crossentropy', metrics=['accuracy'])
            h = clf.fit(x_tr[sel], tf.keras.utils.to_categorical(y_tr[sel], n_classes),
                        epochs=EPOCHS_CLF, batch_size=BATCH, shuffle=True,
                        validation_data=(x_val, y_val_oh), verbose=2)
            hist['clf'] = {k: [float(v) for v in vals] for k, vals in h.history.items()}

            m, pred, prob, rec = evaluate(clf, x_te, y_te, n_classes)     # final model, no early stopping
            row = dict(run_id=run_id, dataset=DATASET, setting="centralized", method=method, seed=seed,
                       labels_total=len(sel), label_source="sample_labeled(y_train, 100, seed), shared with ET-BERT",
                       **{k: round(float(v), 6) for k, v in m.items()},
                       min_class_recall=round(float(rec.min()), 6), classes_below_50=int((rec < 0.5).sum()),
                       **{f"recall_{l}": round(float(r), 6) for l, r in zip(LABELS, rec)},
                       params_encoder=encoder.count_params(), params_decoder=decoder.count_params(),
                       params_head=head.count_params(), encoder_frozen=freeze, attention=attention,
                       epochs_ae=EPOCHS_AE, epochs_clf=EPOCHS_CLF, batch=BATCH, lr=LR,
                       n_train=len(tr), n_test=len(te), train_seconds=round(time.time() - t0, 1),
                       gpu=gpu_name(), tf_version=tf.__version__, timestamp=time.strftime("%Y-%m-%d %H:%M"))
            save_run(run_id, row, y_te, pred, prob, te, tr[sel], hist)
            print(f"Done {run_id}: acc {m['acc']:.4f} | macro-F1 {m['macro_f1']:.4f} | "
                  f"classes below 50% recall {int((rec < 0.5).sum())}")
    summarize()


if __name__ == "__main__":
    main()
