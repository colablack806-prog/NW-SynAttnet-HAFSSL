# -*- coding: utf-8 -*-
"""
SynAttnNet - centralized semi-supervised traffic classification (Table II), R2 revision.

What changed compared with the old 01_centralized_comparison.py:
  1. BUG FIX: the old script saved the autoencoder and the classifier to .h5 and loaded them
     separately, so each got its OWN copy of the encoder. The autoencoder pre-training never
     reached the classifier. Here both models share ONE encoder in memory.
  2. Exact duplicate rows are removed BEFORE the split (about 25% of ISCX test rows had a copy
     in the training data).
  3. One frozen, stratified train / validation / test split, saved to disk and reused by every
     method and every seed.
  4. Seeds (0, 1, 2) control weight initialisation, batch order and the labeled subset.
  5. The test set is never used during training. A separate validation split is used instead.
  6. Metrics are macro-averaged. Predictions are saved for McNemar's test.
     One row per run is appended to results/all_runs.csv.
  7. The invented "total accuracy" (0.3 x reconstruction + 0.7 x accuracy) is removed.
  8. Architecture now identical to the federated HAFSSL script and paper Fig. 3: max-pooling
     BEFORE the hybrid attention block, and no dropout in the classifier head.
     Parameters are unchanged (autoencoder 168,588; encoder + classifier 380,368).

Works on Colab, Kaggle and PC: only DATA_PATH, OUT_DIR and MOUNT_DRIVE change.
Before long runs, do a smoke test: set QUICK_TEST = True (1 epoch, 1 seed, a few minutes).
"""
import os, json, time, platform
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             balanced_accuracy_score, confusion_matrix, classification_report)

# =============================== CONFIG ===============================
# One setting per line. Do not put two settings on one line.
#
# Colab (default below): the CSV is downloaded automatically from DATA_GDRIVE_ID if it is missing.
#   DATA_PATH   = "/content/ISCX_5class_each_normalized_cuttedfloefeature.csv"
#   OUT_DIR     = "/content/drive/My Drive/HAFSSL_R2"
#   MOUNT_DRIVE = True
#
# Kaggle:

DATA_PATH      = "data/10PopularApps/pcapdroid_10class_each_normalized_cuttedfloefeature.csv"
DATA_GDRIVE_ID = ""
OUT_DIR = "/kaggle/working/HAFSSL_R2_Apps10"
MOUNT_DRIVE    = False

DATASET = "Apps10"
LABELS  = ['TikTok', 'iQiyi Video', 'Jingdong Shopping', 'Snack Video', 'QQ Music',
           'QQ', 'Taobao Shopping', 'NetEase Cloud Music', 'Arena of Valor', 'WeChat']
#DATASET = "Apps10"                                   # use "Apps10" for 10 Popular Apps
#LABELS  = ['chat', 'email', 'file', 'streaming', 'voip']
LABELS_PER_CLASS_LIST = [1000, 2000, 3000, 4000]   # same meaning as the old 'labelnum': labeled flows PER CLASS
SEEDS   = [0, 1, 2]
NO_PRETRAIN_CHECK_BUDGETS = [1000]                 # also run "no autoencoder pre-training" at these budgets

LATENT_DIM, NUM_HEADS, KEY_DIM, SE_RATIO = 39, 4, 32, 16
EPOCHS_AE, EPOCHS_CLF, BATCH, LR = 50, 50, 128, 1e-3   # what the old script used: write these in the paper
SPLIT_SEED    = 2026        # fixes the train/val/test split; never change it after the first run
DETERMINISTIC = True        # set False only if TensorFlow raises a determinism error on your GPU
QUICK_TEST = False

if QUICK_TEST:
    EPOCHS_AE = EPOCHS_CLF = 1
    SEEDS, LABELS_PER_CLASS_LIST, NO_PRETRAIN_CHECK_BUDGETS = [0], [200], [200]
    OUT_DIR = OUT_DIR + "_quicktest"


# =============================== SETUP ===============================
def setup_platform():
    if MOUNT_DRIVE:
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            return "colab"
        except ImportError:
            pass
    return "kaggle" if os.path.exists("/kaggle") else "pc"


def gpu_name():
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        return "CPU only"
    return tf.config.experimental.get_device_details(gpus[0]).get('device_name', 'GPU')


# =============================== DATA ===============================
def load_data():
    if not os.path.exists(DATA_PATH) and DATA_GDRIVE_ID:
        import gdown                                     # preinstalled on Colab; on Kaggle: pip install gdown
        gdown.download(f"https://drive.google.com/uc?id={DATA_GDRIVE_ID}", DATA_PATH, quiet=False)
    df = pd.read_csv(DATA_PATH)
    feat_cols = [c for c in df.columns if c != 'label']
    n_before = len(df)
    df = df.drop_duplicates(subset=feat_cols).reset_index(drop=True)   # leakage fix
    print(f"Rows: {n_before} -> {len(df)} after removing {n_before - len(df)} exact duplicates")
    print("Rows per class after de-duplication:", df['label'].value_counts().sort_index().to_dict())
    X = df[feat_cols].values.astype('float32').reshape(len(df), len(feat_cols), 1)
    y = df['label'].values.astype(int)
    return X, y, len(feat_cols)


def get_split(y):
    """Frozen stratified split: test 10%, validation 9%, train 81%. Created once, then reused."""
    path = os.path.join(OUT_DIR, "splits", f"{DATASET}_split.npz")
    if os.path.exists(path):
        s = np.load(path)
        assert int(s['n_rows']) == len(y), "Split file does not match this data. Do not mix datasets."
        return s['train'], s['val'], s['test']
    idx = np.arange(len(y))
    trval, test = train_test_split(idx, test_size=0.10, stratify=y, random_state=SPLIT_SEED)
    train, val = train_test_split(trval, test_size=0.10, stratify=y[trval], random_state=SPLIT_SEED)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, train=train, val=val, test=test, n_rows=len(y))
    print("Saved the frozen split to", path)
    return train, val, test


def sample_labeled(y_train, per_class, seed):
    """Picks 'per_class' labeled flows per class from the TRAINING part only. Same seed = same subset."""
    rng = np.random.default_rng(seed)
    chosen = []
    for c in np.unique(y_train):
        pool = np.where(y_train == c)[0]
        k = min(per_class, len(pool))
        if k < per_class:
            print(f"  Warning: class {c} has only {len(pool)} training flows; using all of them.")
        chosen.extend(rng.choice(pool, k, replace=False))
    return np.array(chosen)


# =============================== MODEL ===============================
def se_block(x, ratio=SE_RATIO):
    """Squeeze-and-excitation, same layers and parameters as the old SEBlock1D, no custom class."""
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


def build_models(n_features, n_classes):
    # Encoder (same order as the federated script and paper Fig. 3):
    # Conv1D(64) -> Conv1D(64) -> MaxPool -> hybrid attention (MHA -> SE) -> Conv1D(32) -> Conv1D(32) -> Dense(latent)
    ie = layers.Input(shape=(n_features, 1))
    x = layers.Conv1D(64, 3, padding='same', activation='relu', name='conv_1')(ie)
    x = layers.Conv1D(64, 3, padding='same', activation='relu', name='conv_2')(x)
    x = layers.MaxPooling1D(name='maxpool_1')(x)          # 66 -> 33 positions
    x = hybrid_attention_block(x)                          # attention over 33 positions
    x = layers.Conv1D(32, 3, padding='same', activation='relu', name='conv_3')(x)
    x = layers.Conv1D(32, 3, padding='same', activation='relu', name='conv_4')(x)
    steps, ch = x.shape[1], x.shape[2]
    encoder = Model(ie, layers.Dense(LATENT_DIM, activation='relu')(layers.Flatten()(x)), name='encoder')

    # Decoder
    idd = layers.Input(shape=(LATENT_DIM,))
    d = layers.Reshape((steps, ch))(layers.Dense(steps * ch)(idd))
    d = layers.Conv1D(32, 3, padding='same', activation='relu')(d)
    d = layers.Conv1D(32, 3, padding='same', activation='relu')(d)
    d = layers.UpSampling1D()(d)
    d = layers.Conv1D(64, 3, padding='same', activation='relu')(d)
    decoder = Model(idd, layers.Conv1D(1, 3, padding='same', activation='relu')(d), name='decoder')

    # Classifier head
    ic = layers.Input(shape=(LATENT_DIM,))
    c = layers.Reshape((steps, ch))(layers.Dense(steps * ch)(ic))
    c = layers.Conv1D(64, 3, padding='same', activation='relu')(c)
    c = layers.Conv1D(64, 3, padding='same', activation='relu')(c)
    c = layers.MaxPooling1D(2)(c)
    c = layers.Conv1D(128, 3, padding='same', activation='relu')(c)
    c = layers.Conv1D(128, 3, padding='same', activation='relu')(c)
    c = layers.MaxPooling1D(2)(c)
    c = layers.Dense(128, activation='relu')(layers.Flatten()(c))     # no dropout, as in the federated script
    head = Model(ic, layers.Dense(n_classes, activation='softmax')(c), name='classifier_head')

    # Both models use the SAME encoder object, so pre-training reaches the classifier.
    inp = layers.Input(shape=(n_features, 1))
    ae = Model(inp, decoder(encoder(inp)), name='autoencoder')
    clf = Model(inp, head(encoder(inp)), name='classifier')
    return encoder, decoder, head, ae, clf


# =============================== EVALUATION ===============================
def evaluate(clf, x_test, y_test):
    prob = clf.predict(x_test, batch_size=1024, verbose=0)
    pred = prob.argmax(axis=1)
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average='macro', zero_division=0)
    m = {'acc': accuracy_score(y_test, pred), 'macro_precision': p, 'macro_recall': r, 'macro_f1': f1,
         'weighted_f1': precision_recall_fscore_support(y_test, pred, average='weighted', zero_division=0)[2],
         'balanced_acc': balanced_accuracy_score(y_test, pred)}
    assert abs(m['macro_recall'] - m['balanced_acc']) < 1e-9, "Metric bug: macro recall must equal balanced accuracy"
    return m, pred, prob


def already_done(run_id):
    path = os.path.join(OUT_DIR, "results", "all_runs.csv")
    return os.path.exists(path) and run_id in set(pd.read_csv(path)['run_id'])


def save_run(run_id, row, y_test, pred, prob, test_idx, histories):
    for sub in ("results", "preds", "reports"):
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)
    np.savez_compressed(os.path.join(OUT_DIR, "preds", f"{run_id}.npz"),
                        y_true=y_test, y_pred=pred, y_prob=prob, test_idx=test_idx)
    rep = classification_report(y_test, pred, target_names=LABELS, digits=4, output_dict=True, zero_division=0)
    pd.DataFrame(rep).transpose().to_csv(os.path.join(OUT_DIR, "reports", f"{run_id}_per_class.csv"))
    pd.DataFrame(confusion_matrix(y_test, pred), index=LABELS, columns=LABELS).to_csv(
        os.path.join(OUT_DIR, "reports", f"{run_id}_confusion.csv"))
    with open(os.path.join(OUT_DIR, "reports", f"{run_id}_history.json"), "w") as fh:
        json.dump(histories, fh)
    path = os.path.join(OUT_DIR, "results", "all_runs.csv")
    pd.DataFrame([row]).to_csv(path, mode='a', header=not os.path.exists(path), index=False)


# =============================== MAIN ===============================
def main():
    plat = setup_platform()
    if DETERMINISTIC:
        tf.config.experimental.enable_op_determinism()
    X, y, n_features = load_data()
    train_idx, val_idx, test_idx = get_split(y)
    x_tr, x_val, x_te = X[train_idx], X[val_idx], X[test_idx]
    y_tr, y_val, y_te = y[train_idx], y[val_idx], y[test_idx]
    n_classes = len(LABELS)
    y_val_oh = tf.keras.utils.to_categorical(y_val, n_classes)
    print(f"Train {len(train_idx)} | Val {len(val_idx)} | Test {len(test_idx)} | {plat} | {gpu_name()}")

    jobs = [(lpc, True) for lpc in LABELS_PER_CLASS_LIST] + \
           [(lpc, False) for lpc in LABELS_PER_CLASS_LIST if lpc in NO_PRETRAIN_CHECK_BUDGETS]
    for lpc, pretrain in jobs:
        for seed in SEEDS:
            method = "SynAttnNet" if pretrain else "SynAttnNet-noAE"
            run_id = f"{DATASET}_{method}_L{lpc}_s{seed}"
            if already_done(run_id):
                print("Skip (already done):", run_id)
                continue
            print(f"\n===== {run_id} =====")
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(seed)     # python, numpy and TensorFlow seeds
            lab = sample_labeled(y_tr, lpc, seed)
            encoder, decoder, head, ae, clf = build_models(n_features, n_classes)
            t0, hist = time.time(), {}

            if pretrain:   # Stage 1: unsupervised autoencoder on ALL training flows (labels not used)
                ae.compile(optimizer=tf.keras.optimizers.Adam(LR), loss='mse')
                h = ae.fit(x_tr, x_tr, epochs=EPOCHS_AE, batch_size=BATCH, shuffle=True,
                           validation_data=(x_val, x_val), verbose=2)
                hist['ae'] = {k: [float(v) for v in vals] for k, vals in h.history.items()}

            # Stage 2: supervised fine-tuning of encoder + head on the labeled subset only
            clf.compile(optimizer=tf.keras.optimizers.Adam(LR), loss='categorical_crossentropy',
                        metrics=['accuracy'])
            h = clf.fit(x_tr[lab], tf.keras.utils.to_categorical(y_tr[lab], n_classes),
                        epochs=EPOCHS_CLF, batch_size=BATCH, shuffle=True,
                        validation_data=(x_val, y_val_oh), verbose=2)
            hist['clf'] = {k: [float(v) for v in vals] for k, vals in h.history.items()}

            m, pred, prob = evaluate(clf, x_te, y_te)    # final model, no early stopping
            row = dict(run_id=run_id, dataset=DATASET, setting="centralized", method=method, seed=seed,
                       labels_per_class=lpc, labels_total=len(lab), **{k: round(float(v), 6) for k, v in m.items()},
                       params_encoder=encoder.count_params(), params_decoder=decoder.count_params(),
                       params_head=head.count_params(), epochs_ae=EPOCHS_AE if pretrain else 0,
                       epochs_clf=EPOCHS_CLF, batch=BATCH, lr=LR, n_train=len(train_idx), n_test=len(test_idx),
                       train_seconds=round(time.time() - t0, 1), platform=plat, gpu=gpu_name(),
                       tf_version=tf.__version__, timestamp=time.strftime("%Y-%m-%d %H:%M"))
            save_run(run_id, row, y_te, pred, prob, test_idx, hist)
            print(f"Done {run_id}: acc {m['acc']:.4f} | macro-F1 {m['macro_f1']:.4f}")

    res = pd.read_csv(os.path.join(OUT_DIR, "results", "all_runs.csv"))
    res = res[res.dataset == DATASET]
    print("\nMean ± std over seeds:")
    print(res.groupby(['method', 'labels_per_class'])[['acc', 'macro_f1']].agg(['mean', 'std']).round(4))


if __name__ == "__main__":
    main()
