# -*- coding: utf-8 -*-
"""
AECNN baseline [10] - centralized semi-supervised traffic classification (Table II), R2 revision.

Same pipeline as 01_centralized_synattnnet_R2.py, so the results can be compared directly:
  - exact duplicate rows removed before the split
  - the same frozen stratified split (same data + same SPLIT_SEED = same test set)
  - the same seeds (0, 1, 2) and the same labeled subset for each seed and budget
  - test set never used during training (separate validation split)
  - macro-averaged metrics, predictions saved for McNemar's test, one row per run in all_runs.csv

AECNN model, as in the code of [10] (exp1/07-Proposed.py):
  encoder   : Conv1D(64) -> Conv1D(64) -> MaxPool -> Conv1D(32) -> Conv1D(32) -> Dense(39)   (NO attention)
  decoder   : same as SynAttnNet
  classifier: same head as SynAttnNet (no dropout)
  training  : autoencoder 50 epochs, then classifier 50 epochs, Adam, batch 128.
              In [10] the encoder is FROZEN while the classifier trains (FREEZE_ENCODER = True).
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")      # hides harmless TensorFlow log lines
import json, time, shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             balanced_accuracy_score, confusion_matrix, classification_report)

# =============================== CONFIG ===============================
# One setting per line. Do not put two settings on one line.

# 10 Popular Apps: use these 4 lines instead of the ISCX lines below
#DATA_PATH   = "data/10PopularApps/pcapdroid_10class_each_normalized_cuttedfloefeature.csv"
#OUT_DIR     = "/kaggle/working/HAFSSL_R2_Apps10_AECNN"
#DATASET     = "Apps10"
#LABELS      = ['TikTok', 'iQiyi Video', 'Jingdong Shopping', 'Snack Video', 'QQ Music','QQ', 'Taobao Shopping', 'NetEase Cloud Music', 'Arena of Valor', 'WeChat']

# ISCX-VPN2016 (active)
DATA_PATH = "data/ISCX-VPN2016/ISCX_5class_each_normalized_cuttedfloefeature.csv"
OUT_DIR   = "/kaggle/working/HAFSSL_R2_ISCX_AECNN"
DATASET   = "ISCX"
LABELS    = ['chat', 'email', 'file', 'streaming', 'voip']

# Optional but safest: use EXACTLY the split file of your SynAttnNet run.
# Add the SynAttnNet output as input (Add Input -> Your Work) and paste the path of its split file, e.g.
#   SPLIT_FROM = "/kaggle/input/.../HAFSSL_R2_Apps10/splits/Apps10_split.npz"
SPLIT_FROM  = ""

METHOD_NAME    = "AECNN"
FREEZE_ENCODER = True      # True = as in [10] (encoder frozen in the classifier stage); False = fine-tuned like SynAttnNet
LABELS_PER_CLASS_LIST = [1000, 2000, 3000, 4000]   # labeled flows PER CLASS, same as SynAttnNet
SEEDS   = [0, 1, 2]

LATENT_DIM = 39
EPOCHS_AE, EPOCHS_CLF, BATCH, LR = 50, 50, 128, 1e-3   # same as SynAttnNet and [10]
SPLIT_SEED    = 2026        # must be the same as in the SynAttnNet runs
DETERMINISTIC = True
QUICK_TEST    = False    # True = short test (1 epoch, 1 seed, 200 labels); False = real run

if QUICK_TEST:
    EPOCHS_AE = EPOCHS_CLF = 1
    SEEDS, LABELS_PER_CLASS_LIST = [0], [200]
    OUT_DIR = OUT_DIR + "_quicktest"


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


# =============================== DATA (identical to SynAttnNet) ===============================
def load_data():
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
    """Same frozen stratified split as SynAttnNet: test 10%, validation 9%, train 81%."""
    path = os.path.join(OUT_DIR, "splits", f"{DATASET}_split.npz")
    if SPLIT_FROM and not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        shutil.copy(SPLIT_FROM, path)
        print("Using the split file of the SynAttnNet run:", SPLIT_FROM)
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
    """Same labeled subset as SynAttnNet for the same seed and budget."""
    rng = np.random.default_rng(seed)
    chosen = []
    for c in np.unique(y_train):
        pool = np.where(y_train == c)[0]
        k = min(per_class, len(pool))
        if k < per_class:
            print(f"  Warning: class {c} has only {len(pool)} training flows; using all of them.")
        chosen.extend(rng.choice(pool, k, replace=False))
    return np.array(chosen)


# =============================== AECNN MODEL ([10]) ===============================
def build_models(n_features, n_classes):
    # Encoder: no attention block
    ie = layers.Input(shape=(n_features, 1))
    x = layers.Conv1D(64, 3, padding='same', activation='relu', name='conv_1')(ie)
    x = layers.Conv1D(64, 3, padding='same', activation='relu', name='conv_2')(x)
    x = layers.MaxPooling1D(name='maxpool_1')(x)
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

    # Classifier head (same as SynAttnNet)
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

    # Shared encoder: the pre-trained encoder is the one the classifier uses
    inp = layers.Input(shape=(n_features, 1))
    ae = Model(inp, decoder(encoder(inp)), name='autoencoder')
    clf = Model(inp, head(encoder(inp)), name='classifier')
    return encoder, decoder, head, ae, clf


# =============================== EVALUATION (identical to SynAttnNet) ===============================
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
    plat = platform_name()
    if DETERMINISTIC:
        tf.config.experimental.enable_op_determinism()
    X, y, n_features = load_data()
    train_idx, val_idx, test_idx = get_split(y)
    x_tr, x_val, x_te = X[train_idx], X[val_idx], X[test_idx]
    y_tr, y_val, y_te = y[train_idx], y[val_idx], y[test_idx]
    n_classes = len(LABELS)
    y_val_oh = tf.keras.utils.to_categorical(y_val, n_classes)
    method = METHOD_NAME if FREEZE_ENCODER else METHOD_NAME + "-ft"
    print(f"Train {len(train_idx)} | Val {len(val_idx)} | Test {len(test_idx)} | {plat} | {gpu_name()}")
    print(f"Method: {method} (encoder {'frozen' if FREEZE_ENCODER else 'fine-tuned'} in the classifier stage)")

    for lpc in LABELS_PER_CLASS_LIST:
        for seed in SEEDS:
            run_id = f"{DATASET}_{method}_L{lpc}_s{seed}"
            if already_done(run_id):
                print("Skip (already done):", run_id)
                continue
            print(f"\n===== {run_id} =====")
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(seed)
            lab = sample_labeled(y_tr, lpc, seed)
            encoder, decoder, head, ae, clf = build_models(n_features, n_classes)
            t0, hist = time.time(), {}

            # Stage 1: unsupervised autoencoder on ALL training flows (labels not used)
            ae.compile(optimizer=tf.keras.optimizers.Adam(LR), loss='mse')
            h = ae.fit(x_tr, x_tr, epochs=EPOCHS_AE, batch_size=BATCH, shuffle=True,
                       validation_data=(x_val, x_val), verbose=2)
            hist['ae'] = {k: [float(v) for v in vals] for k, vals in h.history.items()}

            # Stage 2: classifier on the labeled subset (encoder frozen as in [10], or fine-tuned)
            encoder.trainable = not FREEZE_ENCODER
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
                       params_head=head.count_params(), encoder_frozen=FREEZE_ENCODER,
                       epochs_ae=EPOCHS_AE, epochs_clf=EPOCHS_CLF, batch=BATCH, lr=LR,
                       n_train=len(train_idx), n_test=len(test_idx),
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
