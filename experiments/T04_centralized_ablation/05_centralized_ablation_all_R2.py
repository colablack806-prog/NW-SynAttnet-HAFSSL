# -*- coding: utf-8 -*-
"""
SynAttnNet - centralized ATTENTION ABLATION, ALL FIVE VARIANTS IN ONE SCRIPT (R2 revision).

Why this script: in the previous ablation, MHA->SE came from the Table II notebook while the other
four variants came from the ablation notebook, which reseeds every epoch. Here all five variants
use exactly the same code, training loop and random seeds, so every difference in Table III comes
from the attention block alone.

Same as Table II: duplicate removal, frozen split (SPLIT_SEED 2026), labeled flows of each seed,
50 autoencoder epochs + 50 fine-tuning epochs, batch 128, Adam 1e-3, final model (no early stopping),
macro metrics on the frozen test set, saved predictions.

    variant    method / run_id name        block after the first MaxPool of the encoder
    "MHA->SE"  SynAttnNet                  MHA, then SE on its output (your model)
    "SE->MHA"  SynAttnNet-SEthenMHA        SE, then MHA (reversed order, same parameters)
    "MHA"      SynAttnNet-MHAonly          MHA only
    "SE"       SynAttnNet-SEonly           SE only
    "none"     SynAttnNet-noAttn           no block (AECNN backbone, encoder fine-tuned)
All variants except "none" keep the residual connection, LayerNorm and feed-forward layer of the block.

Label budgets: 100 and 200 labeled flows per class test the label-scarce regime the paper claims;
1,000 and 3,000 match Table III.

Speed-up (no effect on results): the autoencoder stage never sees labels, so for one variant and one
seed it is identical at every label budget. It is trained ONCE per variant and seed, saved (all
weights, including the dropout random-generator state), and reused for every budget.
VERIFY_DETERMINISM re-trains one autoencoder and checks that the weights are identical.

Runs per dataset: 5 variants x 3 seeds autoencoders (15) + 5 x 4 budgets x 3 seeds fine-tunings (60).
Approximate time on a Kaggle T4: ISCX about 3 h, Apps10 about 5-6 h (one session each).

KAGGLE, step by step
  1. Add the dataset CSV as input (if the path below is wrong, the script searches /kaggle/input).
  2. QUICK_TEST = True -> Save Version -> Save & Run All (a few minutes). Check the log.
  3. QUICK_TEST = False -> Save & Run All.
     If the log ends with "STOPPED SAFELY": Add Input -> Your Work -> this notebook's output,
     put the path of its output folder in RESUME_FROM, and Save & Run All again.
  4. Change DATASET to "Apps10" and repeat in a second notebook.
  5. Optional: ONLY_SUMMARIZE = True and SUMMARY_DIRS = [both output folders] -> one combined table.
Outputs in OUT_DIR/summary/: table3_rows.csv, table3_summary.csv, table3_paired_tests.csv,
table3_mcnemar_per_seed.csv and table3_latex.tex (ready-to-paste LaTeX rows).
"""
import os, json, time, shutil, glob
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")      # hides harmless TensorFlow log lines
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             balanced_accuracy_score, confusion_matrix, classification_report)
from scipy import stats

# =============================== CONFIG ===============================
DATASET = "ISCX"                                         # "ISCX" or "Apps10" (one dataset per notebook)
VARIANTS = ["MHA->SE", "SE->MHA", "MHA", "SE", "none"]   # all five, same code and seeds
LABELS_PER_CLASS_LIST = [100, 200, 1000, 3000]           # labeled flows PER CLASS
SEEDS = [0, 1, 2]

DATA = {
    "ISCX": dict(
        path="data/ISCX-VPN2016/ISCX_5class_each_normalized_cuttedfloefeature.csv",
        labels=['chat', 'email', 'file', 'streaming', 'voip']),
    "Apps10": dict(
        path="data/10PopularApps/pcapdroid_10class_each_normalized_cuttedfloefeature.csv",
        labels=['TikTok', 'iQiyi Video', 'Jingdong Shopping', 'Snack Video', 'QQ Music',
                'QQ', 'Taobao Shopping', 'NetEase Cloud Music', 'Arena of Valor', 'WeChat']),
}
OUT_DIR = f"/kaggle/working/R2_{DATASET}_ablation_all"

RESUME_FROM        = ""     # Kaggle: previous version's output folder, to continue after a stop
TIME_BUDGET_HOURS  = 11.0   # stops cleanly before Kaggle's 12-hour limit; progress is kept
ONLY_SUMMARIZE     = False  # True = no training; combine the folders below and write the tables
SUMMARY_DIRS       = []     # e.g. ["/kaggle/input/abl-iscx/R2_ISCX_ablation_all", ...]
VERIFY_DETERMINISM = True   # re-train one autoencoder and check it is bit-identical (~1 extra AE)

LATENT_DIM, NUM_HEADS, KEY_DIM, SE_RATIO = 39, 4, 32, 16
EPOCHS_AE, EPOCHS_CLF, BATCH, LR = 50, 50, 128, 1e-3   # identical to Table II; do not change
SPLIT_SEED    = 2026        # identical to Table II; do not change
DETERMINISTIC = True
QUICK_TEST    = False       # True = 1 epoch, seed 0, budgets 100 and 200 (a few minutes)

LABELS = DATA[DATASET]["labels"]
if QUICK_TEST:
    EPOCHS_AE = EPOCHS_CLF = 1
    SEEDS, LABELS_PER_CLASS_LIST = [0], [100, 200]
    OUT_DIR = OUT_DIR + "_quicktest"

METHOD = {"MHA->SE": "SynAttnNet", "SE->MHA": "SynAttnNet-SEthenMHA", "MHA": "SynAttnNet-MHAonly",
          "SE": "SynAttnNet-SEonly", "none": "SynAttnNet-noAttn"}
VARIANT_OF = {m: v for v, m in METHOD.items()}
ORDER = ["MHA->SE", "SE->MHA", "MHA", "SE", "none"]
REFERENCES = ["MHA->SE", "none"]     # paired tests: your model vs each variant, and no-attention vs each variant
LATEX_NAME = {"MHA->SE": r"MHA$\rightarrow$SE (Ours)", "SE->MHA": r"SE$\rightarrow$MHA",
              "MHA": "MHA-only", "SE": "SE-only", "none": "No attention"}
assert all(v in METHOD for v in VARIANTS), f"Unknown variant in {VARIANTS}"

SESSION_START = time.time()
_STOP_AFTER_EPOCHS = int(os.environ.get("STOP_AFTER_EPOCHS", "0"))   # testing only: simulate a stop
_epochs_this_session = 0


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


def restore_previous_output():
    """Kaggle: copy the previous version's output into the empty working folder."""
    if not RESUME_FROM:
        return
    if not os.path.isdir(RESUME_FROM):
        raise FileNotFoundError(f"RESUME_FROM folder not found: {RESUME_FROM}")
    if os.path.isdir(OUT_DIR) and os.listdir(OUT_DIR):
        print("OUT_DIR is not empty, so RESUME_FROM is not copied:", OUT_DIR)
        return
    shutil.copytree(RESUME_FROM, OUT_DIR, dirs_exist_ok=True)
    print("Copied previous progress from", RESUME_FROM)


# =============================== DATA (same as Table II) ===============================
def data_path():
    path = DATA[DATASET]["path"]
    if os.path.exists(path):
        return path
    name = os.path.basename(path)
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True)) + sorted(glob.glob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found. Add it as input or fix DATA['{DATASET}']['path'].")
    print("Data file found at", hits[0])
    return hits[0]


def load_data():
    df = pd.read_csv(data_path())
    feat_cols = [c for c in df.columns if c != 'label']
    n_before = len(df)
    df = df.drop_duplicates(subset=feat_cols).reset_index(drop=True)   # leakage fix
    print(f"Rows: {n_before} -> {len(df)} after removing {n_before - len(df)} exact duplicates")
    print("Rows per class after de-duplication:", df['label'].value_counts().sort_index().to_dict())
    X = df[feat_cols].values.astype('float32').reshape(len(df), len(feat_cols), 1)
    y = df['label'].values.astype(int)
    print(f"Features: {len(feat_cols)} | value range: {X.min():.4f} to {X.max():.4f}")
    assert len(np.unique(y)) == len(LABELS), f"Data has {len(np.unique(y))} classes, LABELS has {len(LABELS)}"
    return X, y, len(feat_cols)


def get_split(y):
    """Frozen stratified split (test 10%, validation 9%, train 81%), identical to Table II."""
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
    """Squeeze-and-excitation, same layers and parameters as in Table II."""
    ch = x.shape[-1]
    s = layers.GlobalAveragePooling1D()(x)
    s = layers.Dense(ch // ratio, activation='relu')(s)
    s = layers.Dense(ch, activation='sigmoid')(s)
    s = layers.Reshape((1, ch))(s)
    return layers.Multiply()([x, s])


def attention_block(x, variant, key_dim=KEY_DIM):
    """The ablated block. "MHA->SE" is line-for-line the hybrid block of the Table II script."""
    if variant == "none":
        return x
    if variant == "MHA->SE":
        a = layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=key_dim, dropout=0.1)(x, x)
        a = se_block(a)
    elif variant == "SE->MHA":
        s = se_block(x)
        a = layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=key_dim, dropout=0.1)(s, s)
    elif variant == "MHA":
        a = layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=key_dim, dropout=0.1)(x, x)
    elif variant == "SE":
        a = se_block(x)
    else:
        raise ValueError(variant)
    x = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, a]))
    f = layers.Dense(key_dim * 4, activation='relu')(x)
    f = layers.Dense(x.shape[-1])(layers.Dropout(0.1)(f))
    return layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, f]))


def build_models(n_features, n_classes, variant):
    # Encoder: Conv1D(64) -> Conv1D(64) -> MaxPool -> [ablated block] -> Conv1D(32) -> Conv1D(32) -> Dense(latent)
    ie = layers.Input(shape=(n_features, 1))
    x = layers.Conv1D(64, 3, padding='same', activation='relu', name='conv_1')(ie)
    x = layers.Conv1D(64, 3, padding='same', activation='relu', name='conv_2')(x)
    x = layers.MaxPooling1D(name='maxpool_1')(x)
    x = attention_block(x, variant)
    x = layers.Conv1D(32, 3, padding='same', activation='relu', name='conv_3')(x)
    x = layers.Conv1D(32, 3, padding='same', activation='relu', name='conv_4')(x)
    steps, ch = x.shape[1], x.shape[2]
    encoder = Model(ie, layers.Dense(LATENT_DIM, activation='relu')(layers.Flatten()(x)), name='encoder')

    # Decoder (unchanged from Table II)
    idd = layers.Input(shape=(LATENT_DIM,))
    d = layers.Reshape((steps, ch))(layers.Dense(steps * ch)(idd))
    d = layers.Conv1D(32, 3, padding='same', activation='relu')(d)
    d = layers.Conv1D(32, 3, padding='same', activation='relu')(d)
    d = layers.UpSampling1D()(d)
    d = layers.Conv1D(64, 3, padding='same', activation='relu')(d)
    decoder = Model(idd, layers.Conv1D(1, 3, padding='same', activation='relu')(d), name='decoder')

    # Classifier head (unchanged: no attention, no dropout)
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

    # Both models use the SAME encoder object, so pre-training reaches the classifier.
    inp = layers.Input(shape=(n_features, 1))
    ae = Model(inp, decoder(encoder(inp)), name='autoencoder')
    clf = Model(inp, head(encoder(inp)), name='classifier')
    return encoder, decoder, head, ae, clf


def fresh_models(variant, seed, n_features, n_classes):
    """Same seed -> same initial weights for every budget."""
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    return build_models(n_features, n_classes, variant)


# =============================== CHECKPOINTS ===============================
class StopForNow(Exception):
    """Raised to stop cleanly (time budget reached). Progress is already saved."""


def ckpt_path(ck_id):
    return os.path.join(OUT_DIR, "checkpoints", f"{ck_id}.npz")


def _opt_vars(opt):
    v = opt.variables
    return v() if callable(v) else v


def save_ckpt(ck_id, meta, parts, opt):
    os.makedirs(os.path.dirname(ckpt_path(ck_id)), exist_ok=True)
    arrays = {}
    for name, m in parts.items():
        for i, v in enumerate(m.variables):
            arrays[f"{name}__{i}"] = np.asarray(v.numpy())
    for i, v in enumerate(_opt_vars(opt)):
        arrays[f"opt__{i}"] = np.asarray(v.numpy())
    arrays["meta"] = np.array(json.dumps(meta))
    tmp = ckpt_path(ck_id) + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, ckpt_path(ck_id))


def load_ckpt(ck_id, parts, opt, model):
    d = np.load(ckpt_path(ck_id), allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    for name, m in parts.items():
        for i, v in enumerate(m.variables):
            v.assign(d[f"{name}__{i}"])
    opt.build(model.trainable_variables)
    ov = _opt_vars(opt)
    n_saved = len([k for k in d.files if k.startswith("opt__")])
    if n_saved != len(ov):
        raise RuntimeError("Checkpoint does not match this model/optimizer. Delete it to restart this run.")
    for i, v in enumerate(ov):
        v.assign(d[f"opt__{i}"])
    return meta


def check_time(last_epoch_seconds):
    if _STOP_AFTER_EPOCHS and _epochs_this_session >= _STOP_AFTER_EPOCHS:
        raise StopForNow()
    if time.time() - SESSION_START + 1.5 * last_epoch_seconds > TIME_BUDGET_HOURS * 3600:
        raise StopForNow()


def train_stage(stage, model, opt, x, y, val, epochs, start_epoch, seed, ck_id, meta, parts):
    global _epochs_this_session
    last = meta.get("last_epoch_seconds", 0.0)
    for ep in range(start_epoch, epochs):
        check_time(last)
        tf.keras.utils.set_random_seed(seed * 100000 + (0 if stage == "ae" else 50000) + ep)
        t = time.time()
        print(f"  [{stage}] epoch {ep + 1}/{epochs}")
        h = model.fit(x, y, epochs=1, batch_size=BATCH, shuffle=True, validation_data=val, verbose=2)
        last = time.time() - t
        for k, v in h.history.items():
            meta["hist"].setdefault(stage, {}).setdefault(k, []).append(float(v[0]))
        meta.update(stage=stage, epoch_done=ep + 1, last_epoch_seconds=last,
                    train_seconds=meta["train_seconds"] + last)
        save_ckpt(ck_id, meta, parts, opt)
        _epochs_this_session += 1


# =============================== STAGE 1: AUTOENCODER (once per variant and seed) ===============================
def pretrained_path(variant, seed, tag=""):
    return os.path.join(OUT_DIR, "pretrained", f"{DATASET}_{METHOD[variant]}{tag}_s{seed}.npz")


def pretrain(variant, seed, data, tag=""):
    """Unsupervised autoencoder on ALL training flows (labels not used). Saved and reused for every budget."""
    path = pretrained_path(variant, seed, tag)
    if os.path.exists(path):
        return
    n_features, n_classes = data["n_features"], data["n_classes"]
    encoder, decoder, head, ae, clf = fresh_models(variant, seed, n_features, n_classes)
    opt = tf.keras.optimizers.Adam(LR)
    ae.compile(optimizer=opt, loss='mse')
    parts = {"enc": encoder, "dec": decoder}
    ck_id = f"{DATASET}_{METHOD[variant]}_AE{tag}_s{seed}"
    meta = {"stage": "ae", "epoch_done": 0, "hist": {}, "train_seconds": 0.0}
    print(f"\n===== autoencoder {METHOD[variant]}{tag}, seed {seed} =====")
    if os.path.exists(ckpt_path(ck_id)):
        meta = load_ckpt(ck_id, parts, opt, ae)
        print(f"  Resuming from checkpoint: {meta['epoch_done']} epochs done")
    train_stage("ae", ae, opt, data["x_tr"], data["x_tr"], (data["x_val"], data["x_val"]),
                EPOCHS_AE, meta["epoch_done"], seed, ck_id, meta, parts)
    arrays = {}
    for name, m in parts.items():                      # includes the dropout random-generator state
        for i, v in enumerate(m.variables):
            arrays[f"{name}__{i}"] = np.asarray(v.numpy())
    arrays["meta"] = np.array(json.dumps({"hist": meta["hist"], "train_seconds": meta["train_seconds"],
                                          "final_loss": meta["hist"]["ae"]["loss"][-1],
                                          "final_val_loss": meta["hist"]["ae"]["val_loss"][-1]}))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    os.remove(ckpt_path(ck_id))
    print(f"  Saved pre-trained autoencoder: {path} | final loss {meta['hist']['ae']['loss'][-1]:.3e}")


def load_pretrained(variant, seed, parts):
    d = np.load(pretrained_path(variant, seed), allow_pickle=False)
    for name, m in parts.items():
        n_saved = len([k for k in d.files if k.startswith(f"{name}__")])
        if n_saved != len(m.variables):
            raise RuntimeError(f"Pre-trained file does not match the {name} of variant {variant}.")
        for i, v in enumerate(m.variables):
            v.assign(d[f"{name}__{i}"])
    return json.loads(str(d["meta"]))


def verify_determinism(variant, seed, data):
    """Re-trains one autoencoder from scratch and compares all its weights with the saved one."""
    out = os.path.join(OUT_DIR, "results", "determinism_check.json")
    if not VERIFY_DETERMINISM or os.path.exists(out):
        return
    pretrain(variant, seed, data, tag="-verify")
    a = np.load(pretrained_path(variant, seed), allow_pickle=False)
    b = np.load(pretrained_path(variant, seed, "-verify"), allow_pickle=False)
    keys = [k for k in a.files if k != "meta"]
    max_diff = max(float(np.max(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)))) for k in keys)
    info = dict(variant=variant, seed=seed, n_arrays=len(keys), max_abs_weight_difference=max_diff,
                identical=bool(max_diff == 0.0), gpu=gpu_name(), tf_version=tf.__version__)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(info, fh, indent=2)
    print("\n" + "=" * 70)
    if info["identical"]:
        print("DETERMINISM CHECK PASSED: re-training gives bit-identical autoencoder weights.")
    else:
        print(f"DETERMINISM CHECK: re-training differs (max weight difference {max_diff:.3e}).")
        print("  The comparison between variants is still fair (all use the same procedure), but")
        print("  report that runs are not bit-reproducible on this GPU.")
    print("=" * 70)


# =============================== STAGE 2: FINE-TUNING (per budget) ===============================
def evaluate(clf, x_test, y_test):
    prob = clf.predict(x_test, batch_size=1024, verbose=0)
    pred = prob.argmax(axis=1)
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average='macro', zero_division=0)
    m = {'acc': accuracy_score(y_test, pred), 'macro_precision': p, 'macro_recall': r, 'macro_f1': f1,
         'weighted_f1': precision_recall_fscore_support(y_test, pred, average='weighted', zero_division=0)[2],
         'balanced_acc': balanced_accuracy_score(y_test, pred)}
    assert abs(m['macro_recall'] - m['balanced_acc']) < 1e-9, "Metric bug: macro recall must equal balanced accuracy"
    return m, pred, prob


def results_csv(folder=None):
    return os.path.join(folder or OUT_DIR, "results", "all_runs.csv")


def already_done(run_id):
    return os.path.exists(results_csv()) and run_id in set(pd.read_csv(results_csv())['run_id'])


def save_run(run_id, row, y_test, pred, prob, test_idx, histories):
    for sub in ("results", "preds", "reports"):
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)
    np.savez_compressed(os.path.join(OUT_DIR, "preds", f"{run_id}.npz"),
                        y_true=y_test, y_pred=pred, y_prob=prob, test_idx=test_idx)
    rep = classification_report(y_test, pred, labels=list(range(len(LABELS))), target_names=LABELS,
                                digits=4, output_dict=True, zero_division=0)
    pd.DataFrame(rep).transpose().to_csv(os.path.join(OUT_DIR, "reports", f"{run_id}_per_class.csv"))
    pd.DataFrame(confusion_matrix(y_test, pred, labels=list(range(len(LABELS)))), index=LABELS,
                 columns=LABELS).to_csv(os.path.join(OUT_DIR, "reports", f"{run_id}_confusion.csv"))
    with open(os.path.join(OUT_DIR, "reports", f"{run_id}_history.json"), "w") as fh:
        json.dump(histories, fh)
    pd.DataFrame([row]).to_csv(results_csv(), mode='a', header=not os.path.exists(results_csv()), index=False)


def finetune(run_id, variant, lpc, seed, data, plat):
    """Supervised fine-tuning of encoder + head on the labeled subset only, from the saved autoencoder."""
    n_features, n_classes = data["n_features"], data["n_classes"]
    lab = sample_labeled(data["y_tr"], lpc, seed)
    encoder, decoder, head, ae, clf = fresh_models(variant, seed, n_features, n_classes)
    pre = load_pretrained(variant, seed, {"enc": encoder, "dec": decoder})
    opt = tf.keras.optimizers.Adam(LR)
    clf.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    parts = {"enc": encoder, "dec": decoder, "head": head}
    meta = {"stage": "clf", "epoch_done": 0, "hist": {}, "train_seconds": 0.0}
    if os.path.exists(ckpt_path(run_id)):
        meta = load_ckpt(run_id, parts, opt, clf)
        print(f"  Resuming from checkpoint: {meta['epoch_done']} fine-tuning epochs done")
    train_stage("clf", clf, opt, data["x_tr"][lab], tf.keras.utils.to_categorical(data["y_tr"][lab], n_classes),
                (data["x_val"], data["y_val_oh"]), EPOCHS_CLF, meta["epoch_done"], seed, run_id, meta, parts)

    y_te = data["y_te"]
    m, pred, prob = evaluate(clf, data["x_te"], y_te)    # final model, no early stopping
    recall = confusion_matrix(y_te, pred, labels=list(range(n_classes))).diagonal() / np.bincount(y_te, minlength=n_classes)
    row = dict(run_id=run_id, dataset=DATASET, setting="centralized", method=METHOD[variant], variant=variant,
               seed=seed, labels_per_class=lpc, labels_total=len(lab),
               labeled_share=round(len(lab) / len(data["y_tr"]), 4),
               **{k: round(float(v), 6) for k, v in m.items()},
               min_class_recall=round(float(recall.min()), 6),
               **{f"recall_{l.replace(' ', '_')}": round(float(r), 6) for l, r in zip(LABELS, recall)},
               params_encoder=encoder.count_params(), params_decoder=decoder.count_params(),
               params_head=head.count_params(), ae_final_loss=pre["final_loss"],
               ae_final_val_loss=pre["final_val_loss"], epochs_ae=EPOCHS_AE, epochs_clf=EPOCHS_CLF,
               batch=BATCH, lr=LR, n_train=len(data["y_tr"]), n_test=len(y_te),
               ae_seconds=round(pre["train_seconds"], 1), clf_seconds=round(meta["train_seconds"], 1),
               platform=plat, gpu=gpu_name(), tf_version=tf.__version__, timestamp=time.strftime("%Y-%m-%d %H:%M"))
    save_run(run_id, row, y_te, pred, prob, data["test_idx"], {"ae": pre["hist"].get("ae", {}), "clf": meta["hist"].get("clf", {})})
    os.remove(ckpt_path(run_id))
    print(f"Done {run_id}: acc {m['acc']:.4f} | macro-F1 {m['macro_f1']:.4f} | "
          f"lowest class recall {recall.min():.3f} | encoder params {encoder.count_params():,}")


# =============================== SUMMARY + PAIRED TESTS ===============================
def mcnemar_exact(y, pred_a, pred_b):
    a_ok, b_ok = pred_a == y, pred_b == y
    n01, n10 = int(np.sum(a_ok & ~b_ok)), int(np.sum(~a_ok & b_ok))
    if n01 + n10 == 0:
        return n01, n10, 1.0
    return n01, n10, float(stats.binomtest(n01, n01 + n10, 0.5).pvalue)


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order, adj, running = np.argsort(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def write_latex(tab, out):
    """Ready-to-paste LaTeX rows: bold = best mean per dataset and budget."""
    lines = []
    for ds, gds in tab.groupby("dataset", sort=False):
        lines.append(f"% ---- {ds} ----")
        for lpc, g in gds.groupby("labels_per_class", sort=True):
            best_acc, best_f1, best_rec = g.acc_mean.max(), g.f1_mean.max(), g.min_recall_mean.max()
            n = len(g)
            for k, (_, r) in enumerate(g.iterrows()):
                b = lambda val, best, txt: f"\\textbf{{{txt}}}" if np.isclose(val, best) else txt
                acc = b(r.acc_mean, best_acc, f"{100 * r.acc_mean:.2f}") + f" $\\pm$ {100 * r.acc_std:.2f}"
                f1 = b(r.f1_mean, best_f1, f"{100 * r.f1_mean:.2f}") + f" $\\pm$ {100 * r.f1_std:.2f}"
                rec = b(r.min_recall_mean, best_rec, f"{100 * r.min_recall_mean:.1f}")
                first = f"\\multirow{{{n}}}{{*}}{{{lpc:,}}}".replace(",", "{,}") if k == 0 else ""
                lines.append(f"{first} & {LATEX_NAME[r.variant]} & {round(r.params_encoder / 1000)}K & "
                             f"{acc} & {f1} & {rec} \\\\")
            lines.append("\\hline")
    with open(os.path.join(out, "table3_latex.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def summarize(folders):
    rows, preds = [], {}
    for f in [f for f in folders if f]:                     # earlier folders win if a run_id repeats
        if os.path.exists(results_csv(f)):
            rows.append(pd.read_csv(results_csv(f)))
        for p in glob.glob(os.path.join(f, "preds", "*.npz")):
            preds.setdefault(os.path.basename(p)[:-4], p)
    if not rows:
        print("No results found.")
        return
    res = pd.concat(rows, ignore_index=True).drop_duplicates("run_id", keep="first")
    res = res[res["method"].isin(VARIANT_OF)].copy()
    res["variant"] = res["method"].map(VARIANT_OF)
    cols = [c for c in ("epochs_ae", "epochs_clf", "batch", "lr") if c in res.columns]
    if cols and len(res[cols].drop_duplicates()) > 1:
        print("WARNING: the runs in this table do not all use the same training settings:")
        print(res[cols].drop_duplicates().to_string(index=False))

    out = os.path.join(OUT_DIR, "summary")
    os.makedirs(out, exist_ok=True)
    tab = (res.groupby(["dataset", "labels_per_class", "variant"])
              .agg(n_seeds=("seed", "nunique"), acc_mean=("acc", "mean"), acc_std=("acc", "std"),
                   f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std"),
                   min_recall_mean=("min_class_recall", "mean"), params_encoder=("params_encoder", "first"))
              .reset_index())
    tab[["acc_std", "f1_std"]] = tab[["acc_std", "f1_std"]].fillna(0.0)
    tab["_o"] = tab.variant.map({v: i for i, v in enumerate(ORDER)})
    tab = tab.sort_values(["dataset", "labels_per_class", "_o"]).drop(columns="_o")
    tab.to_csv(os.path.join(out, "table3_summary.csv"), index=False)
    pct = lambda m, s: f"{100 * m:.2f} ± {100 * s:.2f}"
    rows_txt = tab.assign(acc=[pct(m, s) for m, s in zip(tab.acc_mean, tab.acc_std)],
                          macro_f1=[pct(m, s) for m, s in zip(tab.f1_mean, tab.f1_std)],
                          lowest_recall=(100 * tab.min_recall_mean).round(1))
    rows_txt = rows_txt[["dataset", "labels_per_class", "variant", "n_seeds", "acc", "macro_f1",
                         "lowest_recall", "params_encoder"]]
    rows_txt.to_csv(os.path.join(out, "table3_rows.csv"), index=False)
    write_latex(tab, out)
    print("\nTABLE III (centralized, test set, mean ± std over seeds, %):")
    print(rows_txt.to_string(index=False))

    pairs = []
    for (ds, lpc), grp in res.groupby(["dataset", "labels_per_class"]):
        for refv in REFERENCES:
            ref = grp[grp.variant == refv].set_index("seed")
            if ref.empty:
                continue
            for v in ORDER:
                if v == refv:
                    continue
                oth = grp[grp.variant == v].set_index("seed")
                for s in sorted(set(ref.index) & set(oth.index)):
                    ra, rb = ref.loc[s], oth.loc[s]
                    row = dict(dataset=ds, labels_per_class=lpc, reference=refv, variant=v, seed=s,
                               acc_reference=ra.acc, acc_variant=rb.acc,
                               f1_reference=ra.macro_f1, f1_variant=rb.macro_f1,
                               only_reference_correct=np.nan, only_variant_correct=np.nan, mcnemar_p=np.nan)
                    if ra.run_id in preds and rb.run_id in preds:
                        da, db = np.load(preds[ra.run_id]), np.load(preds[rb.run_id])
                        if not np.array_equal(da["test_idx"], db["test_idx"]):
                            raise RuntimeError(f"Different test sets: {ra.run_id} vs {rb.run_id}")
                        n01, n10, p = mcnemar_exact(da["y_true"], da["y_pred"], db["y_pred"])
                        row.update(only_reference_correct=n01, only_variant_correct=n10, mcnemar_p=p)
                    pairs.append(row)
    if not pairs:
        return
    pr = pd.DataFrame(pairs)
    pr["mcnemar_p_holm"] = np.nan
    for _, fam in pr.groupby(["dataset", "labels_per_class", "reference"]):    # family: one reference, one budget
        ok = fam.index[fam.mcnemar_p.notna()]
        if len(ok):
            pr.loc[ok, "mcnemar_p_holm"] = holm(pr.loc[ok, "mcnemar_p"])
    pr.to_csv(os.path.join(out, "table3_mcnemar_per_seed.csv"), index=False)

    tests = []
    for (ds, lpc, refv, v), g in pr.groupby(["dataset", "labels_per_class", "reference", "variant"]):
        d_acc, d_f1 = (g.acc_reference - g.acc_variant).values, (g.f1_reference - g.f1_variant).values
        n = len(d_acc)
        t_acc = stats.ttest_rel(g.acc_reference, g.acc_variant).pvalue if n >= 2 and np.std(d_acc) > 0 else np.nan
        t_f1 = stats.ttest_rel(g.f1_reference, g.f1_variant).pvalue if n >= 2 and np.std(d_f1) > 0 else np.nan
        half = stats.t.ppf(0.975, n - 1) * d_acc.std(ddof=1) / np.sqrt(n) if n >= 2 else np.nan
        mc_all = bool(g.mcnemar_p_holm.notna().all() and (g.mcnemar_p_holm < 0.05).all()
                      and (np.sign(d_acc) == np.sign(d_acc.mean())).all())
        tests.append(dict(dataset=ds, labels_per_class=lpc, reference=refv, variant=v, n_seeds=n,
                          diff_acc_pp=100 * d_acc.mean(), ci95_low_pp=100 * (d_acc.mean() - half),
                          ci95_high_pp=100 * (d_acc.mean() + half), ttest_p_acc=t_acc,
                          diff_f1_pp=100 * d_f1.mean(), ttest_p_f1=t_f1, mcnemar_sig_every_seed=mc_all))
    tt = pd.DataFrame(tests)
    tt["ttest_p_acc_holm"] = np.nan
    for _, fam in tt.groupby(["dataset", "labels_per_class", "reference"]):
        ok = fam.index[fam.ttest_p_acc.notna()]
        if len(ok):
            tt.loc[ok, "ttest_p_acc_holm"] = holm(tt.loc[ok, "ttest_p_acc"])
    tt["reference_better"] = (tt.ttest_p_acc_holm < 0.05) & tt.mcnemar_sig_every_seed & (tt.diff_acc_pp > 0)
    tt["variant_better"] = (tt.ttest_p_acc_holm < 0.05) & tt.mcnemar_sig_every_seed & (tt.diff_acc_pp < 0)
    tt["_o"] = tt.variant.map({v: i for i, v in enumerate(ORDER)})
    tt = tt.sort_values(["dataset", "labels_per_class", "reference", "_o"]).drop(columns="_o")
    tt.to_csv(os.path.join(out, "table3_paired_tests.csv"), index=False)
    print("\nPaired tests: reference minus variant (accuracy and macro-F1 in percentage points, paired by seed):")
    print(tt.round(4).to_string(index=False))
    print("\nreference_better / variant_better = paired t-test (Holm within each reference and budget) AND "
          "McNemar (Holm) significant in every seed, same sign. Tables written to", out)


# =============================== MAIN ===============================
def main():
    folders = [OUT_DIR] + list(SUMMARY_DIRS)
    if ONLY_SUMMARIZE:
        summarize(folders)
        return
    plat = platform_name()
    restore_previous_output()
    if DETERMINISTIC:
        tf.config.experimental.enable_op_determinism()
    X, y, n_features = load_data()
    train_idx, val_idx, test_idx = get_split(y)
    n_classes = len(LABELS)
    data = dict(x_tr=X[train_idx], y_tr=y[train_idx], x_val=X[val_idx],
                y_val_oh=tf.keras.utils.to_categorical(y[val_idx], n_classes),
                x_te=X[test_idx], y_te=y[test_idx], test_idx=test_idx,
                n_features=n_features, n_classes=n_classes)
    n_runs = len(VARIANTS) * len(SEEDS) * len(LABELS_PER_CLASS_LIST)
    print(f"ABLATION (all variants) | {DATASET} | train {len(train_idx)} | val {len(val_idx)} | "
          f"test {len(test_idx)} | {plat} | {gpu_name()}")
    print(f"variants {VARIANTS} | labels/class {LABELS_PER_CLASS_LIST} | seeds {SEEDS} | "
          f"epochs AE {EPOCHS_AE} + clf {EPOCHS_CLF} | {len(VARIANTS) * len(SEEDS)} autoencoders | "
          f"{n_runs} fine-tuning runs | QUICK_TEST {QUICK_TEST}")
    try:
        for seed in SEEDS:                       # seed first: complete paired sets are finished early
            for variant in VARIANTS:
                pretrain(variant, seed, data)
                if seed == SEEDS[0] and variant == VARIANTS[0]:
                    verify_determinism(variant, seed, data)
                for lpc in LABELS_PER_CLASS_LIST:
                    run_id = f"{DATASET}_{METHOD[variant]}_L{lpc}_s{seed}"
                    if already_done(run_id):
                        print("Skip (already done):", run_id)
                        continue
                    print(f"\n===== {run_id} =====")
                    finetune(run_id, variant, lpc, seed, data, plat)
    except StopForNow:
        print("\n" + "=" * 70)
        print("STOPPED SAFELY (time budget). All progress is saved in:", OUT_DIR)
        print("Kaggle: add this version's output as input, set RESUME_FROM, Save & Run All.")
        print("=" * 70)
        return
    summarize(folders)


if __name__ == "__main__":
    main()
