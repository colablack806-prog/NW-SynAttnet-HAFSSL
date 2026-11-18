# -*- coding: utf-8 -*-
"""
ET-BERT [Lin et al., WWW 2022] on USTC-TFC2016 (20 classes) - centralized reference (Reviewer 1, Comment 2), R4.

WHAT IS COMPARED (standalone: no federated run is needed):
  * Data      : the de-duplicated USTC data (93,968 flows) and the frozen split of the paper (seed 2026,
                76,113 / 8,458 / 9,397); the sizes are checked.
  * Labels    : 100 labeled flows per class (2,000), drawn from the training split with seed s by the same
                sample_labeled() function as 08_centralized_USTC_L2000.py, so for each seed ET-BERT, SynAttnNet,
                AECNN and AECNN-ft use the SAME labeled flows (checked when CEN_USTC_L2000 is attached).
  * Seeds     : 0, 1, 2 (the seeds of the centralized runs).
  * Model     : the authors' PRE-TRAINED ET-BERT, fine-tuned with the authors' README settings
                (seq_length 128, batch 32, learning rate 2e-5, 10 epochs, word_pos_seg / transformer /
                fully_visible). As in the authors' code, the epoch with the best validation accuracy is kept;
                the test set is used once, at the end. The script stops if the pre-trained model is missing.
  * Comparison: paired tests (McNemar per seed, t-test over seeds) against the centralized SynAttnNet,
                AECNN and AECNN-ft of 08_centralized_USTC_L2000.py.

KAGGLE SETUP (one notebook, GPU T4 x1, Internet ON in Settings, right panel):
  Add as Input:
    1. the USTC dataset (the 6 .npy files)
    2. the ET-BERT pre-trained model (pre-trained_model.bin, ~400 MB) - the same Kaggle dataset as before
    3. (for the paired tests) the output CEN_USTC_L2000 of 08_centralized_USTC_L2000.py. If it is not attached
       yet, the ET-BERT runs still work; rerun later with it attached: finished seeds are skipped and only the
       paired tests are computed.
  Paste this whole file into ONE cell. QUICK_TEST = True first (1 epoch, seed 0), then False -> Save & Run All.
  Expected time on a T4: about 10 minutes per seed.
"""
import os, sys, glob, json, time, subprocess
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, balanced_accuracy_score,
                             confusion_matrix, classification_report)
from scipy import stats

# =============================== CONFIG ===============================
BASE_PATH  = "data/USTC-TFC2016/"   # the 6 .npy files
MODEL_BIN  = "pre-trained_model.bin"                      # path of the pre-trained ET-BERT checkpoint; "" = search /kaggle/input
REF_SEARCH = "/kaggle/input"          # where the CEN_USTC_L2000 output (centralized runs) is searched for
OUT_DIR    = "/kaggle/working/CEN_USTC_ETBERT_L2000"
REPO_DIR   = "/kaggle/working/ET-BERT"

DATASET    = "USTC"
LABELS     = ['BitTorrent', 'Facetime', 'FTP', 'Gmail', 'MySQL', 'Outlook', 'Skype', 'SMB', 'Weibo',
              'WorldOfWarcraft', 'Cridex', 'Geodo', 'Htbot', 'Miuref', 'Neris', 'Nsis-ay', 'Shifu',
              'Tinba', 'Virut', 'Zeus']
SEEDS      = [0, 1, 2]               # the seeds of the centralized runs

# Reference runs (08_centralized_USTC_L2000.py): name -> (output folder prefix, prediction file pattern)
REF_RUNS = {
    "SynAttnNet (centralized)": ("CEN_USTC_L2000", "USTC_SynAttnNet_L2000_s{}.npz"),
    "AECNN (centralized)":      ("CEN_USTC_L2000", "USTC_AECNN_L2000_s{}.npz"),
    "AECNN-ft (centralized)":   ("CEN_USTC_L2000", "USTC_AECNN-ft_L2000_s{}.npz"),
}
LABELS_PER_CLASS = 100                          # 2,000 labeled flows, as in the centralized runs
EXPECTED_SIZES   = (93968, 76113, 8458, 9397)   # flows after de-duplication, train, validation, test

# Checkpoint search
SEARCH_ROOTS = ["/kaggle/input", "/kaggle/working"]
CKPT_EXTS   = (".bin", ".pt", ".pth", ".ckpt")
MIN_CKPT_MB = 50                      # ET-BERT base is ~400 MB
GDRIVE_ID   = ""                      # optional: Drive id from the authors' README (needs Internet ON)

# Data protocol of the USTC runs - do not change
SPLIT_SEED, MAX_LEN = 2026, 128
# ET-BERT fine-tuning (authors' README example) - do not change
EPOCHS, BATCH, LR, SEQ_LEN = 10, 32, 2e-5, 128

REUSE_FINETUNED = True     # reuse finetuned_model.bin of an interrupted run instead of fine-tuning again

QUICK_TEST = False       # True = 1 epoch, seed 0
if QUICK_TEST:
    EPOCHS, SEEDS = 1, [0]
    OUT_DIR += "_quicktest"
_FAKE = os.environ.get("ETBERT_FAKE") == "1"     # testing only: skip BERT, write random predictions


# =============================== INPUT DISCOVERY ===============================
def large_files(root="/kaggle/input", min_mb=10):
    out = []
    for p in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        try:
            if os.path.isfile(p) and not os.path.islink(p):
                mb = os.path.getsize(p) / 1e6
                if mb >= min_mb:
                    out.append((p, mb))
        except OSError:
            continue
    return sorted(out, key=lambda t: -t[1])


def is_lfs_pointer(path):
    try:
        if os.path.getsize(path) > 1024:
            return False
        with open(path, "rb") as f:
            return f.read(40).startswith(b"version https://git-lfs")
    except OSError:
        return False


def autodetect_base_path():
    global BASE_PATH
    if os.path.exists(os.path.join(BASE_PATH, "x_payload_train.npy")):
        return BASE_PATH
    hits = glob.glob(os.path.join(REF_SEARCH, "**", "x_payload_train.npy"), recursive=True)
    if not hits:
        raise FileNotFoundError(
            "x_payload_train.npy not found. Attach the USTC dataset as an Input, "
            f"or set BASE_PATH (currently {BASE_PATH}).")
    BASE_PATH = os.path.dirname(hits[0]) + os.sep
    print("BASE_PATH auto-detected:", BASE_PATH)
    return BASE_PATH


def download_checkpoint():
    if not GDRIVE_ID:
        return None
    dst = "/kaggle/working/etbert_pretrained_model.bin"
    if os.path.exists(dst) and os.path.getsize(dst) > MIN_CKPT_MB * 1e6:
        return dst
    print(f"Downloading the pre-trained checkpoint from Google Drive (id {GDRIVE_ID}) ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=False)
    r = subprocess.run([sys.executable, "-m", "gdown", "--id", GDRIVE_ID, "-O", dst],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(dst) or os.path.getsize(dst) < MIN_CKPT_MB * 1e6:
        print("  download failed:", (r.stderr or r.stdout)[-400:])
        return None
    return dst


def find_model_bin(verbose=True):
    if MODEL_BIN:
        if not os.path.exists(MODEL_BIN):
            raise FileNotFoundError(f"MODEL_BIN is set but does not exist: {MODEL_BIN}")
        return MODEL_BIN
    data_dir, work_root = os.path.abspath(BASE_PATH), os.path.abspath(OUT_DIR)
    found = []
    for root in SEARCH_ROOTS:
        if os.path.isdir(root):
            found += large_files(root, min_mb=1)
    cands, rejected = [], []
    for p, mb in found:
        if os.path.abspath(p).startswith(work_root) or not p.lower().endswith(CKPT_EXTS):
            continue
        if os.path.abspath(os.path.dirname(p)).startswith(data_dir):
            continue
        if is_lfs_pointer(p):
            rejected.append((p, mb, "Git-LFS pointer, not the real file"))
        elif mb < MIN_CKPT_MB:
            rejected.append((p, mb, f"only {mb:.1f} MB, expected ~400 MB"))
        elif "finetuned" in os.path.basename(p).lower():
            rejected.append((p, mb, "a fine-tuned model, not the pre-trained one"))
        else:
            cands.append((p, mb))
    if not cands:
        cands = [(p, os.path.getsize(p) / 1e6) for p in [download_checkpoint() or ""] if p]
    if not cands:
        msg = ["Pre-trained ET-BERT checkpoint not found.", "Large files that ARE present (>10 MB):"]
        big = [t for r in SEARCH_ROOTS if os.path.isdir(r) for t in large_files(r, min_mb=10)]
        msg += [f"  {mb:8.0f} MB  {p}" for p, mb in big[:40]] or ["  (none)"]
        msg += [f"  rejected: {p} -> {why}" for p, mb, why in rejected]
        raise FileNotFoundError("\n".join(msg))

    def score(t):
        s = t[0].lower()
        return (("bert" in s) * 4 + ("pre" in s) * 2 + ("fine" not in s) * 1, t[1])
    cands.sort(key=score, reverse=True)
    if verbose:
        print(f"Pre-trained model: {cands[0][0]} ({cands[0][1]:.0f} MB)")
    return cands[0][0]


def check_checkpoint(path):
    if _FAKE:
        return -1
    import torch
    try:
        sd = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise RuntimeError(f"The checkpoint at {path} could not be loaded by torch:\n  {str(e)[:300]}")
    if not isinstance(sd, dict) or not any(k.startswith(("embedding", "encoder")) for k in sd):
        raise RuntimeError(f"{path} does not look like the uer/ET-BERT pre-trained state dict.")
    n = int(sum(v.numel() for v in sd.values() if hasattr(v, "numel")))
    print(f"  checkpoint OK: {len(sd)} tensors, {n / 1e6:.1f} M parameters")
    if any("target" in k and "classifier" in k for k in sd):
        print("  WARNING: this checkpoint already contains a classification head. "
              "Use the authors' PRE-TRAINED model, not a fine-tuned one.")
    return n


# =============================== DATA: identical to the USTC federated scripts ===============================
def hex_to_tokens(hex_str, max_len=MAX_LEN):
    ids = [int(t, 16) for t in str(hex_str).strip().split() if len(t) == 4][:max_len]
    ids += [0] * max(0, max_len - len(ids))
    return np.array(ids, dtype=np.float32) / 65535.0


def load_data():
    """Same pooling, de-duplication and order as the USTC federated scripts; returns the raw strings."""
    parts_x, parts_y = [], []
    for name in ("train", "valid", "test"):
        parts_x.append(np.load(os.path.join(BASE_PATH, f"x_payload_{name}.npy"), allow_pickle=True))
        parts_y.append(np.load(os.path.join(BASE_PATH, f"y_{name}.npy"), allow_pickle=True))
    raw = np.concatenate(parts_x)
    X = np.stack([hex_to_tokens(s) for s in raw])
    y = np.concatenate(parts_y).astype(int)
    n0 = len(y)
    key = pd.util.hash_pandas_object(pd.DataFrame(X), index=False).values
    labels_per_key = pd.Series(y).groupby(key).nunique()
    ambiguous = np.isin(key, labels_per_key[labels_per_key > 1].index.values)
    keep = (~ambiguous) & (~pd.Series(key).duplicated().values)
    print(f"Rows: {n0} -> {int(keep.sum())} after removing {int(ambiguous.sum())} rows with ambiguous labels "
          f"and {int((~ambiguous).sum() - keep.sum())} exact duplicates")
    return raw[keep], y[keep]


def get_split(y):
    idx = np.arange(len(y))
    trval, test = train_test_split(idx, test_size=0.10, stratify=y, random_state=SPLIT_SEED)
    train, val = train_test_split(trval, test_size=0.10, stratify=y[trval], random_state=SPLIT_SEED)
    return train, val, test


# =============================== split, labeled flows, reference runs ===============================
def sample_labeled(y_train, per_class, seed):
    """Same function as 08_centralized_USTC_L2000.py (Table II rule): same seed = same labeled flows."""
    rng = np.random.default_rng(seed)
    chosen = []
    for c in np.unique(y_train):
        pool = np.where(y_train == c)[0]
        chosen.extend(rng.choice(pool, min(per_class, len(pool)), replace=False))
    return np.array(chosen)


def ref_dirs(prefix="CEN_USTC_L2000"):
    """Output folders of the centralized runs (they contain splits/)."""
    return sorted({d for d in glob.glob(os.path.join(REF_SEARCH, "**", prefix + "*"), recursive=True)
                   if os.path.isdir(d) and os.path.isdir(os.path.join(d, "splits"))})


def check_split(y, tr, va, te):
    got = (len(y), len(tr), len(va), len(te))
    if got != EXPECTED_SIZES:
        raise RuntimeError(f"Data sizes {got} differ from the paper {EXPECTED_SIZES}: check BASE_PATH and the files.")
    checked = []
    for d in ref_dirs():
        f = os.path.join(d, "splits", f"{DATASET}_split.npz")
        if os.path.exists(f):
            s = np.load(f, allow_pickle=True)
            if not (np.array_equal(s["train"], tr) and np.array_equal(s["val"], va) and np.array_equal(s["test"], te)):
                raise RuntimeError(f"Split differs from {f}.")
            checked.append(f)
    return checked


def check_labeled(seed, lab_idx):
    """If the centralized output is attached, its labeled flows of this seed must be the same."""
    for d in ref_dirs():
        f = os.path.join(d, "splits", f"{DATASET}_labeled_s{seed}.npz")
        if os.path.exists(f):
            if not np.array_equal(np.sort(np.load(f)["global_indices"]), np.sort(lab_idx)):
                raise RuntimeError(f"Labeled flows of seed {seed} differ from {f}.")
            return True
    return False


# =============================== helpers ===============================
def write_tsv(path, texts, labels=None):
    with open(path, "w", encoding="utf-8") as f:
        f.write("label\ttext_a\n" if labels is not None else "text_a\n")
        for i, t in enumerate(texts):
            t = " ".join(str(t).split())
            f.write(f"{int(labels[i])}\t{t}\n" if labels is not None else f"{t}\n")


def setup_repo():
    if not os.path.isdir(REPO_DIR):
        r = subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/linwhitehat/ET-BERT.git",
                            REPO_DIR], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("git clone failed. Turn Internet ON (Settings, right panel).\n" + r.stderr[-500:])
    missing = [f for f in ("fine-tuning/run_classifier.py", "inference/run_classifier_infer.py",
                           "models/encryptd_vocab.txt", "models/bert/base_config.json")
               if not os.path.exists(os.path.join(REPO_DIR, f))]
    if missing:
        raise RuntimeError(f"missing in the ET-BERT repository at {REPO_DIR}: {missing}")


def run(cmd, log_path):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([REPO_DIR, os.path.join(REPO_DIR, "fine-tuning")]))
    for attempt in (1, 2):
        with open(log_path, "w") as log:
            r = subprocess.run(cmd, cwd=REPO_DIR, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        full = open(log_path, encoding="utf-8", errors="replace").read()
        if r.returncode == 0:
            return full
        if attempt == 1 and "weights_only" in full:
            print("  PyTorch >= 2.6 'weights_only' refusal: retrying with TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1")
            env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
            continue
        raise RuntimeError(f"ET-BERT step failed (see {log_path}):\n{full[-3000:]}")


def parse_dev_acc(log_path):
    import re
    if not os.path.exists(log_path):
        return []
    out = []
    for line in open(log_path, encoding="utf-8", errors="replace"):
        if "acc" in line.lower() and ("correct" in line.lower() or line.strip().lower().startswith("acc")):
            nums = re.findall(r"\d*\.\d+", line)
            if nums:
                out.append(float(nums[0]))
    return [round(v, 4) for v in out]


def unknown_token_rate(texts, n=2000):
    vocab = {l.rstrip("\n") for l in open(os.path.join(REPO_DIR, "models", "encryptd_vocab.txt"), encoding="utf-8")}
    toks = [t.lower() for s in texts[:n] for t in str(s).split()[:SEQ_LEN - 1]]
    return float(np.mean([t not in vocab for t in toks])) if toks else float("nan")


def evaluate(prob, y, n_classes):
    pred = prob.argmax(axis=1)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, labels=list(range(n_classes)), average='macro',
                                                  zero_division=0)
    m = {'acc': accuracy_score(y, pred), 'macro_precision': p, 'macro_recall': r, 'macro_f1': f1,
         'balanced_acc': balanced_accuracy_score(y, pred)}
    rec = confusion_matrix(y, pred, labels=list(range(n_classes))).diagonal() / np.maximum(
        np.bincount(y, minlength=n_classes), 1)
    return m, pred, rec


def results_csv():
    return os.path.join(OUT_DIR, "results", "all_runs.csv")


def already_done(run_id):
    p = results_csv()
    return os.path.exists(p) and run_id in set(pd.read_csv(p)["run_id"])


def count_params(bin_path):
    try:
        import torch
        sd = torch.load(bin_path, map_location="cpu", weights_only=False)
        return int(sum(v.numel() for v in sd.values() if hasattr(v, "numel")))
    except Exception as e:
        print("  (parameter count skipped:", str(e)[:120], ")")
        return -1


# =============================== ONE RUN ===============================
def run_one(run_id, seed, raw, y, tr, va, te, model_bin):
    n_classes = len(LABELS)
    work = os.path.join(OUT_DIR, "work", run_id)
    os.makedirs(work, exist_ok=True)
    sel = sample_labeled(y[tr], LABELS_PER_CLASS, seed)
    lab_idx = tr[sel]                                          # global indices of the 2,000 labeled flows
    per_class = np.bincount(y[lab_idx], minlength=n_classes)
    same = check_labeled(seed, lab_idx)
    print(f"  labeled flows: {len(lab_idx)} | per class {per_class.min()}-{per_class.max()} | "
          f"{'identical to the centralized runs' if same else 'centralized output not attached: not compared'}")
    assert len(lab_idx) == 2000 and per_class.min() == 100 == per_class.max(), "expected 100 labeled flows per class"

    unk = unknown_token_rate(raw[lab_idx]) if not _FAKE else float("nan")
    if not _FAKE:
        print(f"  tokens not in the ET-BERT vocabulary: {100 * unk:.1f}% (become [UNK])")
    write_tsv(os.path.join(work, "train_dataset.tsv"), raw[lab_idx], y[lab_idx])
    write_tsv(os.path.join(work, "valid_dataset.tsv"), raw[va], y[va])
    write_tsv(os.path.join(work, "nolabel_test_dataset.tsv"), raw[te])
    ft_model = os.path.join(work, "finetuned_model.bin")
    pred_tsv = os.path.join(work, "prediction.tsv")
    common = ["--vocab_path", "models/encryptd_vocab.txt", "--config_path", "models/bert/base_config.json",
              "--embedding", "word_pos_seg", "--encoder", "transformer", "--mask", "fully_visible",
              "--seq_length", str(SEQ_LEN)]
    t0 = time.time()
    if _FAKE:
        rng = np.random.default_rng(seed)
        p = rng.dirichlet(np.ones(n_classes), size=len(te))
        pd.DataFrame({"label": p.argmax(1), "prob": [" ".join(map(str, r)) for r in p]}).to_csv(
            pred_tsv, sep="\t", index=False)
    elif REUSE_FINETUNED and os.path.exists(ft_model):
        print(f"  reusing the fine-tuned model already in {work}")
    else:
        run([sys.executable, "fine-tuning/run_classifier.py", "--pretrained_model_path", model_bin,
             "--output_model_path", ft_model,
             "--train_path", os.path.join(work, "train_dataset.tsv"),
             "--dev_path", os.path.join(work, "valid_dataset.tsv"),
             "--epochs_num", str(EPOCHS), "--batch_size", str(BATCH), "--learning_rate", str(LR),
             "--seed", str(seed), "--report_steps", "100"] + common,
            os.path.join(work, "finetune.log"))
        print(f"  validation accuracy per epoch: {parse_dev_acc(os.path.join(work, 'finetune.log'))}")
        if not os.path.exists(ft_model):
            raise RuntimeError(f"fine-tuning finished but {ft_model} was not written")
    if not _FAKE:
        print("  inference on the test set ...")
        run([sys.executable, "inference/run_classifier_infer.py", "--load_model_path", ft_model,
             "--test_path", os.path.join(work, "nolabel_test_dataset.tsv"), "--prediction_path", pred_tsv,
             "--labels_num", str(n_classes), "--batch_size", "128", "--output_prob"] + common,
            os.path.join(work, "infer.log"))
    sec = time.time() - t0

    pr = pd.read_csv(pred_tsv, sep="\t", dtype={"prob": str})
    assert len(pr) == len(te), f"{len(pr)} predictions for {len(te)} test flows"
    prob = np.array([np.array(s.split(), dtype=np.float32) for s in pr["prob"]])
    assert np.array_equal(prob.argmax(1), pr["label"].values), "predicted labels and probabilities disagree"
    y_te = y[te]
    m, pred, rec = evaluate(prob, y_te, n_classes)

    for sub in ("results", "preds", "reports"):
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)
    np.savez_compressed(os.path.join(OUT_DIR, "preds", f"{run_id}.npz"),
                        y_true=y_te, y_pred=pred, y_prob=prob, test_idx=te, labeled_idx=lab_idx)
    pd.DataFrame(classification_report(y_te, pred, labels=list(range(n_classes)), target_names=LABELS, digits=4,
                                       output_dict=True, zero_division=0)).T.to_csv(
        os.path.join(OUT_DIR, "reports", f"{run_id}_per_class.csv"))
    pd.DataFrame(confusion_matrix(y_te, pred, labels=list(range(n_classes))), index=LABELS, columns=LABELS).to_csv(
        os.path.join(OUT_DIR, "reports", f"{run_id}_confusion.csv"))
    row = dict(run_id=run_id, dataset=DATASET, setting="centralized", method="ET-BERT", seed=seed,
               labels_total=len(lab_idx), label_source="sample_labeled(y_train, 100, seed), shared with the centralized runs",
               epochs=EPOCHS, batch=BATCH, lr=LR, seq_length=SEQ_LEN, model_selection="best validation epoch",
               unknown_token_rate=round(unk, 4) if unk == unk else -1,
               **{k: round(float(v), 6) for k, v in m.items()},
               min_class_recall=round(float(rec.min()), 6), classes_below_50=int((rec < 0.5).sum()),
               **{f"recall_{l}": round(float(r), 6) for l, r in zip(LABELS, rec)},
               params=(-1 if _FAKE else count_params(ft_model)), train_seconds=round(sec, 1),
               timestamp=time.strftime("%Y-%m-%d %H:%M"))
    p = results_csv()
    pd.DataFrame([row]).to_csv(p, mode="a", header=not os.path.exists(p), index=False)
    if os.path.exists(ft_model):
        os.remove(ft_model)
    print(f"Done {run_id}: test acc {m['acc']:.4f} | macro-F1 {m['macro_f1']:.4f} | "
          f"classes below 50% recall {int((rec < 0.5).sum())} | {sec / 60:.1f} min")


# =============================== paired tests vs HAFSSL and FL-AECNN (Table X) ===============================
def holm(pv):
    p = np.asarray(pv, float); order = np.argsort(p); adj = np.empty_like(p); run_max = 0.0
    for rank, i in enumerate(order):
        run_max = max(run_max, (len(p) - rank) * p[i]); adj[i] = min(1.0, run_max)
    return adj


def summarize():
    if not os.path.exists(results_csv()):
        print("\nNo finished runs to summarize.")
        return
    res = pd.read_csv(results_csv())
    print("\nET-BERT on USTC, 2,000 labeled flows (test set, mean ± std over seeds):")
    print(res[["acc", "macro_f1", "min_class_recall", "classes_below_50"]].agg(["mean", "std"]).round(4).to_string())
    rows = []
    for ref, (prefix, pattern) in REF_RUNS.items():
        for s in sorted(res.seed.unique()):
            hits = [f for d in ref_dirs(prefix) for f in glob.glob(os.path.join(d, "preds", pattern.format(s)))]
            own = os.path.join(OUT_DIR, "preds", f"{DATASET}_ET-BERT_L2000_s{s}.npz")
            if not hits or not os.path.exists(own):
                continue
            a, b = np.load(own), np.load(hits[0])
            assert np.array_equal(a["test_idx"], b["test_idx"]) and np.array_equal(a["y_true"], b["y_true"]), \
                "different test sets"
            if "labeled_idx" in b.files:
                assert np.array_equal(np.sort(a["labeled_idx"]), np.sort(b["labeled_idx"])), "different labeled flows"
            y = a["y_true"]
            n01 = int(np.sum((a["y_pred"] == y) & (b["y_pred"] != y)))      # only ET-BERT correct
            n10 = int(np.sum((a["y_pred"] != y) & (b["y_pred"] == y)))      # only the reference correct
            p = 1.0 if n01 + n10 == 0 else float(stats.binomtest(n01, n01 + n10, 0.5).pvalue)
            rows.append(dict(reference=ref, seed=s, acc_etbert=float(np.mean(a["y_pred"] == y)),
                             acc_ref=float(np.mean(b["y_pred"] == y)), only_etbert_correct=n01,
                             only_ref_correct=n10, mcnemar_p=p))
    if not rows:
        print("\nCEN_USTC_L2000 predictions not found -> paired tests skipped (attach it and run again).")
        return
    pr = pd.DataFrame(rows)
    pr["mcnemar_p_holm"] = holm(pr.mcnemar_p)
    pr["diff_pp"] = 100 * (pr.acc_etbert - pr.acc_ref)
    pr.to_csv(os.path.join(OUT_DIR, "etbert_mcnemar_per_seed.csv"), index=False)
    tt = []
    for ref, g in pr.groupby("reference"):
        tp = float(stats.ttest_rel(g.acc_etbert, g.acc_ref).pvalue) if len(g) >= 2 and g.diff_pp.std() > 0 else np.nan
        tt.append(dict(reference=ref, n_seeds=len(g), mean_diff_pp=g.diff_pp.mean(),
                       etbert_higher_in=int((g.diff_pp > 0).sum()), ttest_p=tp,
                       mcnemar_sig_seeds=int((g.mcnemar_p_holm < 0.05).sum())))
    tt = pd.DataFrame(tt)
    ok = tt.ttest_p.notna()
    tt["ttest_p_holm"] = np.nan
    if ok.any():
        tt.loc[ok, "ttest_p_holm"] = holm(tt.loc[ok, "ttest_p"])
    tt.to_csv(os.path.join(OUT_DIR, "etbert_paired_tests.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\nPer seed (ET-BERT minus reference):\n" + pr.round(4).to_string(index=False))
    print("\nPaired over seeds:\n" + tt.round(4).to_string(index=False))


# =============================== PREFLIGHT ===============================
def preflight():
    print("=" * 78)
    print("PREFLIGHT")
    autodetect_base_path()
    print(f"  USTC data            : {BASE_PATH}  ({len(glob.glob(os.path.join(BASE_PATH, '*.npy')))} .npy files)")
    ck = "" if _FAKE else find_model_bin(verbose=False)
    if ck:
        print(f"  pre-trained ET-BERT  : {ck}  ({os.path.getsize(ck) / 1e6:.0f} MB)")
    ds = ref_dirs()
    print(f"  centralized runs     : {ds[0] if ds else 'not attached (paired tests later)'}")
    try:
        import torch
        print("  GPU                  :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
    except Exception:
        print("  GPU                  : torch not importable")
    print("=" * 78)
    return ck


# =============================== MAIN ===============================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model_bin = preflight()
    raw, y = load_data()
    assert len(np.unique(y)) == len(LABELS), f"{len(np.unique(y))} classes found, LABELS has {len(LABELS)}"
    tr, va, te = get_split(y)
    checked = check_split(y, tr, va, te)
    print(f"Train {len(tr)} | Val {len(va)} | Test {len(te)} | sizes as in the paper"
          + (f" | split identical to {checked[0]}" if checked else ""))
    print(f"ET-BERT | epochs {EPOCHS} | batch {BATCH} | lr {LR} | seq_length {SEQ_LEN} | seeds {SEEDS} | "
          f"QUICK_TEST {QUICK_TEST}")
    if not _FAKE:
        setup_repo()
        check_checkpoint(model_bin)
    with open(os.path.join(OUT_DIR, "config_etbert.json"), "w") as f:
        json.dump(dict(seeds=SEEDS, epochs=EPOCHS, batch=BATCH, lr=LR, seq_length=SEQ_LEN, split_seed=SPLIT_SEED,
                       labels="sample_labeled(y_train, 100, seed), shared with the centralized runs",
                       base_path=BASE_PATH, model_bin=model_bin), f, indent=2)
    for seed in SEEDS:
        run_id = f"{DATASET}_ET-BERT_L2000_s{seed}"
        if already_done(run_id):
            print("Skip (already done):", run_id)
            continue
        print(f"\n===== {run_id} =====")
        run_one(run_id, seed, raw, y, tr, va, te, model_bin)
    summarize()


main()
