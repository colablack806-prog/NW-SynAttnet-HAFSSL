# =====================================================================================
# USTC-TFC2016: facts for the dataset paragraph (Section V-A, item 3). About 1 minute.
# Your USTC data are six .npy files (x_payload_*/y_* for train, valid and test).
# Run this cell in a notebook that has the "ustc-dataset" as input and send me all it prints.
# =====================================================================================
import hashlib, os
import numpy as np
import pandas as pd

DATA_DIR = "data/USTC-TFC2016"
N_WORDS = 128                                                                       # max_input_words of your runs
PARTS = ["train", "valid", "test"]

print("Files in", DATA_DIR)
for f in sorted(os.listdir(DATA_DIR)):
    p = os.path.join(DATA_DIR, f)
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    print(f"  {f:22s} {os.path.getsize(p)/1e6:8.1f} MB   sha256 {h}")

X_parts, y_parts = [], []
for part in PARTS:
    x = np.load(os.path.join(DATA_DIR, f"x_payload_{part}.npy"), allow_pickle=True)
    y = np.load(os.path.join(DATA_DIR, f"y_{part}.npy"), allow_pickle=True)
    print(f"\n{part}: x shape {x.shape}, dtype {x.dtype} | y shape {y.shape}, dtype {y.dtype}")
    X_parts.append(x); y_parts.append(y)

X = np.concatenate(X_parts); y = np.concatenate(y_parts)
print(f"\nAll three files together: {len(X):,} flows")
vals, cnt = np.unique(y, return_counts=True)
print(f"Classes: {len(vals)} | flows per class: {dict(zip(vals.tolist(), cnt.tolist()))}")

first = X[0]
print("\nExample of one flow as stored (first 400 characters):")
print(repr(first)[:400])

if isinstance(first, (str, bytes, np.str_)):
    # text format: words separated by spaces
    words = [str(s).split() for s in X]
    lens = np.array([len(w) for w in words])
    print(f"\nWords per flow: min {lens.min()}, median {int(np.median(lens))}, max {lens.max()}; "
          f"flows with fewer than {N_WORDS} words: {(lens < N_WORDS).mean():.3f}")
    print("First 12 words of the first flow:", words[0][:12])
    wl = pd.Series([len(t) for w in words[:2000] for t in w[:N_WORDS]]).value_counts().head(5).to_dict()
    print("Length of the words (characters -> count, first 2,000 flows):", wl)
    keys = [" ".join(w[:N_WORDS]) for w in words]          # what the model sees: the first 128 words
    full = [" ".join(w) for w in words]
else:
    A = np.asarray(X).reshape(len(X), -1).astype(np.float64)
    print(f"\nNumeric format: {A.shape[1]} values per flow, min {A.min():.6g}, max {A.max():.6g}")
    keys = [a[:N_WORDS].tobytes() for a in A]
    full = [a.tobytes() for a in A]

k = pd.Series(keys); f = pd.Series(full)
print(f"\nExact duplicates over the first {N_WORDS} words: {int(k.duplicated().sum()):,} "
      f"-> {int((~k.duplicated()).sum()):,} flows left")
print(f"Exact duplicates over the whole stored flow: {int(f.duplicated().sum()):,} "
      f"-> {int((~f.duplicated()).sum()):,} flows left")
print(f"Duplicates over the first {N_WORDS} words with a different label: "
      f"{int(pd.DataFrame({'k': keys, 'y': y}).drop_duplicates().duplicated('k').sum()):,}")
print("Your runs used 93,968 flows (76,113 train / 8,458 val / 9,397 test).")
