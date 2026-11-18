# =====================================================================================
# PARTITION CHECK (E24): do FEAT, FLUIDS-fed and CNN-FedAvg use the same client partitions and
# labeled flows as HAFSSL / FL-AECNN in the aggregator runs?
#
# Run it in a NEW, separate notebook (CPU is enough, about 1 minute, no training).
# Inputs needed (you already have all three):
#   - notebook  fluids-fed-vs-feat-federated   (its __notebook__.ipynb = the baseline code)
#   - notebook  aggregator-hafssl-fl-aecnn     (T7_gpu0/splits/ISCX_partition_a0.5_K10_L1000_s*.npz)
#   - dataset   isxc-dataset                   (ISCX_5class_each_normalized_cuttedfloefeature.csv)
# Paste this whole file into one cell and run it. Read the VERDICT at the end.
#
# The baseline notebook is NOT run: only its imports, constants and function definitions are
# loaded, so no training starts.
# =====================================================================================
import ast, json, os, inspect
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_ROOT     = "experiments"   # folder searched for the saved partition files (on Kaggle: "/kaggle/input")
DATA_CSV       = "data/ISCX-VPN2016/ISCX_5class_each_normalized_cuttedfloefeature.csv"
BASELINE_HINT  = "fluids-fed-vs-feat-federated"     # folder name of the baseline notebook
REFERENCE_HINT = "aggregator-hafssl-fl-aecnn"        # folder name of the aggregator notebook
SEEDS, ALPHA, K, PER_CLASS, SPLIT_SEED = [0, 1, 2], 0.5, 10, 1000, 2026

# Leave empty. Only if the script says it cannot pick a function, write the right names here.
PART_FUNC_NAME = ""
LAB_FUNC_NAME  = ""


def find_all(filename):
    """All files with this exact name under INPUT_ROOT (follows Kaggle's linked folders)."""
    hits = []
    for root, _, files in os.walk(INPUT_ROOT, followlinks=True):
        if filename in files:
            hits.append(os.path.join(root, filename))
    return sorted(hits)


def prefer(paths, *words):
    for p in paths:
        if all(w in p for w in words):
            return p
    return paths[0] if paths else None


# ---------- 1. Reference partitions (aggregator runs) ----------
ref_paths = {}
for s in SEEDS:
    hits = find_all(f"ISCX_partition_a{ALPHA}_K{K}_L{PER_CLASS}_s{s}.npz")
    ref_paths[s] = prefer(hits, REFERENCE_HINT, "T7_gpu0")
if any(p is None for p in ref_paths.values()):
    print("Folders under", INPUT_ROOT, ":", sorted(os.listdir(INPUT_ROOT)))
    raise FileNotFoundError("Partition files not found. Add the notebook 'aggregator-hafssl-fl-aecnn' as input.")
print("Reference partition files (aggregator runs):")
for s, p in ref_paths.items():
    print(f"  seed {s}: {p}")
for s, p in ref_paths.items():                      # the two GPU halves must agree with each other
    other = p.replace("T7_gpu0", "T7_gpu1")
    if other != p and os.path.exists(other):
        a, b = np.load(p), np.load(other)
        print(f"  seed {s}: T7_gpu0 and T7_gpu1 files identical:",
              all(np.array_equal(a[k], b[k]) for k in a.files))

# ---------- 2. Training labels, built exactly as in all R2 scripts ----------
df = pd.read_csv(DATA_CSV)
feat_cols = [c for c in df.columns if c != "label"]
df = df.drop_duplicates(subset=feat_cols).reset_index(drop=True)
y = df["label"].values.astype(int)
idx = np.arange(len(y))
trval, test = train_test_split(idx, test_size=0.10, stratify=y, random_state=SPLIT_SEED)
train, val = train_test_split(trval, test_size=0.10, stratify=y[trval], random_state=SPLIT_SEED)
print(f"\nRows after de-duplication: {len(y)} | train {len(train)} | val {len(val)} | test {len(test)}")
split_ref = os.path.join(os.path.dirname(ref_paths[SEEDS[0]]), "ISCX_split.npz")
if os.path.exists(split_ref):
    sr = np.load(split_ref)
    same_split = all(np.array_equal(sr[k], v) for k, v in [("train", train), ("val", val), ("test", test)])
    print("Split identical to the aggregator split file:", same_split)
    if not same_split:
        raise RuntimeError("The rebuilt split differs from the aggregator split; check DATA_CSV.")
y_tr = y[train]


# ---------- 3. Load the baseline notebook's definitions WITHOUT running it ----------
def code_cells(ipynb_path):
    nb = json.load(open(ipynb_path, encoding="utf-8"))
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            src = cell.get("source", "")
            src = "".join(src) if isinstance(src, list) else src
            yield "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("!", "%")))


def has_call(node):
    return any(isinstance(n, ast.Call) for n in ast.walk(node))


def load_definitions(ipynb_path):
    """Executes only imports, function/class definitions and constant assignments."""
    ns = {"__name__": "baseline"}
    for i, src in enumerate(code_cells(ipynb_path)):
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            print(f"  (cell {i} skipped: {e})")
            continue
        for node in tree.body:
            keep = isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)) or \
                   (isinstance(node, (ast.Assign, ast.AnnAssign)) and not has_call(node))
            if not keep:
                continue
            try:
                exec(compile(ast.Module(body=[node], type_ignores=[]), f"<baseline cell {i}>", "exec"), ns)
            except Exception as e:
                print(f"  (skipped one statement in cell {i}: {type(e).__name__}: {e})")
    return ns


nb_hits = [p for p in find_all("__notebook__.ipynb") if BASELINE_HINT in p]
if not nb_hits:
    raise FileNotFoundError("Baseline notebook source not found. Add 'fluids-fed-vs-feat-federated' as input.")
print("\nBaseline notebook:", nb_hits[0])
ns = load_definitions(nb_hits[0])
funcs = {n: f for n, f in ns.items() if inspect.isfunction(f) and f.__module__ == "baseline"}

SKIP = ("check", "save", "load", "print", "plot")

def pick(name_given, must, any_of=()):
    if name_given:
        if name_given not in funcs:
            raise NameError(f"No function named '{name_given}' in the baseline notebook.")
        return name_given, funcs[name_given]
    cands = [n for n in funcs if all(w in n.lower() for w in must)
             and (not any_of or any(w in n.lower() for w in any_of))
             and not any(w in n.lower() for w in SKIP)]
    if len(cands) == 1:
        return cands[0], funcs[cands[0]]
    related = [n for n in funcs if any(w in n.lower() for w in ("part", "label", "client", "split"))]
    raise NameError(f"Could not pick the function automatically (candidates: {cands}). Related functions "
                    f"in the baseline notebook: {related}. Write the right names in PART_FUNC_NAME and "
                    "LAB_FUNC_NAME at the top and run again.")

part_name, part_func = pick(PART_FUNC_NAME, ("partition",))
lab_name, lab_func = pick(LAB_FUNC_NAME, ("label",), ("sample", "pick", "select", "choose", "draw"))
print(f"Baseline functions: {part_name}{inspect.signature(part_func)} | {lab_name}{inspect.signature(lab_func)}")


# ---------- 4. Is the code the same as in the aggregator script? ----------
def body_without_docstring(fn_node):
    body = fn_node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    return ast.dump(ast.Module(body=body, type_ignores=[]))

t7_src = prefer(find_all("03_federated_table7_R2.py"), REFERENCE_HINT)
if t7_src:
    t7_defs = {n.name: n for n in ast.parse(open(t7_src).read()).body if isinstance(n, ast.FunctionDef)}
    for mine, ref_name in [(part_func, "partition_clients"), (lab_func, "sample_labeled")]:
        try:
            node = ast.parse(inspect.getsource(mine)).body[0]
            same = ref_name in t7_defs and body_without_docstring(node) == body_without_docstring(t7_defs[ref_name])
        except (OSError, TypeError):
            node = None
            for src in code_cells(nb_hits[0]):
                try:
                    node = next((n for n in ast.parse(src).body
                                 if isinstance(n, ast.FunctionDef) and n.name == mine.__name__), node)
                except SyntaxError:
                    pass
            same = node is not None and ref_name in t7_defs and \
                body_without_docstring(node) == body_without_docstring(t7_defs[ref_name])
        print(f"Code of {mine.__name__} identical to {ref_name} in the aggregator script: {same}")


# ---------- 5. Compare the partitions, seed by seed ----------
def call_by_name(f, table):
    args = {}
    for p in inspect.signature(f).parameters.values():
        key = p.name.lower()
        for names, value in table:
            if key in names:
                args[p.name] = value
                break
        else:
            if p.default is inspect.Parameter.empty:
                raise TypeError(f"Cannot fill the parameter '{p.name}' of {f.__name__}; send me this message.")
    return f(**args)

Y_NAMES    = {"y", "y_train", "y_tr", "ytr", "labels", "y_lab", "train_labels"}
SEED_NAMES = {"seed", "s", "rs", "random_state", "rng_seed"}

def to_clients(out, n):
    if isinstance(out, dict):
        out = [out[k] for k in sorted(out)]
    if isinstance(out, np.ndarray) and out.ndim == 1 and len(out) == n and out.dtype != object:
        return [np.where(out == k)[0] for k in range(K)]              # one client id per flow
    return [np.sort(np.asarray(p, dtype=np.int64)) for p in out]

def to_indices(out, n):
    out = np.asarray(out)
    if out.dtype == bool and len(out) == n:
        out = np.where(out)[0]
    return np.sort(out.astype(np.int64))

all_ok = True
print()
for s in SEEDS:
    ref = np.load(ref_paths[s])
    parts = to_clients(call_by_name(part_func, [
        (Y_NAMES, y_tr), ({"kind", "mode", "partition", "scheme", "how"}, "dirichlet"),
        ({"alpha", "dir_alpha", "concentration"}, ALPHA), (SEED_NAMES, s),
        ({"n_clients", "num_clients", "k", "clients", "nclients", "n_client"}, K)]), len(y_tr))
    labeled = to_indices(call_by_name(lab_func, [
        (Y_NAMES, y_tr), (SEED_NAMES, s),
        ({"per_class", "n_per_class", "lpc", "labeled_per_class", "budget", "k_per_class", "n"}, PER_CLASS)]),
        len(y_tr))
    same_clients = len(parts) == K and all(np.array_equal(parts[k], ref[f"client_{k}"]) for k in range(K))
    same_labeled = np.array_equal(labeled, ref["labeled"])
    all_ok &= same_clients and same_labeled
    print(f"seed {s}: clients identical: {same_clients} | labeled flows identical: {same_labeled}")
    if not same_clients:
        print("   client sizes, baseline:  ", [len(p) for p in parts])
        print("   client sizes, aggregator:", [len(ref[f'client_{k}']) for k in range(K)])
    if not same_labeled:
        print(f"   labeled flows: baseline {len(labeled)}, aggregator {len(ref['labeled'])}, "
              f"shared {len(np.intersect1d(labeled, ref['labeled']))}")

print("\nVERDICT:", "ALL IDENTICAL. FEAT, FLUIDS-fed and CNN-FedAvg use the same clients and labeled flows as "
      "HAFSSL and FL-AECNN. Keep the manuscript sentences (no E24 change)." if all_ok else
      "NOT IDENTICAL. Use the E24 wording (same seeds, Dirichlet parameter, label budget and test set).")
