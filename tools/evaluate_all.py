# -*- coding: utf-8 -*-
"""
Recompute every metric of the paper from the saved test predictions (preds/*.npz).

Every run of every experiment saved one file  <run_id>_s<seed>.npz  with
  y_true, y_pred (and y_prob, test_idx).
This script finds all of them under experiments/, recomputes the metrics with one function, and writes
  tools/output/all_runs_recomputed.csv   one row per run: accuracy, macro precision/recall/F1,
                                          per-class recall, lowest recall, classes below 50% (collapse)
  tools/output/summary.csv               mean and sample standard deviation over seeds per method

Paired comparison of two methods (McNemar's exact test per seed, paired t-test across seeds, Holm):
  python tools/evaluate_all.py --compare "USTC_SynAttnNet_L2000" "USTC_AECNN-ft_L2000"
The two arguments are run names without the _s<seed> part (see the run_id column of summary.csv).

Usage:
  python tools/evaluate_all.py                      # recompute everything
  python tools/evaluate_all.py --root experiments   # another root folder
"""
import argparse, glob, os, re
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

SEED_RE = re.compile(r"^(?P<run>.+)_s(?P<seed>\d+)$")


def metrics(y_true, y_pred):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    n_cls = int(max(labels.max(), y_true.max()) + 1)
    labels = list(range(n_cls))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    support = cm.sum(axis=1)
    present = support > 0                                    # classes present in the test set
    rec = np.where(present, cm.diagonal() / np.maximum(support, 1), np.nan)
    return dict(acc=accuracy_score(y_true, y_pred), macro_precision=p, macro_recall=r, macro_f1=f1,
                min_class_recall=float(np.nanmin(rec)), classes_below_50=int(np.nansum(rec < 0.5)),
                collapsed=bool(np.nanmin(rec) < 0.5)), rec


def find_runs(root):
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "**", "preds", "*.npz"), recursive=True)):
        name = os.path.basename(f)[:-4]
        m = SEED_RE.match(name)
        if not m:
            continue
        d = np.load(f, allow_pickle=True)
        if "y_true" not in d.files or "y_pred" not in d.files:
            continue
        rows.append(dict(file=f, run=m["run"], seed=int(m["seed"]),
                         experiment=os.path.relpath(f, root).split(os.sep)[0]))
    return rows


def holm(pv):
    p = np.asarray(pv, float); order = np.argsort(p); adj = np.empty_like(p); run_max = 0.0
    for rank, i in enumerate(order):
        run_max = max(run_max, (len(p) - rank) * p[i]); adj[i] = min(1.0, run_max)
    return adj


def compare(runs, a, b):
    """McNemar's exact test per seed and paired t-test across seeds: run a against run b."""
    fa = {r["seed"]: r["file"] for r in runs if r["run"] == a}
    fb = {r["seed"]: r["file"] for r in runs if r["run"] == b}
    seeds = sorted(set(fa) & set(fb))
    if not seeds:
        raise SystemExit(f"No common seeds for {a} and {b}.")
    rows = []
    for s in seeds:
        x, y = np.load(fa[s]), np.load(fb[s])
        if "test_idx" in x.files and "test_idx" in y.files:
            assert np.array_equal(x["test_idx"], y["test_idx"]), f"seed {s}: different test sets"
        assert np.array_equal(x["y_true"], y["y_true"]), f"seed {s}: different test labels"
        t = x["y_true"]
        n01 = int(np.sum((x["y_pred"] == t) & (y["y_pred"] != t)))
        n10 = int(np.sum((x["y_pred"] != t) & (y["y_pred"] == t)))
        p = 1.0 if n01 + n10 == 0 else float(stats.binomtest(n01, n01 + n10, 0.5).pvalue)
        rows.append(dict(seed=s, acc_a=float(np.mean(x["y_pred"] == t)), acc_b=float(np.mean(y["y_pred"] == t)),
                         only_a_correct=n01, only_b_correct=n10, mcnemar_p=p))
    df = pd.DataFrame(rows)
    df["mcnemar_p_holm"] = holm(df.mcnemar_p)
    df["diff_pp"] = 100 * (df.acc_a - df.acc_b)
    tp = float(stats.ttest_rel(df.acc_a, df.acc_b).pvalue) if len(df) >= 2 and df.diff_pp.std() > 0 else np.nan
    print(f"\n{a}  vs  {b}")
    print(df.round(4).to_string(index=False))
    print(f"mean difference {df.diff_pp.mean():.2f} pp | {a} higher in {(df.diff_pp > 0).sum()}/{len(df)} seeds | "
          f"McNemar significant (Holm) in {(df.mcnemar_p_holm < 0.05).sum()} seeds | paired t-test p = {tp:.4f}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments")
    ap.add_argument("--out", default=os.path.join("tools", "output"))
    ap.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"))
    args = ap.parse_args()

    runs = find_runs(args.root)
    if not runs:
        raise SystemExit(f"No preds/*.npz found under {args.root}.")
    if args.compare:
        compare(runs, *args.compare)
        return

    rows = []
    for r in runs:
        d = np.load(r["file"], allow_pickle=True)
        m, rec = metrics(d["y_true"].astype(int), d["y_pred"].astype(int))
        rows.append(dict(experiment=r["experiment"], run_id=r["run"], seed=r["seed"],
                         n_test=len(d["y_true"]), **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items()},
                         **{f"recall_class_{i}": (round(float(v), 6) if v == v else np.nan) for i, v in enumerate(rec)}))
    res = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    res.to_csv(os.path.join(args.out, "all_runs_recomputed.csv"), index=False)

    g = res.groupby(["experiment", "run_id"])
    summary = g[["acc", "macro_f1", "min_class_recall"]].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary["n_seeds"] = g.size()
    summary["collapsed_runs"] = g["collapsed"].sum().astype(int)
    for c in summary.columns:
        if c.startswith(("acc", "macro_f1", "min_class")):
            summary[c] = (100 * summary[c]).round(2)              # in %
    summary = summary.reset_index()
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    pd.set_option("display.width", 220)
    print(f"{len(res)} runs recomputed from {args.root}\n")
    print(summary.to_string(index=False))
    print(f"\nWritten: {os.path.join(args.out, 'all_runs_recomputed.csv')} and summary.csv")


if __name__ == "__main__":
    main()
