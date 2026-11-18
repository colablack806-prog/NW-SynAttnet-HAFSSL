# -*- coding: utf-8 -*-
"""
Table V (federated attention ablation, ISCX-VPN2016): every number of the table and of its text,
computed from the saved outputs of federated_ablation_R2.py (results/, preds/, reports/).

Usage: set RUN_DIRS to the output folder(s) that hold the 12 runs (4 forms x seeds 0-2) and run.
       If the runs are in several folders (e.g. last night's 9 runs and the 3 MHA->SE runs), list them all.
No training; a CPU is enough. Prints: sources and TensorFlow versions, pairing checks, per-seed results,
the LaTeX rows of Table V, McNemar per seed, paired t-tests, the Fisher test and the lost classes.
"""
import os, glob
import numpy as np
import pandas as pd
from scipy import stats

RUN_DIRS = ["outputs/FED_ISCX_ablation"]          # folder(s) with results/all_runs.csv, preds/, reports/
LABELS   = ['chat', 'email', 'file', 'streaming', 'voip']
SHOW     = ['Chat', 'Email', 'File', 'Stream.', 'VoIP']
FORMS    = [("HAFSSL-MHAonly", "MHA-only"), ("HAFSSL-SEonly", "SE-only"),
            ("HAFSSL-SE>MHA", "SE$\\rightarrow$MHA"), ("HAFSSL", "MHA$\\rightarrow$SE (HAFSSL)")]
REF      = "HAFSSL"                                # MHA->SE, the reference of the tests
SEEDS    = [0, 1, 2]
COLLAPSE = 0.50                                    # a run is collapsed if any class recall < 50%
LAST_N   = 10                                      # "Low rds.": last 10 rounds with val acc < 90%


def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0.0
    for i, k in enumerate(order):
        run = max(run, (m - i) * p[k]); adj[k] = min(1.0, run)
    return adj


def mcnemar_exact(y, a, b):
    ca, cb = a == y, b == y
    n01, n10 = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))       # only a correct / only b correct
    p = stats.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return n01, n10, float(p)


# ---------------------------------------------------------------- load
rows = []
for d in RUN_DIRS:
    for f in glob.glob(os.path.join(d, "**", "results", "all_runs.csv"), recursive=True):
        r = pd.read_csv(f); r["folder"] = os.path.dirname(os.path.dirname(f)); rows.append(r)
if not rows:
    raise SystemExit(f"No results/all_runs.csv under {RUN_DIRS}")
res = pd.concat(rows, ignore_index=True)
res = res[(res.dataset == "ISCX") & (res.aggregator == "FedAvg") & (res.alpha == 0.5)
          & (res.labels_per_class == 1000) & (res.method.isin([m for m, _ in FORMS]))]
res = res.drop_duplicates("run_id", keep="last").set_index(["method", "seed"]).sort_index()

print("=" * 90 + "\nSOURCES")
for m, name in FORMS:
    have = sorted(res.loc[m].index) if m in res.index.get_level_values(0) else []
    tfv = sorted(set(res.loc[m].tf_version)) if have else []
    print(f"  {name:28s} seeds {have}  tf {tfv}")
    if have != SEEDS:
        raise SystemExit(f"  -> {name}: need seeds {SEEDS}. Add the folder with these runs to RUN_DIRS.")
versions = set(res.tf_version)
print("  TensorFlow versions:", sorted(versions),
      "OK (one version)" if len(versions) == 1 else "<-- MIXED: Table V should use runs of one version")


def load_run(m, s):
    r = res.loc[(m, s)]
    p = np.load(os.path.join(r.folder, "preds", f"{r.run_id}.npz"))
    rounds = pd.read_csv(os.path.join(r.folder, "reports", f"{r.run_id}_rounds.csv"))
    cc = pd.read_csv(os.path.join(r.folder, "reports", f"{r.run_id}_client_classes.csv"), index_col=0)
    return r, p["y_true"], p["y_pred"], p["test_idx"], rounds, cc


# ---------------------------------------------------------------- pairing checks + per-run numbers
print("=" * 90 + "\nPAIRING CHECKS (same test flows and same clients/labeled flows in each seed)")
run = {}
for s in SEEDS:
    ref = None
    for m, name in FORMS:
        r, y, yp, idx, rounds, cc = load_run(m, s)
        rec = np.array([np.mean(yp[y == c] == c) for c in range(len(LABELS))])
        conf = pd.crosstab(pd.Series(y, name="true"), pd.Series(yp, name="pred"), normalize="index")
        run[(m, s)] = dict(acc=r.acc, f1=r.macro_f1, rec=rec, y=y, yp=yp, conf=conf,
                           low=int((rounds.val_acc.tail(LAST_N) < 0.90).sum()), r90=r.rounds_to_90)
        cols = [c for c in cc.columns if c != "Unnamed: 0"]
        if ref is None:
            ref = (idx, y, cc[cols].values)
        else:
            same = np.array_equal(idx, ref[0]) and np.array_equal(y, ref[1]) and np.array_equal(cc[cols].values, ref[2])
            if not same:
                raise SystemExit(f"  seed {s}: {name} does not share test flows / clients with {FORMS[0][1]}")
    print(f"  seed {s}: all four forms share test flows, client partition and labeled flows: OK")

print("=" * 90 + "\nPER SEED (accuracy %, macro-F1 %, lowest recall %, lost classes with main confusion)")
for m, name in FORMS:
    for s in SEEDS:
        q = run[(m, s)]
        lost = []
        for c in np.where(q["rec"] < COLLAPSE)[0]:
            row = q["conf"].loc[c].drop(c, errors="ignore")
            lost.append(f"{SHOW[c]} {100*q['rec'][c]:.1f}% ({100*row.max():.0f}% -> {SHOW[int(row.idxmax())]})")
        print(f"  {name:28s} s{s}: acc {100*q['acc']:.2f} | F1 {100*q['f1']:.2f} | min rec {100*q['rec'].min():.1f}"
              f" | low rds {q['low']}/{LAST_N} | R90 {q['r90']} | lost: {', '.join(lost) or '-'}")

# ---------------------------------------------------------------- tests vs MHA->SE
print("=" * 90 + "\nMcNEMAR per seed, MHA->SE vs each form (Holm over the 9 seed-form pairs)")
mc = []
for m, name in FORMS:
    if m == REF:
        continue
    for s in SEEDS:
        a, b = run[(REF, s)], run[(m, s)]
        n01, n10, p = mcnemar_exact(a["y"], a["yp"], b["yp"])
        mc.append(dict(form=name, method=m, seed=s, only_MHASE=n01, only_form=n10, p=p,
                       diff=100 * (a["acc"] - b["acc"])))
mc = pd.DataFrame(mc); mc["p_holm"] = holm(mc.p)
print(mc.round(4).to_string(index=False))

print("\nPAIRED t-TEST across seeds, MHA->SE vs each form (Holm over the 3 forms)")
tt = []
for m, name in FORMS:
    if m == REF:
        continue
    a = np.array([run[(REF, s)]["acc"] for s in SEEDS]); b = np.array([run[(m, s)]["acc"] for s in SEEDS])
    d = 100 * (a - b)
    tt.append(dict(form=name, method=m, mean_diff_pp=d.mean(), per_seed=np.round(d, 2).tolist(),
                   p=float(stats.ttest_rel(a, b).pvalue) if np.std(d) > 0 else np.nan))
tt = pd.DataFrame(tt); tt["p_holm"] = holm(tt.p.fillna(1.0))
print(tt.round(4).to_string(index=False))

col = {m: sum(run[(m, s)]["rec"].min() < COLLAPSE for s in SEEDS) for m, _ in FORMS}
mha_first = col[REF] + col["HAFSSL-MHAonly"]; se_first = col["HAFSSL-SE>MHA"] + col["HAFSSL-SEonly"]
fp = stats.fisher_exact([[mha_first, 6 - mha_first], [se_first, 6 - se_first]]).pvalue
print(f"\nFISHER: collapsed runs MHA-first {mha_first}/6 vs SE-first {se_first}/6, p = {fp:.3f}")

# ---------------------------------------------------------------- LaTeX rows of Table V
print("=" * 90 + "\nLATEX ROWS OF TABLE V (paste between the header \\hline and the closing \\hline)")
summ = {}
for m, name in FORMS:
    acc = 100 * np.array([run[(m, s)]["acc"] for s in SEEDS]); f1 = 100 * np.array([run[(m, s)]["f1"] for s in SEEDS])
    rec = 100 * np.mean([run[(m, s)]["rec"] for s in SEEDS], axis=0)
    r90 = [run[(m, s)]["r90"] for s in SEEDS]; ok = [x for x in r90 if x > 0]
    summ[m] = dict(acc=acc.mean(), acc_sd=acc.std(ddof=1), f1=f1.mean(), f1_sd=f1.std(ddof=1), rec=rec,
                   col=col[m], low=sum(run[(m, s)]["low"] for s in SEEDS),
                   r90=(np.mean(ok) if ok else np.nan), r90_n=len(ok))
best = dict(acc=max(v["acc"] for v in summ.values()), f1=max(v["f1"] for v in summ.values()),
            rec=np.max([v["rec"] for v in summ.values()], axis=0), col=min(v["col"] for v in summ.values()),
            low=min(v["low"] for v in summ.values()), r90=np.nanmin([v["r90"] for v in summ.values()]))
B = lambda txt, cond: f"\\textbf{{{txt}}}" if cond else txt
for m, name in FORMS:
    v = summ[m]
    cells = [B(f"{v['acc']:.2f}", np.isclose(v["acc"], best["acc"])) + f" $\\pm$ {v['acc_sd']:.2f}",
             B(f"{v['f1']:.2f}", np.isclose(v["f1"], best["f1"])) + f" $\\pm$ {v['f1_sd']:.2f}"]
    cells += [B(f"{x:.2f}", np.isclose(x, best["rec"][i])) for i, x in enumerate(v["rec"])]
    cells += [B(f"{v['col']}/3", v["col"] == best["col"]), B(f"{v['low']}/30", v["low"] == best["low"])]
    r90 = "---" if np.isnan(v["r90"]) else f"{v['r90']:.1f}" + ("" if v["r90_n"] == 3 else f" ({v['r90_n']}/3)")
    cells.append(B(r90, not np.isnan(v["r90"]) and np.isclose(v["r90"], best["r90"])))
    if m == REF:
        cells += ["---", "---"]
    else:
        g = mc[mc.method == m]
        cells.append(f"{int(((g.p_holm < 0.05) & (g['diff'] > 0)).sum())}/{int(((g.p_holm < 0.05) & (g['diff'] < 0)).sum())}")
        cells.append(f"{float(tt[tt.method == m].p_holm.iloc[0]):.3f}")
    print(f"{name} & " + " & ".join(cells) + " \\\\")
print("\nColumns: Acc. | F1 | Chat | Email | File | Stream. | VoIP | Collapsed | Low rds. | R90 | McNemar (MHA->SE/form) | p")
