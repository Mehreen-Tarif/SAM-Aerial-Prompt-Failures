r"""
FULL ANALYSIS — implements STATISTICAL_PREREGISTRATION.txt exactly
=====================================================================
Run AFTER verify_and_clean_results.py has produced
raw_results_full_CLEANED.csv.

Produces, in results/analysis/:
  01_pooled_overall.csv           - headline numbers, pooled across all instances
  02_macro_averaged.csv           - per-category-then-averaged (guards against
                                     Small_Vehicle's 72% dominance)
  03_failure_rate_by_stratum.csv  - THE core "capability boundary" table:
                                     % of instances with IoU<0.5, per
                                     strategy x size stratum
  04_conditional_quality_by_stratum.csv - mean IoU GIVEN a usable mask
                                     (IoU>=0.5), per strategy x stratum
  05_pairwise_effect_sizes.csv    - the 8 pre-registered comparisons,
                                     effect-size-first (rank-biserial +
                                     bootstrap CI), Wilcoxon/Cohen's d secondary
  06_interaction_by_stratum.csv   - does the BPC-vs-CP gap differ by
                                     stratum? (the paper's central novel finding)
  07_elongation_correlation.txt   - box-vs-point advantage vs object
                                     elongation (mechanism evidence)
  08_paired_unpaired_agreement.csv - robustness check

Also prints a plain-language summary at the end indicating whether the
data supports a "failure boundary" (sharp collapse) or "crossover/
threshold" (gradual decline) framing — per the pre-registration, this
determines which version of the paper gets written, decided from data,
not assumed in advance.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from pycocotools.coco import COCO

# ── CONFIG ────────────────────────────────────────────────────────
CLEAN_FILE = r"C:\Users\Lenovo\Desktop\SAM_Revision\results\raw_results_full_CLEANED.csv"
ANN_FILE = r"D:\iSAID_Data\data\iSAID\Annotations\iSAID_val_20190823_114742.json"
OUT_DIR = r"C:\Users\Lenovo\Desktop\SAM_Revision\results\analysis"
SEED = 42
N_BOOTSTRAP = 2000
SUCCESS_THRESHOLD = 0.5  # matches original submission's IoU>=0.5 metric
STRATA_ORDER = ["small", "medium", "large"]
STRATEGIES = ["CP", "BB", "MP3", "MP5", "MP10", "BPC", "CPC", "CPN"]

# The 8 pre-registered pairwise comparisons (STATISTICAL_PREREGISTRATION.txt)
PLANNED_COMPARISONS = [
    ("BPC", "CP"), ("BPC", "CPC"), ("BPC", "BB"), ("BPC", "CPN"),
    ("BB", "CP"), ("BB", "MP10"), ("CP", "CPN"), ("BB", "CPN"),
]

os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(SEED)


def log(msg=""):
    print(msg)


# ═══════════════════════════════════════════════════════════════════
log("Loading cleaned results...")
df = pd.read_csv(CLEAN_FILE, low_memory=False)
before = len(df)
df = df[df["error"].fillna("") == ""].copy()
df["iou"] = df["iou"].astype(float)
log(f"  Loaded {before} rows, {len(df)} after dropping {before-len(df)} error rows")
log(f"  Instances: {df['ann_id'].nunique()}, Strategies: {df['strategy'].nunique()}")

# ═══════════════════════════════════════════════════════════════════
# 01 — POOLED OVERALL (headline numbers, every instance weighted equally)
# ═══════════════════════════════════════════════════════════════════
log("\n[01] Pooled overall performance...")
rows = []
for s in STRATEGIES:
    sub = df[df["strategy"] == s]
    fail_rate = (sub["iou"] < SUCCESS_THRESHOLD).mean() * 100
    cond_iou = sub.loc[sub["iou"] >= SUCCESS_THRESHOLD, "iou"].mean()
    rows.append({
        "strategy": s, "n": len(sub), "mean_iou": sub["iou"].mean(),
        "failure_rate_pct": fail_rate, "conditional_mean_iou_given_success": cond_iou,
        "mean_precision": sub["precision"].mean(), "mean_recall": sub["recall"].mean(),
        "mean_f1": sub["f1"].mean(),
    })
t01 = pd.DataFrame(rows).sort_values("mean_iou", ascending=False)
t01.to_csv(os.path.join(OUT_DIR, "01_pooled_overall.csv"), index=False)
log(t01.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════
# 02 — MACRO-AVERAGED (per-category first, then averaged equally)
# ═══════════════════════════════════════════════════════════════════
log("\n[02] Macro-averaged (guards against Small_Vehicle's 72% dominance)...")
cat_means = df.groupby(["strategy", "category"])["iou"].mean().reset_index()
t02 = cat_means.groupby("strategy")["iou"].mean().reset_index()
t02.columns = ["strategy", "macro_mean_iou"]
t02 = t02.sort_values("macro_mean_iou", ascending=False)
t02.to_csv(os.path.join(OUT_DIR, "02_macro_averaged.csv"), index=False)
log(t02.to_string(index=False))
log("\n  Comparison — pooled vs macro-averaged ranking:")
merged_check = t01[["strategy", "mean_iou"]].merge(t02, on="strategy")
log(merged_check.sort_values("mean_iou", ascending=False).to_string(index=False))

# ═══════════════════════════════════════════════════════════════════
# 03 & 04 — FAILURE RATE + CONDITIONAL QUALITY BY STRATUM (the core result)
# ═══════════════════════════════════════════════════════════════════
log("\n[03] Failure rate by strategy x stratum (THE capability-boundary table)...")


def bootstrap_ci_proportion(vals, n=N_BOOTSTRAP):
    boots = [np.mean(np.random.choice(vals, len(vals), replace=True)) for _ in range(n)]
    return np.percentile(boots, [2.5, 97.5])


rows03, rows04 = [], []
for s in STRATEGIES:
    for strat_name in STRATA_ORDER:
        sub = df[(df["strategy"] == s) & (df["stratum"] == strat_name)]
        if len(sub) == 0:
            continue
        fail_bool = (sub["iou"] < SUCCESS_THRESHOLD).values.astype(float)
        fail_rate = fail_bool.mean() * 100
        ci = bootstrap_ci_proportion(fail_bool, n=1000) * 100
        rows03.append({"strategy": s, "stratum": strat_name, "n": len(sub),
                        "failure_rate_pct": fail_rate,
                        "ci_lower": ci[0], "ci_upper": ci[1]})

        success_sub = sub[sub["iou"] >= SUCCESS_THRESHOLD]
        cond_mean = success_sub["iou"].mean() if len(success_sub) > 0 else np.nan
        rows04.append({"strategy": s, "stratum": strat_name,
                        "n_success": len(success_sub),
                        "conditional_mean_iou": cond_mean})

t03 = pd.DataFrame(rows03)
t03_pivot = t03.pivot(index="strategy", columns="stratum", values="failure_rate_pct")[STRATA_ORDER]
t03.to_csv(os.path.join(OUT_DIR, "03_failure_rate_by_stratum.csv"), index=False)
log("\n  Failure rate %% (IoU < 0.5), by strategy x stratum:")
log(t03_pivot.round(1).to_string())

t04 = pd.DataFrame(rows04)
t04_pivot = t04.pivot(index="strategy", columns="stratum", values="conditional_mean_iou")[STRATA_ORDER]
t04.to_csv(os.path.join(OUT_DIR, "04_conditional_quality_by_stratum.csv"), index=False)
log("\n[04] Conditional mean IoU (given IoU>=0.5), by strategy x stratum:")
log(t04_pivot.round(3).to_string())

# ═══════════════════════════════════════════════════════════════════
# 05 — PAIRWISE EFFECT SIZES (effect-size-first, given n=116k makes
#       p-values close to uninformative on their own)
# ═══════════════════════════════════════════════════════════════════
log("\n[05] Pairwise comparisons — effect size first (8 pre-registered comparisons)...")

pivot_iou = df.pivot_table(index="ann_id", columns="strategy", values="iou")

rows05 = []
for a, b in PLANNED_COMPARISONS:
    paired = pivot_iou[[a, b]].dropna()
    va, vb = paired[a].values, paired[b].values
    diff = va - vb
    n = len(diff)

    # matched-pairs rank-biserial (simple difference formula): proportion
    # of pairs favoring A minus proportion favoring B, ties excluded
    n_pos = (diff > 0).sum()
    n_neg = (diff < 0).sum()
    rank_biserial = (n_pos - n_neg) / (n_pos + n_neg) if (n_pos + n_neg) > 0 else 0.0

    # bootstrap CI on the raw mean difference
    boot_diffs = [np.mean(np.random.choice(diff, n, replace=True)) for _ in range(N_BOOTSTRAP)]
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])

    # secondary: Wilcoxon signed-rank + Cohen's d
    try:
        w_stat, w_p = stats.wilcoxon(va, vb)
    except ValueError:
        w_stat, w_p = np.nan, np.nan
    cohens_d = diff.mean() / (diff.std() + 1e-8)

    rows05.append({
        "comparison": f"{a} vs {b}", "n_paired": n,
        "mean_diff": diff.mean(), "ci_lower": ci_lo, "ci_upper": ci_hi,
        "rank_biserial": rank_biserial,
        "wilcoxon_p": w_p, "cohens_d": cohens_d,
    })

t05 = pd.DataFrame(rows05)
t05.to_csv(os.path.join(OUT_DIR, "05_pairwise_effect_sizes.csv"), index=False)
log(t05.round(4).to_string(index=False))

# ═══════════════════════════════════════════════════════════════════
# 06 — INTERACTION BY STRATUM (the central novel finding: does the
#       BPC-vs-CP gap actually differ by object size?)
# ═══════════════════════════════════════════════════════════════════
log("\n[06] Interaction: does the BPC-vs-CP gap differ across strata? (CORE FINDING)")
rows06 = []
for strat_name in STRATA_ORDER:
    sub_ids = df[df["stratum"] == strat_name]["ann_id"].unique()
    paired = pivot_iou.loc[pivot_iou.index.isin(sub_ids), ["BPC", "CP"]].dropna()
    diff = (paired["BPC"] - paired["CP"]).values
    n = len(diff)
    boot_diffs = [np.mean(np.random.choice(diff, n, replace=True)) for _ in range(N_BOOTSTRAP)]
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    rows06.append({"stratum": strat_name, "n": n, "mean_gap_BPC_minus_CP": diff.mean(),
                    "ci_lower": ci_lo, "ci_upper": ci_hi})
t06 = pd.DataFrame(rows06)
t06.to_csv(os.path.join(OUT_DIR, "06_interaction_by_stratum.csv"), index=False)
log(t06.round(4).to_string(index=False))

# omnibus test: does the per-instance BPC-CP difference vary across strata?
groups = [pivot_iou.loc[pivot_iou.index.isin(df[df["stratum"] == s]["ann_id"].unique()), ["BPC", "CP"]].dropna()
          for s in STRATA_ORDER]
diffs_by_stratum = [(g["BPC"] - g["CP"]).values for g in groups]
try:
    kw_stat, kw_p = stats.kruskal(*diffs_by_stratum)
    log(f"\n  Kruskal-Wallis test (does the gap distribution differ across strata): "
        f"H={kw_stat:.2f}, p={kw_p:.2e}")
except ValueError as e:
    log(f"\n  Kruskal-Wallis test could not be run: {e}")

# ═══════════════════════════════════════════════════════════════════
# 07 — ELONGATION CORRELATION (mechanism evidence, correlational)
# ═══════════════════════════════════════════════════════════════════
log("\n[07] Elongation (bbox aspect ratio) vs box-advantage correlation...")
log("  Loading annotation bboxes for aspect ratio...")
coco = COCO(ANN_FILE)
bbox_map = {}
for aid in df["ann_id"].unique():
    try:
        ann = coco.loadAnns([int(aid)])[0]
        x, y, w, h = ann["bbox"]
        aspect = max(w, h) / max(min(w, h), 1e-6)
        bbox_map[aid] = aspect
    except Exception:
        continue

aspect_df = pd.Series(bbox_map, name="aspect_ratio").rename_axis("ann_id").reset_index()

paired_bpc_cp = pivot_iou[["BPC", "CP"]].dropna().reset_index()
paired_bpc_cp["box_advantage"] = paired_bpc_cp["BPC"] - paired_bpc_cp["CP"]
merged = paired_bpc_cp.merge(aspect_df, on="ann_id")

r, p = stats.spearmanr(merged["aspect_ratio"], merged["box_advantage"])
with open(os.path.join(OUT_DIR, "07_elongation_correlation.txt"), "w") as f:
    f.write(f"Spearman correlation: box advantage (BPC - CP) vs bbox aspect ratio\n")
    f.write(f"n = {len(merged)}\n")
    f.write(f"rho = {r:.4f}, p = {p:.2e}\n")
    f.write(f"\nInterpretation: this is CORRELATIONAL evidence only, consistent with\n")
    f.write(f"(not proof of) the architectural explanation that box prompts encode\n")
    f.write(f"extent information a point cannot, which should matter more for\n")
    f.write(f"elongated objects where a point's positional information is a poorer\n")
    f.write(f"proxy for the object's true extent.\n")
log(f"  Spearman rho = {r:.4f}, p = {p:.2e}  (n={len(merged)})")

# ═══════════════════════════════════════════════════════════════════
# 08 — PAIRED VS UNPAIRED AGREEMENT (robustness check)
# ═══════════════════════════════════════════════════════════════════
log("\n[08] Paired vs unpaired agreement by stratum (robustness check)...")
rows08 = []
for strat_name in STRATA_ORDER:
    sub = df[df["stratum"] == strat_name]
    a_vals = sub[sub["strategy"] == "BPC"]["iou"].values
    b_vals = sub[sub["strategy"] == "CP"]["iou"].values
    unpaired_diff = a_vals.mean() - b_vals.mean()

    sub_ids = sub["ann_id"].unique()
    paired = pivot_iou.loc[pivot_iou.index.isin(sub_ids), ["BPC", "CP"]].dropna()
    paired_diff = (paired["BPC"] - paired["CP"]).mean()

    rows08.append({"stratum": strat_name, "unpaired_mean_diff": unpaired_diff,
                    "paired_mean_diff": paired_diff,
                    "agree": "YES" if abs(unpaired_diff - paired_diff) < 0.02 else "CHECK"})
t08 = pd.DataFrame(rows08)
t08.to_csv(os.path.join(OUT_DIR, "08_paired_unpaired_agreement.csv"), index=False)
log(t08.round(4).to_string(index=False))

# ═══════════════════════════════════════════════════════════════════
# SUMMARY — which framing does the data support?
# ═══════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("  SUMMARY — which framing does the data support?")
log("=" * 70)
cp_small_fail = t03_pivot.loc["CP", "small"]
cp_large_fail = t03_pivot.loc["CP", "large"]
bpc_small_fail = t03_pivot.loc["BPC", "small"]
bpc_large_fail = t03_pivot.loc["BPC", "large"]

log(f"\n  CP  (point) failure rate:  small={cp_small_fail:.1f}%   large={cp_large_fail:.1f}%")
log(f"  BPC (box)   failure rate:  small={bpc_small_fail:.1f}%   large={bpc_large_fail:.1f}%")

if cp_small_fail > 90:
    log("\n  --> CP failure rate on SMALL objects exceeds 90%: this supports a")
    log("      'FAILURE BOUNDARY' framing — point prompts essentially collapse")
    log("      below a size threshold.")
elif cp_small_fail - cp_large_fail > 20:
    log("\n  --> Meaningful but not near-complete gap between small/large: this")
    log("      supports a 'CROSSOVER/THRESHOLD' framing — gradual degradation,")
    log("      not a sharp collapse.")
else:
    log("\n  --> Small/large gap is modest: the scale-dependence story may be")
    log("      weaker than hypothesized. Report honestly — this would still be")
    log("      a legitimate, if less dramatic, finding.")

log(f"\nAll tables written to: {OUT_DIR}")
