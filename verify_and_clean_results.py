"""
Data integrity check on the completed raw_results_full.csv, run BEFORE
any statistical analysis. Specifically checks for duplicate
(ann_id, strategy) rows that could exist from the earlier interrupted
run — an instance that had a PARTIAL set of strategy rows written
before that interrupt would not have been marked complete by
migrate_checkpoint.py (which required all 8 rows), and so would have
been fully reprocessed, potentially leaving old partial rows sitting
alongside the new complete set for the same instance.
"""
import pandas as pd

RESULTS_FILE = r"C:\Users\Lenovo\Desktop\SAM_Revision\results\raw_results_full.csv"
CLEAN_FILE = r"C:\Users\Lenovo\Desktop\SAM_Revision\results\raw_results_full_CLEANED.csv"
EXPECTED_INSTANCES = 116649
EXPECTED_STRATEGIES = 8

print("Loading results...")
df = pd.read_csv(RESULTS_FILE, low_memory=False)
print(f"  Total rows: {len(df)}")
print(f"  Expected (clean): {EXPECTED_INSTANCES * EXPECTED_STRATEGIES}")

# --- Check 1: duplicate (ann_id, strategy) pairs ---
dupe_mask = df.duplicated(subset=["ann_id", "strategy"], keep=False)
n_dupe_rows = dupe_mask.sum()
n_dupe_instances = df.loc[dupe_mask, "ann_id"].nunique()
print(f"\n[1] Duplicate (ann_id, strategy) rows: {n_dupe_rows}")
print(f"    Affecting {n_dupe_instances} distinct instances")

if n_dupe_rows > 0:
    # Keep the LAST occurrence of each (ann_id, strategy) — this
    # corresponds to the most recently written row, i.e. the complete
    # reprocessed set from after the interrupt, not the stale partial
    # rows from before it.
    df_clean = df.drop_duplicates(subset=["ann_id", "strategy"], keep="last")
    print(f"    Resolved by keeping the LAST row per (ann_id, strategy) —")
    print(f"    this keeps the reprocessed-after-interrupt version, drops the stale partial one.")
else:
    df_clean = df

print(f"\n  Rows after dedup: {len(df_clean)}")

# --- Check 2: instance/strategy completeness ---
counts = df_clean.groupby("ann_id").size()
n_complete = (counts == EXPECTED_STRATEGIES).sum()
n_incomplete = (counts != EXPECTED_STRATEGIES).sum()
unique_instances = df_clean["ann_id"].nunique()
print(f"\n[2] Unique instances in cleaned data: {unique_instances} / {EXPECTED_INSTANCES}")
print(f"    Instances with exactly {EXPECTED_STRATEGIES} strategy rows: {n_complete}")
print(f"    Instances with a different count: {n_incomplete}")
if n_incomplete > 0:
    print(f"    Distribution of row counts for incomplete instances:")
    print(counts[counts != EXPECTED_STRATEGIES].value_counts().sort_index().to_string())

# --- Check 3: NaN / error rows ---
n_nan_iou = df_clean["iou"].isna().sum()
n_error = (df_clean["error"].fillna("") != "").sum()
print(f"\n[3] Rows with NaN IoU: {n_nan_iou}")
print(f"    Rows with a non-empty error message: {n_error}")
if n_error > 0:
    print(f"    Sample error messages:")
    print(df_clean.loc[df_clean["error"].fillna("") != "", "error"].value_counts().head(10).to_string())

# --- Check 4: strategy coverage ---
print(f"\n[4] Row count per strategy (should all be ~equal):")
print(df_clean["strategy"].value_counts().to_string())

# Save cleaned version
df_clean.to_csv(CLEAN_FILE, index=False)
print(f"\nCleaned data written to: {CLEAN_FILE}")
print("Use this file (not the original) for all downstream analysis.")
