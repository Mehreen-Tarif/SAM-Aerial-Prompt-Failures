r"""
Converts the 57 point-array-construction-exception rows from EXCLUDED
to IoU=0 (complete failure), consistent with the pre-registered
principle that a failure is a legitimate data point, not something to
drop. This overwrites raw_results_full_CLEANED.csv in place, so every
other analysis script (which already points at this filename) picks
up the corrected treatment automatically — no other script needs
changing.

Rationale: these 57 crashes are concentrated on complex Harbor
geometry (see 12_elongation_BB_CP_and_error_check.txt) — i.e. likely
correlated with difficulty. Excluding them would have removed hard
cases specifically from MP3/MP5/MP10's own statistics, flattering
those strategies slightly. Scoring them IoU=0 removes that exposure
entirely and costs nothing (0.006% of data, none in the headline
strategies BB/BPC/CP/CPN).
"""
import pandas as pd

RAW_FILE = r"C:\Users\Lenovo\Desktop\SAM_Revision\results\raw_results_full.csv"
CLEAN_FILE = r"C:\Users\Lenovo\Desktop\SAM_Revision\results\raw_results_full_CLEANED.csv"
BACKUP_FILE = r"C:\Users\Lenovo\Desktop\SAM_Revision\results\raw_results_full_CLEANED_backup_before_zerofill.csv"

print("Loading raw (deduplicated) results...")
raw = pd.read_csv(RAW_FILE, low_memory=False)
raw_dedup = raw.drop_duplicates(subset=["ann_id", "strategy"], keep="last")

error_mask = raw_dedup["error"].fillna("") != ""
n_errors = error_mask.sum()
print(f"Found {n_errors} error rows to convert to IoU=0")
assert n_errors == 57, f"Expected 57 error rows, found {n_errors} — investigate before proceeding"

# Zero-fill: IoU/precision/recall/f1 = 0, and CLEAR the error text —
# analysis_v1.py filters out any row with non-empty 'error', so leaving
# the text in place would silently undo this fix (this happened on the
# first attempt: the numbers didn't change because analysis_v1.py kept
# dropping these rows via that filter). Traceability is preserved via
# a separate boolean flag column instead, which nothing else filters on.
fixed = raw_dedup.copy()
fixed["was_exception"] = error_mask
fixed.loc[error_mask, ["iou", "precision", "recall", "f1"]] = 0.0
fixed.loc[error_mask, "error"] = ""

# Sanity check: no more NaN IoU values anywhere
n_remaining_nan = fixed["iou"].isna().sum()
print(f"Remaining NaN IoU after fix: {n_remaining_nan} (should be 0)")
assert n_remaining_nan == 0

# Backup the old (exclusion-based) cleaned file before overwriting
import os
if os.path.exists(CLEAN_FILE):
    os.replace(CLEAN_FILE, BACKUP_FILE)
    print(f"Old (exclusion-based) cleaned file backed up to: {BACKUP_FILE}")

fixed.to_csv(CLEAN_FILE, index=False)
print(f"\nCorrected file written to: {CLEAN_FILE}")
print(f"Total rows: {len(fixed)} (should be 933,192 — same as before, but 0 dropped, 57 now IoU=0)")
print(f"\nNEXT STEP: re-run analysis_v1.py — it already points at this filename,")
print(f"so tables 01 and 03 will regenerate with the corrected MP3/MP5/MP10")
print(f"failure rates automatically. BB/BPC/CP/CPN numbers are unaffected")
print(f"(zero errors occurred in those strategies).")
