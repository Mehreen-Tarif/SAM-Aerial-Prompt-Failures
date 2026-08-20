# Note on Superseded Scripts

The repository root contains scripts from an earlier stage of this project
(`SAM_FINAL_EXPERIMENT.py`, `SAM_ABLATION.py`, `real_isaid_experiment.py`,
`complete_experiment.py`, `run_*_experiment.py`, `check_*.py`, and others).

**These do not produce the results reported in the current paper.** They were
written for the earlier size-filtered protocol (333 instances, 25 largest per
category, evaluated at a fixed 1024x1024 downsampled resolution). They are
retained for transparency about the development process, not for reproduction.

## Scripts that reproduce the current paper

Run these in order:

| Order | Script | Purpose |
|:-----:|:-------|:--------|
| 1 | `audit_dataset.py` | Validate the dataset before spending GPU time (no GPU needed) |
| 2 | `sam_experiment_v3.py` | Main experiment, full population, instance-level checkpointing |
| 3 | `verify_and_clean_results.py` | Deduplicate and validate raw output |
| 4 | `finalize_zero_fill.py` | Score the 57 exception rows as IoU = 0 rather than dropping them |
| 5 | `analysis_v1.py` | Pre-registered analysis: tables, effect sizes, stratified results |

## Supporting analyses

| Script | Purpose |
|:-------|:--------|
| `failure_rate_gap_by_stratum.py` | Failure-rate gap with bootstrap CIs (paper Table II) |
| `bpc_vs_bb_detail.py` | BPC vs BB breakdown by size stratum |
| `elongation_BB_CP_precommitted.py` | Shape-correlation mechanism test (honest null) |
| `reconcile_with_original.py` | Reconstructs the size-filtered sampling criterion |
| `reconcile_random_sample.py` | Reconstructs random sampling for comparison |
| `diagnostic_side_by_side.py` | Single-instance trace, old vs new evaluation protocol |
| `migrate_checkpoint.py` | One-time checkpoint format migration (image-level to instance-level) |
| `check_missing_images.py` | Cross-references annotations against images on disk |

## Figure generation

| Script | Output |
|:-------|:-------|
| `generate_figure1_motivating_example.py` | Fig. 1, motivating example (requires GPU) |
| `generate_figure2_heatmap.py` | Per-category heatmap (supplementary) |
| `generate_figure4_failure_gap.py` | Fig. 3, failure-rate gap by object size |
| `regenerate_figure3.py` | Confidence vs IoU calibration (supplementary) |

## Pre-registration

`STATISTICAL_PREREGISTRATION.txt` records the analysis plan as fixed before the
full-population experiment was run: the eight planned pairwise comparisons, the
COCO-style size-stratum thresholds, effect-size-first reporting, and the
exclusion criteria.
