# SAM Prompt Strategy Comparison on iSAID

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![SAM](https://img.shields.io/badge/SAM-ViT--H-green.svg)](https://github.com/facebookresearch/segment-anything)
[![Paper](https://img.shields.io/badge/Paper-GRSL%202026-red.svg)](#-citation)

> Code, data, and pre-registered analysis plan for the paper:
> **"Segment Anything Model for Remote Sensing Instance Segmentation: A Full-Population Evaluation of Prompt Reliability on iSAID"**
>
> Mehreen Tarif, Wu Xu, Sana Abbas
> College of Computer Science and Cyber Security, Chengdu University of Technology, China
>
> Submitted to *IEEE Geoscience and Remote Sensing Letters* (GRSL), 2026

---

## ⚠️ Note on Earlier Results in This Repository

An earlier version of this repository reported results from a **size-filtered sample** (333 instances, the 25 largest per category) evaluated at a fixed 1024×1024 downsampled resolution. Those results indicated that Box-plus-Center (BPC) outperformed Bounding-Box-alone (BB).

**That ordering does not hold at population scale.** Re-evaluating across the complete iSAID validation set (116,649 instances) at native resolution reverses it: **BB outperforms BPC**. Two evaluation choices account for the difference, and the current paper quantifies both:

| Factor | Contribution to the reversal |
|:-------|:----------------------------|
| IoU computed at fixed downsampled resolution instead of native | ~2/3 |
| Size-filtered rather than representative instance sampling | ~1/3 |

The superseded numbers have been removed from this README to avoid confusion. This note is retained deliberately, since the methodological finding — that evaluation design alone can flip which prompt strategy appears superior — is a contribution of the current paper.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Findings](#-key-findings)
- [Results](#-results)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Reproducing Results](#-reproducing-results)
- [Citation](#-citation)
- [Contact](#-contact)

---

## 🎯 Overview

This repository contains the experimental framework for evaluating **eight prompt configurations** of the Segment Anything Model (SAM) on the iSAID aerial imagery benchmark, across the **complete validation population**: 116,649 instances spanning all 458 images and 15 object categories, for 933,192 SAM inferences.

No size-based or category-based subsampling is applied. Every annotation passing basic validity checks is included.

The central finding is that prompt geometry affects segmentation primarily through **reliability** (whether a usable mask is returned at all) rather than **fidelity** (its quality once one is).

---

## 🔬 Key Findings

1. **Reliability, not fidelity.** Strategies differ mainly in how often they return a usable mask (IoU ≥ 0.5), not in mask quality when they succeed. Conditional IoU on success is comparable across prompt families; failure rates are not.

2. **The box advantage is scale-dependent, and grows with object size.** The failure-rate gap between point and box prompts is *narrowest* on small objects (6–10 percentage points, where every strategy struggles) and *widest* on large objects (14–19 points, where prompt choice becomes decisive).

3. **BB outperforms BPC at population scale** (BPC − BB = −0.027 mean IoU; rank-biserial −0.50, p < 0.001). Adding a center point to a box shifts reliability only, with no meaningful quality difference (|Δ| ≤ 0.007 in every size stratum).

4. **Negative points do not rescue point prompts.** A negative-point variant (CPN) improves modestly over Center Point (0.159 vs 0.149 mIoU) but remains far short of box-level reliability.

5. **The mechanism remains open (honest null).** A shape-correlation test found no meaningful relationship between object elongation and the box advantage (Spearman ρ = −0.007 and ρ = −0.0007; both negligible despite large *n*).

---

## 📊 Results

### Per-Strategy Performance (full validation population)

n = 116,649 instances, SAM ViT-H, native resolution, seed = 42.

| Strategy | mIoU | Macro mIoU | Failure % | Conditional IoU |
|:---------|:----:|:----------:|:---------:|:---------------:|
| **BB** (Bounding Box) | **0.325** | **0.553** | **72.9** | 0.682 |
| BPC (Box + Center) | 0.299 | 0.525 | 76.4 | 0.685 |
| CPN (Center + 3 Negative) | 0.159 | 0.356 | 82.7 | 0.717 |
| CP (Center Point) | 0.149 | 0.327 | 83.5 | 0.714 |
| MP3 (Multi-Point 3) | 0.120 | 0.304 | 87.2 | 0.706 |
| MP5 (Multi-Point 5) | 0.119 | 0.310 | 87.4 | 0.702 |
| MP10 (Multi-Point 10) | 0.117 | 0.313 | 88.2 | 0.697 |
| CPC (Corners + Center) | 0.052 | 0.123 | 96.9 | 0.620 |

The ranking is identical under pooled and macro-averaged aggregation, confirming it is not an artifact of class imbalance (Small_Vehicle alone is 72% of instances).

### Failure Rate and Conditional Quality by Object Size

S/M/L = small (<32×32 px) / medium / large (≥96×96 px), COCO-style thresholds fixed before running.

| Strategy | Fail% S | Fail% M | Fail% L | Cond. IoU S | Cond. IoU M | Cond. IoU L |
|:---------|:-------:|:-------:|:-------:|:-----------:|:-----------:|:-----------:|
| BB | **78.9** | **16.0** | **12.4** | 0.653 | 0.742 | 0.815 |
| BPC | 82.3 | 19.3 | 13.5 | 0.656 | 0.735 | 0.815 |
| CP | 88.7 | 33.6 | 31.3 | 0.693 | 0.742 | 0.793 |
| CPN | 88.2 | 29.5 | 26.7 | 0.694 | 0.748 | 0.799 |
| CPC | 97.9 | 86.4 | 92.0 | 0.610 | 0.633 | 0.674 |

Note the dissociation: failure rates diverge sharply between prompt families, while conditional IoU converges.

### Failure-Rate Gap by Stratum (95% bootstrap CI)

| Comparison | Small | Medium | Large |
|:-----------|:-----:|:------:|:-----:|
| CP vs BB | 9.85 [9.65, 10.03] | 17.63 [16.84, 18.42] | 18.92 [16.64, 21.20] |
| CP vs BPC | 6.38 [6.21, 6.55] | 14.31 [13.51, 15.08] | 17.78 [15.50, 20.21] |

### Pre-Registered Pairwise Comparisons

Differences are first-named minus second-named strategy; a negative value means the second performs better. All reach p < 0.001 after Bonferroni correction (k = 8). Led by effect size, since at n = 116,649 p-values are significant for even negligible effects.

| Comparison | Δ IoU | 95% CI | Rank-biserial | Cohen's d |
|:-----------|:-----:|:------:|:-------------:|:---------:|
| BPC vs CPC | +0.247 | [0.246, 0.248] | **0.798** | 1.131 |
| BB vs MP10 | +0.208 | [0.207, 0.209] | 0.689 | 1.121 |
| BB vs CP | +0.176 | [0.175, 0.177] | 0.570 | 0.927 |
| BB vs CPN | +0.166 | [0.165, 0.167] | 0.539 | 0.887 |
| BPC vs CP | +0.150 | [0.149, 0.151] | 0.585 | 0.858 |
| BPC vs CPN | +0.139 | [0.138, 0.140] | 0.550 | 0.817 |
| BPC vs BB | −0.027 | [−0.027, −0.026] | −0.504 | −0.342 |
| CP vs CPN | −0.011 | [−0.011, −0.010] | −0.222 | −0.103 |

---

## ⚙️ Installation

### Prerequisites

- **Python 3.8+**
- **CUDA-capable GPU** (RTX 3090 used in paper; ViT-H needs ~16 GB VRAM)
- **~20 GB disk space** for the iSAID dataset

### Step 1: Clone

```bash
git clone https://github.com/Mehreen-Tarif/SAM-Prompt-Comparison.git
cd SAM-Prompt-Comparison
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download SAM checkpoint

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### Step 4: Download iSAID

Annotations and masks: **https://captain-whu.github.io/iSAID/**

Source RGB images come from DOTA v1.0: **https://captain-whu.github.io/DOTA/dataset.html**

> ⚠️ The iSAID distribution ships annotations and masks but **not** the raw validation images. Those must be downloaded separately from DOTA v1.0. All 458 validation images are required.

---

## 🚀 Quick Start

```bash
# 1. Validate the dataset before spending GPU time (no GPU needed)
python audit_dataset.py

# 2. Fast sanity check on ~30 images
python sam_experiment_v3.py --smoke

# 3. Full run over all 116,649 instances (resumable)
python sam_experiment_v3.py

# 4. Clean and verify the raw output
python verify_and_clean_results.py

# 5. Run the pre-registered analysis
python analysis_v1.py
```

**Expected runtime:** approximately 12–14 hours on an RTX 3090 for the full population.

> 💡 The experiment checkpoints at the **instance** level. If interrupted, re-run the same command and it resumes from the last completed instance — no work is lost, even mid-image.

---

## 🔄 Reproducing Results

All results use a fixed random seed (42) with CUDA in deterministic mode. An environment manifest (library versions, GPU, CUDA version) is written alongside the results on every run.

| Paper Reference | Output File |
|:----------------|:------------|
| Table I (per-strategy, pooled and by stratum) | `results/analysis/01_pooled_overall.csv`, `03_failure_rate_by_stratum.csv` |
| Table II (failure-rate gap by stratum) | `results/analysis/13_failure_rate_gap_by_stratum.csv` |
| Table III (pairwise comparisons) | `results/analysis/05_pairwise_effect_sizes.csv` |
| Macro-averaged robustness check | `results/analysis/02_macro_averaged.csv` |
| Mechanism test (honest null) | `results/analysis/12_elongation_BB_CP_and_error_check.txt` |
| Paired vs unpaired robustness | `results/analysis/08_paired_unpaired_agreement.csv` |

### Pre-Registration

The statistical analysis plan — the eight planned pairwise comparisons, the size-stratum thresholds, the choice of effect-size-first reporting, and the exclusion criteria — was fixed **before** the full-population experiment was run. See `STATISTICAL_PREREGISTRATION.txt`.

### Sampling Note

The evaluation uses the **complete** iSAID validation population. Zero instances were excluded on any validity criterion; all 116,649 annotations were valid.

Separately, 57 of 933,192 evaluations (0.006%, confined to MP3/MP5/MP10 and concentrated on complex Harbor geometry) raised a point-array construction exception in the SAM predictor. These are scored as IoU = 0 rather than dropped, consistent with treating an unsuccessful segmentation as a failure. BB, BPC, CP, and CPN are unaffected.

### Known Scope

Prompts are derived from ground-truth annotations as a deliberate **oracle-localization control**, isolating prompt geometry from detector localization error. An end-to-end evaluation with detector-generated prompts is planned as follow-up work and is **not** included here.

---

## 📝 Citation

```bibtex
@article{tarif2026sam,
  title   = {Segment Anything Model for Remote Sensing Instance Segmentation:
             A Full-Population Evaluation of Prompt Reliability on iSAID},
  author  = {Tarif, Mehreen and Xu, Wu and Abbas, Sana},
  journal = {IEEE Geoscience and Remote Sensing Letters},
  year    = {2026},
  note    = {Submitted}
}
```

---

## 🙏 Acknowledgments

- **[iSAID benchmark](https://captain-whu.github.io/iSAID/)** by the CAPTAIN laboratory, Wuhan University
- **[Segment Anything Model](https://github.com/facebookresearch/segment-anything)** by Meta AI Research
- **[DOTA dataset](https://captain-whu.github.io/DOTA/)** (which iSAID builds upon)

---

## 📧 Contact

| Role | Name | Email |
|:-----|:-----|:------|
| First Author | Mehreen Tarif | mehreentarif17@gmail.com |
| Corresponding Author | Wu Xu | wuxu2022@cdut.edu.cn |

For questions about the code or paper, please open an [Issue](https://github.com/Mehreen-Tarif/SAM-Prompt-Comparison/issues) or contact the authors directly.

---

## 📜 License

Released under the [MIT License](LICENSE) for research and reproducibility purposes.
