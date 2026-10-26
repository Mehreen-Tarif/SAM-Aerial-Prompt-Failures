# Diagnosing prompt failures of the Segment Anything Model on aerial images

Code for the paper *Diagnosing prompt failures of the Segment Anything Model on aerial images: input scale, mask selection and adaptation* (M. Tarif, W. Xu, B. Tarif, S. Abbas).

The per-instance results of every configuration, the adapter weights and the detector weights are archived separately (see *Data and weights* below), because the result files are too large for GitHub.

## Environment

- Python 3.10, PyTorch 2.7.1 with CUDA 11.8 (one NVIDIA GPU)
- segment-anything 1.0 (SAM ViT-H, `sam_vit_h_4b8939.pth`; ViT-L and ViT-B for the backbone comparison)
- SAM 2, commit 2b90b9f (December 2024), `sam2_hiera_large.pt`
- pycocotools 2.0.11, numpy, pandas, scipy, opencv-python, matplotlib, Pillow
- ultralytics 8.4.182 (detector experiment only)

## Data

- iSAID validation split (images from DOTA v1.0, annotations `iSAID_val_20190823_114742.json`); the training split for adaptation and for the detector.
- NWPU VHR-10, positive image set, with the instance masks of Su et al. (2019) in COCO format.

Paths are set with environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `ISAID_ROOT` | folder containing `data\iSAID\...` and the SAM checkpoint | `D:\iSAID_Data` |
| `SAM_REV_BASE` | working folder (results are written below it) | folder of the scripts |
| `SAM2_REPO`, `SAM2_CKPT` | SAM 2 repository and checkpoint | `..\sam2` |
| `EVAL_ANN`, `EVAL_IMG_DIR` | external dataset (NWPU VHR-10) | unset (iSAID) |
| `DET_BOXES` | detector boxes for the `DET` prompt | unset |

## Pipeline

| Step | Script | Paper section |
|---|---|---|
| Prompt construction, metrics, data loading | `scripts/sam_experiment_v3.py` (library) | Methods: prompt strategies, metrics |
| Evaluation of a model / protocol with all prompt types | `sam_eval_v4.py` | Results: prompts, scale, failure analysis, confidence, backbones |
| Tables, paired statistics, cluster bootstrap | `analysis_v4.py`, `audit_extra.py`, `audit_extra2.py` | Results: prompts, scale, failure analysis, confidence, backbones |
| LoRA adapters (box-only; mixed-prompt with IoU head, ablations, seeds) | `lora_sam.py`, `lora_train_v3.py`, `lora_train_v4.py` | Methods and Results: adaptation |
| Adapter comparisons, seeds, NWPU statistics | `paired_extra.py`, `analyze_external.py`, `harbor_check.py` | Results: adaptation, NWPU VHR-10 |
| Shifted-prompt adapter | `lora_train_v4_jit.py`, `run_jitter.ps1`, `jitter_report.py`, `seeds_report.py` | Results: imperfect prompts |
| Detector boxes (YOLO11-m) | `det_prepare.py`, `det_train.py`, `det_boxes.py`, `det_report.py`, `run_extra.ps1` | Results: boxes from a detector |
| Figures | `make_figures.py`, `make_adapt_figures.py`, `make_qualitative.py`, `make_examples_v4.py`, `make_overview_v2.py` | Figs. 1-8 |

Examples:

```
# frozen SAM ViT-H, 1024-pixel tiles, all nine prompt types
python sam_eval_v4.py --model sam1 --mode tile1024

# mixed-prompt adapter (LoRA + IoU head), seed 42, then its evaluation
python lora_train_v4.py --mode tile1024 --out-name lora_v4
python sam_eval_v4.py --model sam1 --mode tile1024 --lora lora_v4\tile1024\best.pt --tag v4

# shifted prompts at +-5, 10, 20 pixels
python sam_eval_v4.py --model sam1 --mode tile1024 --strategies BBN5,BBN10,BBN20,CPJ5,CPJ10,CPJ20 --out results_v4_noise

# all tables
python analysis_v4.py
```

Every script states its inputs, outputs and approximate run time at the top. Evaluation and training scripts are resumable.

## Data and weights

Per-instance results (one CSV row per instance, prompt type and configuration, with the IoU and predicted IoU of all three candidate masks), adapter weights and detector weights: Zenodo, [doi:10.5281/zenodo.23540023](https://doi.org/10.5281/zenodo.23540023).

## Citation

Please cite the paper if you use this code. The datasets and models used here are subject to their own licences (iSAID, DOTA, NWPU VHR-10, SAM, SAM 2, Ultralytics).
