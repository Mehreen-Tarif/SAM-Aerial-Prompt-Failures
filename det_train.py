r"""
det_train.py -- trains a YOLO11-m detector (COCO-pretrained, Ultralytics) on the iSAID training tiles from det_prepare.py.
Settings are fixed in advance (no tuning on iSAID val): 1024-px input, 36 epochs, batch 8, early stopping after 12 epochs
without improvement on the held-out TRAIN images, seed 42.  ~4-6 h on one RTX 3090.

Run from the repository folder:  python det_train.py          (resumable: python det_train.py --resume)
Output: det\yolo11m_isaid\weights\best.pt
"""
import os, sys, argparse

B = os.environ.get("SAM_REV_BASE", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(B, "scripts"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--resume", action="store_true"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    import sam_experiment_v3 as v3
    from ultralytics import YOLO
    data = os.path.join(os.environ.get("YOLO_DIR", os.path.join(v3.DATA_DIR, "yolo_isaid")), "data.yaml")
    proj, name = os.path.join(B, "det"), "yolo11m_isaid" + ("_smoke" if a.smoke else "")
    last = os.path.join(proj, name, "weights", "last.pt")
    if a.resume and os.path.exists(last):
        YOLO(last).train(resume=True)
    else:
        YOLO("yolo11m.pt").train(data=data, imgsz=1024, epochs=1 if a.smoke else 36, batch=8, patience=12, seed=42,
                             deterministic=False, workers=4, project=proj, name=name, exist_ok=True,
                             fraction=0.02 if a.smoke else 1.0, max_det=1500, plots=True, verbose=True)
    open(os.path.join(proj, name, "TRAINING_DONE"), "w").write("ok\n")


if __name__ == "__main__":          # required on Windows (data-loader workers)
    main()
