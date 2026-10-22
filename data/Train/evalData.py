from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]   # data/Train/evalData.py → Yolo-snake/
# ============================
RUN_NAME = "v7_yolo26n_aug2"
DATA_VERSION = "Snake-6"
# ============================

WEIGHTS = ROOT / "runs" / RUN_NAME / "weights" / "best.pt"
DATA = ROOT / "data" / "Train" / DATA_VERSION / "data.yaml"


def evaluate():
    model = YOLO(str(WEIGHTS))
    model.val(
        data=str(DATA),
        split="test",       # use test set, not val
        conf=0.001,         # standard setting for mAP, don't change
        imgsz=640,
        device=0,
        project=str(ROOT / "runs"),
        name=f"{RUN_NAME}_test",
    )


if __name__ == "__main__":
    evaluate()