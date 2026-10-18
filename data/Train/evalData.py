from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]   # data/Train/evalData.py → Yolo-snake/

WEIGHTS = ROOT / "runs" / "v6_yolo26n_baseline" / "weights" / "best.pt"
DATA = ROOT / "data" / "Train" / "Snake-6" / "data.yaml"

def evaluate():
    model = YOLO(str(WEIGHTS))
    model.val(
        data=str(DATA),
        split="test",       # 用 test 集，不是 val
        conf=0.001,         # 算 mAP 的标准设置，别改成 0.45
        imgsz=640,
        device=0,
        project=str(ROOT / "runs"),
        name="v6_yolo26n_baseline_test",
    )

if __name__ == "__main__":
    evaluate()