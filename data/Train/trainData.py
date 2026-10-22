from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]   # data/Train/trainData.py → Yolo-snake/

# ===== 每次训练前只改这里 =====
MODEL = "yolo26n.pt"                 # yolo26s.pt
DATA_VERSION = "Snake-6"
EPOCHS = 100                         # change 200
RUN_NAME = "v7_yolo26n_aug2"
# ============================

def train_model():
    model = YOLO(MODEL)
    model.train(
        data=str(ROOT / "data" / "Train" / DATA_VERSION / "data.yaml"),
        imgsz=640,
        epochs=EPOCHS,
        batch=16,
        degrees=20,      # 90 degree too much
        flipud=0.5,      # up and down flip
        seed=0,
        device=0,
        workers=4,
        amp=True,
        project=str(ROOT / "runs"),
        name=RUN_NAME,
    )

if __name__ == "__main__":
    train_model()