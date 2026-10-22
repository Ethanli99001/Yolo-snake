import sys
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "runs" / "v7_yolo26n_aug2" / "weights" / "best.pt"

# 类名要跟模型里的一模一样（目前是大写版本）
VENOMOUS = {
    "Agkistrodon_Piscivorus", "Agkistrodon_contortrix",
    "Crotalus_Horridus", "Crotalus_adamanteus",
    "Micrurus_Fulvius", "Sistrurus_Miliarius",
}
CONF = 0.1


def predict(image_path):
    model = YOLO(str(WEIGHTS))
    result = model(image_path, conf=CONF)[0]

    if len(result.boxes) == 0:
        print("No snake detected (or confidence level below the threshold)")
    else:
        for box in result.boxes:
            species = result.names[int(box.cls)]
            conf = float(box.conf)
            status = " venomous" if species in VENOMOUS else " non venomous"
            print(f"{species}  {conf:.0%}  →  {status}")

    result.show()   # 弹出带框的结果图
    out = ROOT / "runs" / "predict_result.jpg"
    result.save(filename=str(out))
    print(f"save loaction: {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('用法: python data/Train/predictImage.py "图片路径"')
    else:
        predict(sys.argv[1])