from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]   # data/Train/exportOnnx.py → Yolo-snake/
RUN_NAME = "v7_yolo26n_aug2"   # 要导出的训练结果
WEIGHTS = ROOT / "runs" / RUN_NAME / "weights" / "best.pt"

def export_onnx():
    model = YOLO(str(WEIGHTS))
    path = model.export(
        format="onnx",
        imgsz=640,
        simplify=True,
        dynamic=False,
        nms=True,         # 把 NMS 打包进模型，输出直接是最终检测框
    )
    print(f"ONNX 已导出到: {path}")

    onnx_model = YOLO(path)
    test_dir = ROOT / "data" / "Train" / "Snake-6" / "test" / "images"
    sample = next(test_dir.glob("*.jpg"))
    result = onnx_model(str(sample), conf=0.5)[0]
    print(f"测试图: {sample.name}，检测到 {len(result.boxes)} 个目标")


if __name__ == "__main__":
    export_onnx()