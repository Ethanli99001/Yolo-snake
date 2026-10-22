import ast
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
ONNX_PATH = ROOT / "runs" / "v7_yolo26n_aug2" / "weights" / "best.onnx"
CONF = 0.5
SIZE = 640

# 类名要跟模型里的一模一样（目前是大写版本）
VENOMOUS = {
    "Agkistrodon_Piscivorus", "Agkistrodon_contortrix",
    "Crotalus_Horridus", "Crotalus_adamanteus",
    "Micrurus_Fulvius", "Sistrurus_Miliarius",
}


def preprocess(img):
    """跟手机端完全一样：拉伸到 640x640 → RGB → /255 → NCHW"""
    resized = img.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0   # HWC, 0~1
    arr = arr.transpose(2, 0, 1)                          # HWC → CHW
    return arr[np.newaxis, ...]                           # → [1, 3, 640, 640]


def predict(image_path):
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    names = ast.literal_eval(sess.get_modelmeta().custom_metadata_map["names"])

    img = Image.open(image_path)
    W, H = img.size
    tensor = preprocess(img)

    out = sess.run(None, {"images": tensor})[0][0]        # [300, 6]
    dets = out[out[:, 4] >= CONF]                         # 按置信度过滤

    draw = ImageDraw.Draw(img := img.convert("RGB"))
    if len(dets) == 0:
        print("没有检测到蛇（或置信度低于阈值）")

    for x1, y1, x2, y2, conf, cls in dets:
        # 640x640 坐标 → 原图坐标
        sx, sy = W / SIZE, H / SIZE
        x1, x2 = x1 * sx, x2 * sx
        y1, y2 = y1 * sy, y2 * sy

        species = names[int(cls)]
        status = "VENOMOUS" if species in VENOMOUS else "non-venomous"
        print(f"{species}  {conf:.0%}  →  {' venomous' if species in VENOMOUS else 'non venomous'}")

        color = "red" if species in VENOMOUS else "lime"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=max(3, W // 200))
        draw.text((x1 + 5, max(0, y1 - 15)), f"{species} {conf:.0%} {status}", fill=color)

    out_path = ROOT / "runs" / "predict_onnx_result.jpg"
    img.save(out_path)
    img.show()
    print(f"save: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('用法: python data/Train/predictOnnx.py "图片路径"')
    else:
        predict(sys.argv[1])