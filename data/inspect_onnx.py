import ast
from pathlib import Path
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]   # data/inspect_onnx.py → Yolo-snake/
ONNX_PATH = ROOT / "runs" / "v7_yolo26n_aug2" / "weights" / "best.onnx"

sess = ort.InferenceSession(str(ONNX_PATH))

for i in sess.get_inputs():
    print("INPUT ", i.name, i.shape, i.type)
for o in sess.get_outputs():
    print("OUTPUT", o.name, o.shape, o.type)

meta = sess.get_modelmeta().custom_metadata_map
print("类名:", ast.literal_eval(meta["names"]))