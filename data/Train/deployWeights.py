import os
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

WORKSPACE = "ethan-li-zbawo"
PROJECT = "snake-jfgly"
VERSION = 6
RUN_DIR = ROOT / "runs" / "v7_yolo26n_aug2"
MODEL_TYPE = "yolo26"                         # 报错不支持的话改成 "yolov8"


def deploy():
    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    version = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION)
    version.deploy(
        model_type=MODEL_TYPE,
        model_path=str(RUN_DIR),
        filename="weights/best.pt",
    )
    print(f"已上传 {RUN_DIR / 'weights' / 'best.pt'} 到 {PROJECT} v{VERSION}")


if __name__ == "__main__":
    deploy()