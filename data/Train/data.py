import os
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

ROOT = Path(__file__).resolve().parents[2]

# .env read ROBOFLOW_API_KEY
load_dotenv(ROOT / ".env")

# ===============================================
WORKSPACE = "ethan-li-zbawo"
PROJECT = "snake-jfgly"
VERSION = 6          # after change to 7
FORMAT = "yolo26"
# ================================================


def download_dataset():
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("did not find ROBOFLOW_API_KEY，check .env file")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    dataset = project.version(VERSION).download(
        FORMAT,
        location=str(ROOT / "data" / "Train" / f"Snake-{VERSION}"),
    )
    print(f"data download location: {dataset.location}")


if __name__ == "__main__":
    download_dataset()