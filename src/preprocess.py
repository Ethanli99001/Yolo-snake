"""Prepare a phone photo for YOLO26 inference (letterbox to a square input)."""
import io
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

INPUT_SIZE = 640
MIN_SIDE = 32
MAX_BYTES = 20 * 1024 * 1024
PAD_COLOR = (114, 114, 114)  # YOLO default letterbox gray


def load_image(data: bytes) -> Image.Image:
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("image data must be bytes")
    if len(data) == 0:
        raise ValueError("empty file")
    if len(data) > MAX_BYTES:
        raise ValueError("file too large")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError("not a valid image") from e
    img = ImageOps.exif_transpose(img)  # phone photos store rotation in EXIF
    if min(img.size) < MIN_SIDE:
        raise ValueError("image too small")
    return img.convert("RGB")       # handles grayscale, RGBA, palette


def letterbox(img: Image.Image, size: int = INPUT_SIZE):
    """Resize keeping aspect ratio, pad to size x size. Returns (array, scale, (pad_x, pad_y))."""
    w, h = img.size
    scale = min(size / w, size / h)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), PAD_COLOR)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return arr, scale, (pad_x, pad_y)


def preprocess(data: bytes, size: int = INPUT_SIZE):
    return letterbox(load_image(data), size)
