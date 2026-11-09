import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import io
import numpy as np
import pytest
from PIL import Image
from preprocess import preprocess, load_image, letterbox, INPUT_SIZE, PAD_COLOR


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

def make_bytes(w, h, mode="RGB", fmt="JPEG", color=(200, 50, 50), exif_orientation=None):
    img = Image.new(mode, (w, h), color if mode == "RGB" else None)
    buf = io.BytesIO()
    kwargs = {}
    if exif_orientation:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
        kwargs["exif"] = exif
    img.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


# --- Output size: always 640x640x3, values in [0, 1] ---
@pytest.mark.parametrize("w, h", [
    (640, 640),     # already correct
    (4032, 3024),   # typical phone photo (landscape)
    (3024, 4032),   # portrait
    (1920, 1080),   # 16:9
    (100, 80),      # small -> upscale
    (5000, 500),    # extreme aspect ratio
])
def test_output_shape(w, h):
    arr, _, _ = preprocess(make_bytes(w, h))
    assert arr.shape == (INPUT_SIZE, INPUT_SIZE, 3)
    assert arr.dtype == np.float32
    assert 0.0 <= arr.min() and arr.max() <= 1.0


# --- Aspect ratio preserved (no stretching a snake) ---
def test_aspect_ratio_preserved_landscape():
    arr, scale, (pad_x, pad_y) = preprocess(make_bytes(1280, 640))
    assert scale == pytest.approx(0.5)
    assert pad_x == 0 and pad_y == 160          # 640x320 content, padded top/bottom

def test_padding_uses_letterbox_gray():
    arr, _, _ = preprocess(make_bytes(1280, 640))
    assert np.allclose(arr[0, 0] * 255, PAD_COLOR, atol=1)       # top pad
    assert not np.allclose(arr[320, 320] * 255, PAD_COLOR, atol=1)  # image area


# --- Color modes ---
@pytest.mark.parametrize("mode, fmt", [("L", "PNG"), ("RGBA", "PNG"), ("P", "PNG")])
def test_color_modes_converted_to_rgb(mode, fmt):
    arr, _, _ = preprocess(make_bytes(300, 300, mode=mode, fmt=fmt))
    assert arr.shape[2] == 3


# --- File formats ---
@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "BMP", "WEBP"])
def test_supported_formats(fmt):
    arr, _, _ = preprocess(make_bytes(300, 200, fmt=fmt))
    assert arr.shape == (INPUT_SIZE, INPUT_SIZE, 3)


# --- EXIF rotation (phones save portrait photos sideways + a rotation tag) ---
def test_exif_rotation_applied():
    data = make_bytes(400, 200, exif_orientation=6)  # 6 = rotate 90 degrees
    img = load_image(data)
    assert img.size == (200, 400)


# --- Invalid input ---
def test_empty_file():
    with pytest.raises(ValueError):
        preprocess(b"")

def test_corrupted_file():
    with pytest.raises(ValueError):
        preprocess(b"this is not an image")

def test_truncated_jpeg():
    data = make_bytes(500, 500)
    with pytest.raises(ValueError):
        preprocess(data[: len(data) // 3])

def test_too_small_image():
    with pytest.raises(ValueError):
        preprocess(make_bytes(10, 10))

def test_too_large_file(monkeypatch):
    import preprocess as p
    monkeypatch.setattr(p, "MAX_BYTES", 100)
    with pytest.raises(ValueError):
        p.preprocess(make_bytes(300, 300))

@pytest.mark.parametrize("bad", [None, "photo.jpg", 123, np.zeros((10, 10))])
def test_wrong_input_type(bad):
    with pytest.raises(TypeError):
        preprocess(bad)
