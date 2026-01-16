from typing import List

import numpy as np
from PIL import Image


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize_for_svd(img: Image.Image, target_long: int = 1024, target_short: int = 576) -> Image.Image:
    # SVD в diffusers обычно ожидает 1024x576 или 576x1024.
    w, h = img.size
    if w >= h:
        tw, th = target_long, target_short
    else:
        tw, th = target_short, target_long
    return img.resize((tw, th), Image.LANCZOS)


def frames_to_uint8(frames: List[Image.Image]) -> np.ndarray:
    return np.stack([np.array(f.convert("RGB"), dtype=np.uint8) for f in frames], axis=0)
