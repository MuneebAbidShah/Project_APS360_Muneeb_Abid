"""Image preprocessing: resize to a fixed height and normalize.

Augmentation (webcam-style distortions, Model B) is in src/data/augment.py.
"""
import numpy as np
import torch
from PIL import Image

# After normalization, white paper = +1.0 and black ink = -1.0.
# The collate function pads with this value so padding looks like paper.
PAD_VALUE = 1.0


def resized_width(w: int, h: int, height: int, min_width: int, max_width: int) -> int:
    """Width after scaling an image of size (w, h) to `height` px, clamped."""
    return max(min_width, min(max_width, round(w * height / h)))


def resize_keep_aspect(img: Image.Image, height: int, min_width: int, max_width: int) -> Image.Image:
    """Scale a grayscale image to `height` px, keeping its width/height ratio.

    The width is clamped to [min_width, max_width]: tiny words are stretched a
    little and very long words are squeezed a little.
    """
    new_w = resized_width(*img.size, height, min_width, max_width)
    return img.resize((new_w, height), Image.Resampling.BILINEAR)


def to_tensor_normalized(img: Image.Image) -> torch.Tensor:
    """PIL grayscale image -> float tensor of shape (1, H, W) in [-1, 1]."""
    arr = np.asarray(img, dtype=np.float32) / 255.0   # [0, 1]
    arr = (arr - 0.5) / 0.5                            # [-1, 1]
    return torch.from_numpy(arr).unsqueeze(0)          # add channel dim


class Preprocess:
    """Resize + normalize. Optionally runs an augmentation first."""

    def __init__(self, height=32, min_width=16, max_width=512, augment=None):
        self.height, self.min_width, self.max_width = height, min_width, max_width
        self.augment = augment  # callable PIL -> PIL, or None

    def resize(self, img: Image.Image) -> Image.Image:
        return resize_keep_aspect(img, self.height, self.min_width, self.max_width)

    def resized_width(self, w: int, h: int) -> int:
        return resized_width(w, h, self.height, self.min_width, self.max_width)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        if self.augment is not None:
            img = self.augment(img)
        return to_tensor_normalized(self.resize(img))


# Webcam-style augmentation (Model B, Step 6) lives in src/data/augment.py: it wraps the
# TRAINING dataset only, so validation/test (and Model A) keep using this clean Preprocess.
