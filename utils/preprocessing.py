"""
utils/preprocessing.py
=======================
Image preprocessing pipeline that exactly mirrors the DenseNet121 training
preprocessing used in pneumonia.ipynb (Phase 6 / Phase 7).

Pipeline:
    1. Open PIL image from bytes or file-like object
    2. Convert to RGB (handles grayscale, RGBA, palette modes)
    3. Resize to (224, 224) using LANCZOS resampling
    4. Cast to float32 numpy array
    5. Apply DenseNet-specific preprocess_input (ImageNet mean subtraction)
    6. Expand dims → (1, 224, 224, 3) batch tensor

Author: Abdulrhman Darwish
"""

from __future__ import annotations

import io
import numpy as np
from PIL import Image
from tensorflow.keras.applications.densenet import preprocess_input

# ─── Constants (must match training) ─────────────────────────────────────────
IMG_SIZE: tuple[int, int] = (224, 224)
SUPPORTED_MODES: frozenset[str] = frozenset({"RGB", "L", "RGBA", "P", "LA", "1"})


# ─── Public API ───────────────────────────────────────────────────────────────

def load_image(source: "bytes | io.BytesIO | Image.Image") -> Image.Image:
    """Open an image from bytes, a BytesIO buffer, or a PIL Image.

    Parameters
    ----------
    source : bytes | io.BytesIO | PIL.Image.Image
        Raw image bytes, a BytesIO buffer returned by ``st.file_uploader``,
        or an already-opened PIL Image.

    Returns
    -------
    PIL.Image.Image
        The opened image object (mode not yet standardised).

    Raises
    ------
    ValueError
        If the bytes/buffer cannot be decoded as an image.
    """
    if isinstance(source, Image.Image):
        return source
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    try:
        return Image.open(source)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Cannot decode image: {exc}") from exc


def convert_to_rgb(image: Image.Image) -> Image.Image:
    """Ensure the image has exactly 3 RGB channels.

    The chest X-ray dataset contains a mix of grayscale (mode 'L') and
    pseudo-RGB images. DenseNet121 expects 3 channels, so we replicate the
    generator's ``color_mode='rgb'`` behaviour.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image in any PIL mode.

    Returns
    -------
    PIL.Image.Image
        Image converted to mode 'RGB'.
    """
    if image.mode == "RGB":
        return image
    return image.convert("RGB")


def resize_image(image: Image.Image, size: tuple[int, int] = IMG_SIZE) -> Image.Image:
    """Resize image to ``size`` using high-quality LANCZOS resampling.

    Parameters
    ----------
    image : PIL.Image.Image
        Input RGB image of any size.
    size : tuple[int, int]
        Target ``(width, height)`` — defaults to ``(224, 224)``.

    Returns
    -------
    PIL.Image.Image
        Resized image.
    """
    return image.resize(size, resample=Image.Resampling.LANCZOS)


def normalize_for_densenet(array: np.ndarray) -> np.ndarray:
    """Apply DenseNet-specific normalisation (ImageNet channel mean/std).

    Calls ``tf.keras.applications.densenet.preprocess_input`` which:
      - Scales pixel values from [0, 255] to [-1, 1] via mean subtraction
        of ImageNet channel means {R: 123.68, G: 116.779, B: 103.939}.
    This is NOT a simple /255 rescaling — it must match the training
    generator that used ``preprocessing_function=preprocess_input``.

    Parameters
    ----------
    array : np.ndarray
        Float32 array of shape (H, W, 3) with values in [0.0, 255.0].

    Returns
    -------
    np.ndarray
        Normalised float32 array of shape (H, W, 3).
    """
    return preprocess_input(array)


def prepare_batch(image: Image.Image) -> np.ndarray:
    """Full end-to-end preprocessing: PIL Image → model-ready batch.

    Runs the complete preprocessing pipeline:
        convert_to_rgb → resize_image → float32 cast →
        normalize_for_densenet → add batch dimension

    Parameters
    ----------
    image : PIL.Image.Image
        Raw uploaded image in any PIL mode.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(1, 224, 224, 3)`` ready for
        ``model.predict()``.
    """
    img = convert_to_rgb(image)
    img = resize_image(img)
    arr = np.array(img, dtype=np.float32)    # (224, 224, 3), range [0, 255]
    arr = normalize_for_densenet(arr)         # DenseNet ImageNet normalisation
    return np.expand_dims(arr, axis=0)        # (1, 224, 224, 3)


def validate_image(image: Image.Image) -> tuple[bool, str]:
    """Perform basic sanity checks on an uploaded image.

    Parameters
    ----------
    image : PIL.Image.Image
        Opened PIL image to validate.

    Returns
    -------
    tuple[bool, str]
        ``(is_valid, message)`` — ``True`` and an empty string when valid,
        ``False`` and a descriptive error message when invalid.
    """
    # Minimum dimension check
    w, h = image.size
    if w < 32 or h < 32:
        return False, f"Image too small ({w}×{h}). Minimum 32×32 pixels required."

    # Maximum dimension guard (prevent absurdly large uploads)
    if w > 8000 or h > 8000:
        return False, f"Image too large ({w}×{h}). Maximum 8000×8000 pixels allowed."

    # Mode check
    if image.mode not in SUPPORTED_MODES:
        return False, (
            f"Unsupported image mode '{image.mode}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_MODES))}."
        )

    return True, ""
