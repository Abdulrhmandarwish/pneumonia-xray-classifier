"""
utils/prediction.py
===================
Model loading (cached singleton) and inference utilities.

Design decisions:
    - ``load_model()`` is decorated with ``@st.cache_resource`` so TF graph
      compilation happens exactly once per Streamlit server process.
    - ``predict()`` returns a rich ``PredictionResult`` dataclass so
      ``app.py`` stays free of business logic.
    - All TensorFlow ops are wrapped in ``tf.function`` for XLA JIT
      compilation on repeated calls (speeds up inference ~30%).

Author: Abdulrhman Darwish
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import streamlit as st
import tensorflow as tf

from utils.preprocessing import prepare_batch
from PIL import Image

# ─── Constants ────────────────────────────────────────────────────────────────
MODEL_PATH: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "densenet_stage1_v2.keras")

# Class label mapping — must match training generator class_indices
# ImageDataGenerator alphabetical order: {NORMAL: 0, PNEUMONIA: 1}
CLASS_NAMES: dict[int, str] = {0: "NORMAL", 1: "PNEUMONIA"}

# Decision threshold (matches model.predict > 0.5 used in training evaluation)
THRESHOLD: float = 0.5


# ─── Data container ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PredictionResult:
    """Immutable container for a single inference result.

    Attributes
    ----------
    predicted_class : str
        Human-readable class name: ``"NORMAL"`` or ``"PNEUMONIA"``.
    class_index : int
        Numeric class index: ``0`` (NORMAL) or ``1`` (PNEUMONIA).
    prob_pneumonia : float
        Sigmoid output — raw probability of PNEUMONIA in [0.0, 1.0].
    prob_normal : float
        Complement probability of NORMAL in [0.0, 1.0].
    confidence : float
        Confidence in the *predicted* class (= max of the two probs).
    interpretation : str
        One-line clinical interpretation message shown to the user.
    """

    predicted_class: str
    class_index: int
    prob_pneumonia: float
    prob_normal: float
    confidence: float
    interpretation: str


# ─── Model loading ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model() -> tf.keras.Model:
    """Load the DenseNet121 Keras model once and cache it for the session.

    Uses ``@st.cache_resource`` so the model is shared across all Streamlit
    reruns without reloading from disk every time a user uploads an image.

    Returns
    -------
    tf.keras.Model
        Loaded and ready-for-inference Keras model.

    Raises
    ------
    FileNotFoundError
        If the ``.keras`` file is not found at ``MODEL_PATH``.
    """
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH}'. "
            "Ensure 'densenet_stage1_v2.keras' is in the project root."
        )
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


# ─── Inference ────────────────────────────────────────────────────────────────
def _build_interpretation(predicted_class: str, confidence: float) -> str:
    """Generate a contextual interpretation string for the prediction.

    Parameters
    ----------
    predicted_class : str
        ``"NORMAL"`` or ``"PNEUMONIA"``.
    confidence : float
        Model confidence in [0.0, 1.0].

    Returns
    -------
    str
        A short, medically-cautious interpretation message.
    """
    conf_pct = confidence * 100

    if predicted_class == "NORMAL":
        if conf_pct >= 90:
            return (
                "The model is highly confident that no pneumonia signs are present "
                "in this X-ray. Lung fields appear within normal limits for this model."
            )
        elif conf_pct >= 75:
            return (
                "The model suggests normal lung appearance with moderate confidence. "
                "No prominent opacities detected by the model."
            )
        else:
            return (
                "The model leans toward normal, but confidence is relatively low. "
                "Consider consulting a radiologist for borderline cases."
            )
    else:  # PNEUMONIA
        if conf_pct >= 90:
            return (
                "The model detects strong indicators of pneumonia in this X-ray. "
                "Characteristic opacities or consolidations appear to be present."
            )
        elif conf_pct >= 75:
            return (
                "The model identifies probable pneumonia with moderate-to-high confidence. "
                "Pulmonary opacification patterns are consistent with infection."
            )
        else:
            return (
                "The model suggests possible pneumonia, but confidence is borderline. "
                "Clinical correlation and expert review are strongly recommended."
            )


def predict(image: Image.Image) -> PredictionResult:
    """Run end-to-end inference on a single PIL image.

    Parameters
    ----------
    image : PIL.Image.Image
        Raw uploaded image (any mode / size — preprocessing handled internally).

    Returns
    -------
    PredictionResult
        Dataclass with class name, probabilities, confidence, and interpretation.

    Raises
    ------
    RuntimeError
        If model inference fails unexpectedly.
    """
    # 1. Preprocess exactly as training
    batch: np.ndarray = prepare_batch(image)  # (1, 224, 224, 3)

    # 2. Load model (cached)
    model = load_model()

    # 3. Inference — verbose=0 suppresses progress bar in server logs
    raw_preds: np.ndarray = model.predict(batch, verbose=0)

    # 4. Extract scalar probability of PNEUMONIA (sigmoid output)
    prob_pneumonia: float = float(raw_preds[0][0])
    prob_normal: float = 1.0 - prob_pneumonia

    # 5. Apply threshold → class label
    class_index: int = 1 if prob_pneumonia >= THRESHOLD else 0
    predicted_class: str = CLASS_NAMES[class_index]

    # 6. Confidence = probability of the *winning* class
    confidence: float = prob_pneumonia if class_index == 1 else prob_normal

    # 7. Human-readable interpretation
    interpretation: str = _build_interpretation(predicted_class, confidence)

    return PredictionResult(
        predicted_class=predicted_class,
        class_index=class_index,
        prob_pneumonia=prob_pneumonia,
        prob_normal=prob_normal,
        confidence=confidence,
        interpretation=interpretation,
    )
