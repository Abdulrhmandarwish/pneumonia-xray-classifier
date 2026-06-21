"""
Pneumonia Chest X-Ray Classifier — Streamlit App
=================================================
Loads a trained DenseNet121 (.keras) model and classifies uploaded chest
X-ray images as **NORMAL** or **PNEUMONIA** with a confidence score.

Preprocessing mirrors the training pipeline exactly:
  • Resize to 224 × 224
  • Convert to RGB (3 channels)
  • Apply  tf.keras.applications.densenet.preprocess_input
  • Class mapping: {NORMAL: 0, PNEUMONIA: 1}
"""

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Pneumonia X-Ray Classifier",
    page_icon="🫁",
    layout="centered",
)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
MODEL_PATH = "densenet_stage1_v2.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}

# ──────────────────────────────────────────────
# Load model (cached so it's only loaded once)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


# ──────────────────────────────────────────────
# Preprocessing — exactly matches training
# ──────────────────────────────────────────────
def preprocess_image(uploaded_image: Image.Image) -> np.ndarray:
    """Resize → RGB → array → densenet preprocess_input → add batch dim."""
    img = uploaded_image.resize(IMG_SIZE)
    img = img.convert("RGB")
    arr = np.array(img, dtype=np.float32)        # (224, 224, 3), 0-255
    arr = preprocess_input(arr)                   # DenseNet-specific scaling
    return np.expand_dims(arr, axis=0)            # (1, 224, 224, 3)


# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Overall dark background & text */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* Header styling */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2.2rem;
        background: linear-gradient(90deg, #00d2ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: -0.5rem;
    }

    /* Result cards */
    .result-card {
        border-radius: 16px;
        padding: 1.8rem 1.5rem;
        margin: 1rem 0;
        text-align: center;
    }
    .result-normal {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
        border: 1px solid #10b981;
    }
    .result-pneumonia {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        border: 1px solid #ef4444;
    }
    .result-label {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }
    .result-confidence {
        font-size: 1.1rem;
        opacity: 0.85;
    }

    /* Disclaimer box */
    .disclaimer-box {
        background: rgba(234, 179, 8, 0.08);
        border: 1px solid rgba(234, 179, 8, 0.3);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-top: 1.5rem;
        color: #fbbf24;
        font-size: 0.85rem;
        line-height: 1.5;
    }

    /* Upload area tweaks */
    [data-testid="stFileUploader"] {
        border-radius: 12px;
    }

    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🫁 Pneumonia X-Ray Classifier</h1>
    <p>Upload a chest X-ray to classify it as <strong>Normal</strong> or <strong>Pneumonia</strong></p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a chest X-ray image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG",
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-Ray", use_container_width=True)

    # Predict
    with st.spinner("Analyzing image …"):
        model = load_model()
        processed = preprocess_image(image)
        prediction = model.predict(processed, verbose=0)
        prob = float(prediction[0][0])  # sigmoid output: P(PNEUMONIA)

    predicted_class = CLASS_NAMES[1] if prob >= 0.5 else CLASS_NAMES[0]
    confidence = prob if prob >= 0.5 else 1 - prob

    # Result card
    card_class = "result-normal" if predicted_class == "NORMAL" else "result-pneumonia"
    icon = "✅" if predicted_class == "NORMAL" else "⚠️"
    st.markdown(f"""
    <div class="result-card {card_class}">
        <div class="result-label">{icon} {predicted_class}</div>
        <div class="result-confidence">Confidence: {confidence:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

    # Details expander
    with st.expander("📊 Prediction details"):
        col1, col2 = st.columns(2)
        col1.metric("P(PNEUMONIA)", f"{prob:.4f}")
        col2.metric("P(NORMAL)", f"{1 - prob:.4f}")
        st.progress(prob, text=f"Sigmoid output: {prob:.4f}")

# ──────────────────────────────────────────────
# Disclaimer (always visible)
# ──────────────────────────────────────────────
st.markdown("""
<div class="disclaimer-box">
    ⚠️ <strong>Disclaimer</strong><br>
    This application is an <strong>educational / student project</strong> and is
    <strong>NOT</strong> intended for clinical or medical diagnostic use. Predictions
    are generated by a deep-learning model trained on a limited public dataset and
    have <strong>not</strong> been validated for real-world medical decision-making.
    Always consult a qualified healthcare professional for medical advice.
</div>
""", unsafe_allow_html=True)
