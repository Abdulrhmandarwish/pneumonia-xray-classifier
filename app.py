"""
app.py
======
Pneumonia Chest X-Ray Classifier — Main Streamlit Application
=============================================================

Professional web application for classifying chest X-ray images as
NORMAL or PNEUMONIA using a DenseNet121 deep learning model.

Features:
    • Sidebar with model information and usage instructions
    • Drag-and-drop / browse file upload for JPG, JPEG, PNG
    • Image display with metadata
    • Animated prediction result card (NORMAL = green, PNEUMONIA = red)
    • Confidence bar and probability breakdown
    • Contextual clinical interpretation
    • Mandatory medical disclaimer
    • Error handling with user-friendly messages
    • Model loaded once via @st.cache_resource

Author: Abdulrhman Darwish
"""

from __future__ import annotations

import io

import numpy as np
import streamlit as st
from PIL import Image

from utils.prediction import load_model, predict, PredictionResult
from utils.preprocessing import validate_image

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration — MUST be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pneumonia X-Ray Classifier | DenseNet121",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/",
        "Report a bug": "https://github.com/",
        "About": (
            "**Pneumonia X-Ray Classifier** — Educational deep learning project.\n\n"
            "Built with DenseNet121 + Streamlit. NOT for medical use."
        ),
    },
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — Dark gradient theme matching .streamlit/config.toml
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Import Google Font ──────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

/* ── Global reset / base ─────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}

/* ── Hero header ─────────────────────────────────────────────────────── */
.hero-header {
    text-align: center;
    padding: 2rem 0 1rem 0;
    animation: fadeInDown 0.6s ease;
}
.hero-header h1 {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00d2ff 0%, #7b2ff7 60%, #ff6b9d 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
    line-height: 1.2;
    margin-bottom: 0.4rem;
}
.hero-header .subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    font-weight: 400;
}

/* ── Section headings ────────────────────────────────────────────────── */
.section-label {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #7b2ff7;
    margin-bottom: 0.5rem;
}

/* ── Upload zone card ────────────────────────────────────────────────── */
.upload-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1.5px dashed rgba(123, 47, 247, 0.45);
    border-radius: 20px;
    padding: 2rem;
    transition: border-color 0.25s ease, background 0.25s ease;
}
.upload-card:hover {
    border-color: rgba(0, 210, 255, 0.6);
    background: rgba(0, 210, 255, 0.04);
}

/* ── Image display card ──────────────────────────────────────────────── */
.image-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 1rem;
    animation: fadeIn 0.4s ease;
}
.image-meta {
    font-size: 0.78rem;
    color: #64748b;
    text-align: center;
    margin-top: 0.4rem;
}

/* ── Result cards ────────────────────────────────────────────────────── */
.result-card {
    border-radius: 20px;
    padding: 2rem 1.8rem;
    margin: 1rem 0;
    text-align: center;
    animation: scaleIn 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
}
.result-normal {
    background: linear-gradient(135deg, #022c22 0%, #064e3b 60%, #065f46 100%);
    border: 1.5px solid #10b981;
}
.result-pneumonia {
    background: linear-gradient(135deg, #450a0a 0%, #7f1d1d 60%, #991b1b 100%);
    border: 1.5px solid #ef4444;
}
.result-icon {
    font-size: 3.2rem;
    margin-bottom: 0.4rem;
}
.result-label {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: 1px;
    margin-bottom: 0.2rem;
}
.result-confidence {
    font-size: 1.15rem;
    opacity: 0.85;
    font-weight: 500;
}
.result-normal .result-label  { color: #34d399; }
.result-pneumonia .result-label { color: #f87171; }
.result-normal .result-confidence  { color: #6ee7b7; }
.result-pneumonia .result-confidence { color: #fca5a5; }

/* ── Interpretation box ──────────────────────────────────────────────── */
.interp-box {
    background: rgba(255, 255, 255, 0.04);
    border-left: 4px solid #7b2ff7;
    border-radius: 0 14px 14px 0;
    padding: 1rem 1.2rem;
    margin-top: 0.8rem;
    color: #cbd5e1;
    font-size: 0.92rem;
    line-height: 1.65;
    animation: fadeIn 0.5s ease 0.2s both;
}

/* ── Probability badge pills ─────────────────────────────────────────── */
.prob-pills {
    display: flex;
    gap: 0.8rem;
    justify-content: center;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.prob-pill {
    padding: 0.45rem 1.1rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
}
.pill-normal    { background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; color: #34d399; }
.pill-pneumonia { background: rgba(239, 68, 68,  0.15); border: 1px solid #ef4444; color: #f87171; }

/* ── Sidebar ─────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    border-right: 1px solid rgba(123, 47, 247, 0.2);
}
.sidebar-badge {
    display: inline-block;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 0.15rem 0.15rem 0.15rem 0;
}
.badge-blue   { background: rgba(59, 130, 246, 0.2); border: 1px solid #3b82f6; color: #93c5fd; }
.badge-purple { background: rgba(139, 92, 246, 0.2); border: 1px solid #8b5cf6; color: #c4b5fd; }
.badge-green  { background: rgba(16, 185, 129, 0.2); border: 1px solid #10b981; color: #6ee7b7; }
.badge-red    { background: rgba(239, 68,  68, 0.2); border: 1px solid #ef4444; color: #fca5a5; }

/* ── Disclaimer ──────────────────────────────────────────────────────── */
.disclaimer-box {
    background: rgba(234, 179, 8, 0.07);
    border: 1px solid rgba(234, 179, 8, 0.3);
    border-radius: 14px;
    padding: 1rem 1.3rem;
    margin-top: 2rem;
    color: #fbbf24;
    font-size: 0.84rem;
    line-height: 1.6;
}

/* ── Error box ───────────────────────────────────────────────────────── */
.error-box {
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.35);
    border-radius: 14px;
    padding: 1rem 1.3rem;
    color: #f87171;
    font-size: 0.9rem;
}

/* ── Divider ─────────────────────────────────────────────────────────── */
.custom-divider {
    border: none;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    margin: 1.5rem 0;
}

/* ── Animations ──────────────────────────────────────────────────────── */
@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes scaleIn {
    from { opacity: 0; transform: scale(0.88); }
    to   { opacity: 1; transform: scale(1); }
}

/* ── Hide Streamlit chrome ───────────────────────────────────────────── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar() -> None:
    """Render the left sidebar with model details and usage instructions."""
    with st.sidebar:
        st.markdown("## 🫁 X-Ray Classifier")
        st.markdown(
            '<p style="color:#64748b;font-size:0.82rem;margin-top:-0.5rem;">'
            "DenseNet121 · Transfer Learning</p>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # ── Model Info ──────────────────────────────────────────────────────
        st.markdown("### 🤖 Model Details")
        st.markdown(
            """
<span class="sidebar-badge badge-blue">DenseNet121</span>
<span class="sidebar-badge badge-purple">ImageNet Pretrained</span>
<br>
<span class="sidebar-badge badge-green">Binary Classifier</span>
<span class="sidebar-badge badge-red">Sigmoid Output</span>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
| Property | Value |
|----------|-------|
| Input size | 224 × 224 × 3 |
| Base layers | Frozen (Stage 1) |
| Head | GAP → Dense(128) → Dropout(0.3) → Dense(1) |
| Threshold | 0.50 |
| Loss | Binary Cross-Entropy |
| Optimizer | Adam (lr=5e-5) |
"""
        )

        st.markdown("---")

        # ── Classes ─────────────────────────────────────────────────────────
        st.markdown("### 🏷️ Classes")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                '<div style="background:rgba(16,185,129,0.1);border:1px solid #10b981;'
                'border-radius:10px;padding:0.6rem;text-align:center;">'
                '<div style="font-size:1.4rem">✅</div>'
                '<div style="color:#34d399;font-weight:700;font-size:0.85rem">NORMAL</div>'
                '<div style="color:#64748b;font-size:0.75rem">Index 0</div>'
                "</div>",
                unsafe_allow_html=True,
            )
        with col_b:
            st.markdown(
                '<div style="background:rgba(239,68,68,0.1);border:1px solid #ef4444;'
                'border-radius:10px;padding:0.6rem;text-align:center;">'
                '<div style="font-size:1.4rem">⚠️</div>'
                '<div style="color:#f87171;font-weight:700;font-size:0.85rem">PNEUMONIA</div>'
                '<div style="color:#64748b;font-size:0.75rem">Index 1</div>'
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── How to use ───────────────────────────────────────────────────────
        st.markdown("### 📖 How to Use")
        st.markdown(
            """
1. **Upload** a chest X-ray image (JPG/JPEG/PNG)
2. **Wait** for the model to load (first run only)
3. **View** the prediction and confidence score
4. **Expand** the details panel for full probability breakdown
5. **Note** this is an educational tool — not for clinical use
"""
        )

        st.markdown("---")

        # ── Dataset Info ─────────────────────────────────────────────────────
        st.markdown("### 📊 Training Dataset")
        st.markdown(
            """
- **Source**: Kaggle Chest X-Ray Images
- **Total images**: ~5,863
- **Train / Val / Test**: 80% / 20% / held-out
- **Classes**: NORMAL · PNEUMONIA
- **Augmentation**: zoom, shift, horizontal flip
- **Class weights**: Balanced (imbalance correction)
"""
        )

        st.markdown("---")
        st.markdown(
            '<p style="color:#475569;font-size:0.75rem;text-align:center;">'
            "© 2024 Abdulrhman Darwish<br>Educational Project</p>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Hero Header
# ─────────────────────────────────────────────────────────────────────────────
def render_hero() -> None:
    """Render the animated page hero section."""
    st.markdown(
        """
<div class="hero-header">
    <h1>🫁 Pneumonia X-Ray Classifier</h1>
    <p class="subtitle">
        Deep learning–powered chest X-ray analysis using
        <strong style="color:#7b2ff7;">DenseNet121</strong> transfer learning
    </p>
</div>
""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Result Visualization
# ─────────────────────────────────────────────────────────────────────────────
def render_result(result: PredictionResult) -> None:
    """Render the prediction result card, metrics, and interpretation.

    Parameters
    ----------
    result : PredictionResult
        The inference result returned by ``predict()``.
    """
    is_normal = result.predicted_class == "NORMAL"
    card_class = "result-normal" if is_normal else "result-pneumonia"
    icon = "✅" if is_normal else "⚠️"
    confidence_pct = result.confidence * 100

    # ── Animated result card ─────────────────────────────────────────────────
    st.markdown(
        f"""
<div class="result-card {card_class}">
    <div class="result-icon">{icon}</div>
    <div class="result-label">{result.predicted_class}</div>
    <div class="result-confidence">Confidence: {confidence_pct:.1f}%</div>
</div>
""",
        unsafe_allow_html=True,
    )

    # ── Confidence progress bar ──────────────────────────────────────────────
    st.markdown(
        f'<p class="section-label" style="margin-top:1rem;">Confidence Gauge</p>',
        unsafe_allow_html=True,
    )
    st.progress(result.confidence)

    # ── Interpretation ───────────────────────────────────────────────────────
    st.markdown(
        f'<div class="interp-box">💡 {result.interpretation}</div>',
        unsafe_allow_html=True,
    )

    # ── Probability pills ────────────────────────────────────────────────────
    st.markdown(
        f"""
<div class="prob-pills">
    <span class="prob-pill pill-normal">✅ NORMAL: {result.prob_normal:.4f}</span>
    <span class="prob-pill pill-pneumonia">⚠️ PNEUMONIA: {result.prob_pneumonia:.4f}</span>
</div>
""",
        unsafe_allow_html=True,
    )

    # ── Expandable probability details ───────────────────────────────────────
    with st.expander("📊 Detailed Prediction Breakdown", expanded=False):
        st.markdown("##### Raw Model Output")

        col1, col2, col3 = st.columns(3)
        col1.metric(
            label="P(PNEUMONIA)",
            value=f"{result.prob_pneumonia:.4f}",
            delta=f"{(result.prob_pneumonia - 0.5):+.4f} vs threshold",
        )
        col2.metric(
            label="P(NORMAL)",
            value=f"{result.prob_normal:.4f}",
        )
        col3.metric(
            label="Predicted Class",
            value=result.predicted_class,
        )

        st.markdown("---")
        st.markdown("##### Probability Distribution")

        bar_data = {
            "Class": ["NORMAL", "PNEUMONIA"],
            "Probability": [result.prob_normal, result.prob_pneumonia],
        }
        import pandas as pd  # local import to avoid top-level overhead

        df = pd.DataFrame(bar_data)
        st.bar_chart(df.set_index("Class"), color="#7b2ff7", height=200)

        st.markdown("---")
        st.markdown(
            f"**Decision boundary:** Sigmoid output ≥ 0.50 → PNEUMONIA  \n"
            f"**Raw sigmoid output:** `{result.prob_pneumonia:.6f}`  \n"
            f"**Decision:** `{'≥' if result.prob_pneumonia >= 0.5 else '<'} 0.50` → **{result.predicted_class}**"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Disclaimer
# ─────────────────────────────────────────────────────────────────────────────
def render_disclaimer() -> None:
    """Render the mandatory medical / educational disclaimer."""
    st.markdown(
        """
<div class="disclaimer-box">
    ⚠️ <strong>Medical Disclaimer</strong><br>
    This application is an <strong>educational / student project</strong> and is
    <strong>NOT</strong> intended for clinical or medical diagnostic use.
    Predictions are generated by a deep-learning model trained on a limited public
    dataset and have <strong>not</strong> been validated for real-world medical
    decision-making. Always consult a qualified healthcare professional or licensed
    radiologist for medical advice. The developers assume no liability for any
    decisions made based on this tool.
</div>
""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    """Entry point for the Streamlit application."""

    render_sidebar()
    render_hero()

    # ── Model warm-up ────────────────────────────────────────────────────────
    # Pre-load the model in the background so first prediction is snappy
    with st.spinner("🔄 Loading DenseNet121 model…"):
        try:
            load_model()  # cached — subsequent calls are instant
            st.success("✅ Model loaded successfully!", icon="🤖")
        except FileNotFoundError as exc:
            st.markdown(
                f'<div class="error-box">❌ <strong>Model not found</strong><br>{exc}</div>',
                unsafe_allow_html=True,
            )
            st.stop()

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ── Upload section ───────────────────────────────────────────────────────
    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.markdown('<p class="section-label">Upload X-Ray Image</p>', unsafe_allow_html=True)
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            label="Drag & drop or browse a chest X-ray",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG · Max size: 200 MB",
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded_file is None:
            st.markdown(
                """
<div style="text-align:center;color:#475569;font-size:0.88rem;margin-top:1rem;">
    📂 No file selected — upload a chest X-ray above to begin
</div>
""",
                unsafe_allow_html=True,
            )

        # ── Image preview ────────────────────────────────────────────────────
        if uploaded_file is not None:
            st.markdown('<p class="section-label" style="margin-top:1rem;">Uploaded Image</p>', unsafe_allow_html=True)

            try:
                image = Image.open(uploaded_file)
            except Exception:
                st.markdown(
                    '<div class="error-box">❌ <strong>Invalid file</strong> — '
                    "Could not open the uploaded file as an image. "
                    "Please upload a valid JPG, JPEG, or PNG file.</div>",
                    unsafe_allow_html=True,
                )
                render_disclaimer()
                st.stop()

            # Validate image
            is_valid, error_msg = validate_image(image)
            if not is_valid:
                st.markdown(
                    f'<div class="error-box">❌ <strong>Invalid image</strong> — {error_msg}</div>',
                    unsafe_allow_html=True,
                )
                render_disclaimer()
                st.stop()

            # Display image
            st.markdown('<div class="image-card">', unsafe_allow_html=True)
            st.image(image, caption="", use_container_width=True)
            w, h = image.size
            st.markdown(
                f'<p class="image-meta">{uploaded_file.name} · {w}×{h}px · {image.mode} · '
                f'{uploaded_file.size / 1024:.1f} KB</p>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Prediction section ───────────────────────────────────────────────────
    with right_col:
        if uploaded_file is not None:
            st.markdown('<p class="section-label">Analysis Result</p>', unsafe_allow_html=True)

            with st.spinner("🔬 Analyzing X-ray image…"):
                try:
                    result = predict(image)
                except Exception as exc:
                    st.markdown(
                        f'<div class="error-box">❌ <strong>Prediction failed</strong><br>'
                        f"An unexpected error occurred during inference:<br><code>{exc}</code></div>",
                        unsafe_allow_html=True,
                    )
                    render_disclaimer()
                    st.stop()

            render_result(result)
        else:
            # Placeholder when no file is uploaded
            st.markdown(
                """
<div style="
    border:1.5px dashed rgba(123,47,247,0.25);
    border-radius:20px;
    padding:4rem 2rem;
    text-align:center;
    color:#475569;
    margin-top:2.4rem;
">
    <div style="font-size:3rem;margin-bottom:0.8rem;">🔬</div>
    <div style="font-size:1rem;font-weight:600;color:#64748b;">
        Awaiting Image Upload
    </div>
    <div style="font-size:0.85rem;margin-top:0.4rem;color:#374151;">
        Upload a chest X-ray to see the AI prediction here
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

    # ── Disclaimer ───────────────────────────────────────────────────────────
    render_disclaimer()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
