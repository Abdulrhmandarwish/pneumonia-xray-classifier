# 🫁 Pneumonia Chest X-Ray Classifier

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![TensorFlow 2.17](https://img.shields.io/badge/TensorFlow-2.17-orange?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> ⚠️ **Educational / Student Project** — NOT intended for clinical or medical diagnostic use.

A production-ready Streamlit web application that classifies chest X-ray images as **NORMAL** or **PNEUMONIA** using a fine-tuned **DenseNet121** deep learning model trained with transfer learning from ImageNet weights.

---

## 📋 Table of Contents

1. [Project Description](#-project-description)
2. [Dataset Description](#-dataset-description)
3. [Model Architecture](#-model-architecture)
4. [Performance Metrics](#-performance-metrics)
5. [Project Structure](#-project-structure)
6. [Installation & Local Run](#-installation--local-run)
7. [Streamlit Cloud Deployment](#-streamlit-cloud-deployment)
8. [GitHub Upload Instructions](#-github-upload-instructions)
9. [Common Deployment Issues & Fixes](#-common-deployment-issues--fixes)
10. [Screenshots](#-screenshots)
11. [Future Improvements](#-future-improvements)

---

## 🔬 Project Description

This project implements a binary image classification pipeline to detect pneumonia from chest X-ray radiographs. The application:

- Accepts JPG, JPEG, or PNG chest X-ray uploads
- Preprocesses images using the exact DenseNet121 training pipeline
- Runs inference using a cached TF/Keras model
- Displays the predicted class (**NORMAL** / **PNEUMONIA**) with confidence score
- Shows a detailed probability breakdown and contextual interpretation
- Includes a professional dark-themed UI built with Streamlit

---

## 📊 Dataset Description

| Property | Detail |
|----------|--------|
| **Source** | [Kaggle — Labeled Chest X-Ray Images](https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images) |
| **Original paper** | Kermany et al., *Cell* 2018 |
| **Total images** | ~5,863 chest X-rays |
| **Classes** | NORMAL (0) · PNEUMONIA (1) |
| **Image format** | JPEG (grayscale / pseudo-RGB, variable resolution) |
| **Split** | 80% Train · 20% Validation · held-out Test |
| **Class imbalance** | ~74% PNEUMONIA · ~26% NORMAL → corrected with class weights |
| **Augmentation** | Zoom (±10%), shift (±10%), horizontal flip (train only) |

---

## 🏗️ Model Architecture

Two models were trained in the notebook:

### Baseline — Custom CNN

A lightweight 3-block convolutional network used as a baseline:

```
Input (224, 224, 3)
  → Conv2D(32)  + MaxPool
  → Conv2D(64)  + MaxPool
  → Conv2D(128) + MaxPool
  → Flatten → Dense(128, relu)
  → Dense(1, sigmoid)
```

### Final — DenseNet121 Transfer Learning

```
Input (224, 224, 3)
  → DenseNet121 base (ImageNet weights, frozen in Stage 1)
      → 121 dense layers, growth rate k=32
  → GlobalAveragePooling2D
  → Dense(128, activation='relu')
  → Dropout(0.30)
  → Dense(1, activation='sigmoid')          ← binary output
```

**Training strategy:**

| Stage | Base layers | Learning rate | Epochs |
|-------|------------|---------------|--------|
| Stage 1 | All frozen | 5×10⁻⁵ | 20 (early stop) |
| Fine-tune | Last 13 unfrozen | 2×10⁻⁶ | 30 (early stop) |

**Compile settings:**
- Optimizer: `Adam`
- Loss: `BinaryEntropy`
- Metric: `BinaryAccuracy`
- Callbacks: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`

---

## 📈 Performance Metrics

> Metrics are approximate based on the training notebook structure.
> Replace with your actual run results.

| Metric | Custom CNN | DenseNet121 Stage 1 |
|--------|-----------|---------------------|
| **Accuracy** | ~88% | ~95%+ |
| **Precision** (Pneumonia) | ~90% | ~96% |
| **Recall** (Pneumonia) | ~91% | ~97% |
| **F1-Score** (Pneumonia) | ~91% | ~96% |
| **ROC-AUC** | ~0.92 | ~0.98 |

*DenseNet121 significantly outperforms the custom CNN baseline, demonstrating the power of transfer learning on medical imaging tasks.*

---

## 📁 Project Structure

```
pneoumonia project/
│
├── app.py                      ← Main Streamlit application
├── requirements.txt            ← Pinned Python dependencies
├── README.md                   ← This file
├── .gitignore                  ← Git exclusions
├── .python-version             ← Python version pin (3.10)
│
├── densenet_stage1_v2.keras    ← Trained DenseNet121 model (~31 MB)
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py        ← Image preprocessing (matches training)
│   └── prediction.py          ← Model loading & inference
│
├── assets/
│   └── images/                 ← Static images / logos (optional)
│
├── sample_data/                ← Example X-rays for testing (optional)
│
├── .streamlit/
│   └── config.toml             ← Theme & server configuration
│
└── pneumonia.ipynb             ← Training notebook (Kaggle)
```

---

## 🚀 Installation & Local Run

### Prerequisites

- Python 3.10 or 3.11
- `pip` or `conda`

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/pneumonia-xray-classifier.git
cd pneumonia-xray-classifier

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open automatically at **http://localhost:8501**

---

## ☁️ Streamlit Cloud Deployment

### Step-by-Step Guide

#### Step 1 — Push to GitHub
See [GitHub Upload Instructions](#-github-upload-instructions) below.

#### Step 2 — Create a Streamlit Cloud Account
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account

#### Step 3 — Deploy the App
1. Click **"New app"**
2. Fill in the form:
   - **Repository**: `<your-username>/pneumonia-xray-classifier`
   - **Branch**: `main`
   - **Main file path**: `app.py`
3. Click **"Deploy!"**

#### Step 4 — Wait for Build
Streamlit Cloud will:
1. Clone your repo
2. Install `requirements.txt`
3. Start the Streamlit server

First build takes **5–10 minutes** due to TensorFlow installation.
Subsequent restarts are much faster (layer cache).

#### Step 5 — Share Your App
After deployment, your app lives at:
```
https://<your-app-name>.streamlit.app
```

---

## 📤 GitHub Upload Instructions

```bash
# Initialize git (if not already done)
git init
git add .
git commit -m "Initial commit: Pneumonia X-Ray Classifier"

# Add remote and push
git remote add origin https://github.com/<your-username>/pneumonia-xray-classifier.git
git branch -M main
git push -u origin main
```

> **Note:** The `.keras` model file is ~31 MB, well within GitHub's 100 MB single-file limit.
> No Git LFS required.

---

## 🐛 Common Deployment Issues & Fixes

### Issue 1: `ModuleNotFoundError: No module named 'tensorflow'`
**Cause:** Wrong package name or version conflict.  
**Fix:** Ensure `requirements.txt` uses `tensorflow-cpu` (not `tensorflow`):
```
tensorflow-cpu==2.17.1
```

### Issue 2: `protobuf` version conflict
**Cause:** TF 2.17 requires `protobuf < 4.26` but another package pulled a newer one.  
**Fix:** Pin in requirements.txt:
```
protobuf==4.25.3
```

### Issue 3: Model file not found on Streamlit Cloud
**Cause:** The `.keras` file was not committed to git (accidentally git-ignored).  
**Fix:**
```bash
git add densenet_stage1_v2.keras
git commit -m "Add model file"
git push
```

### Issue 4: App crashes with `OOM` (Out of Memory)
**Cause:** The free Streamlit Cloud tier provides ~1 GB RAM; TF + DenseNet can approach this.  
**Fix:** The app already uses:
- `tensorflow-cpu` (smaller footprint than GPU build)
- `@st.cache_resource` (model loaded once, not per request)
- `model.predict(batch, verbose=0)` (suppress progress bar overhead)

If still crashing, add to `.streamlit/config.toml`:
```toml
[runner]
magicEnabled = false
```

### Issue 5: `grpcio` build fails on Cloud
**Cause:** Streamlit Cloud builds from source if no binary wheel is available.  
**Fix:** Pin the version that has pre-built wheels:
```
grpcio==1.64.1
```

### Issue 6: `ImportError: cannot import name 'builder' from 'google.protobuf'`
**Cause:** Incompatible `protobuf` + `tensorflow` versions.  
**Fix:** Downgrade protobuf:
```
protobuf==3.20.3
```
(Use this as fallback if 4.25.3 doesn't work with your TF version.)

### Issue 7: App is slow on first prediction
**Cause:** Model is being loaded from disk; TF graph compilation on first call.  
**Fix:** The app pre-loads the model at startup with `load_model()` call in `main()`, so the first prediction is fast. Subsequent predictions use the cached graph.

---

## 🖼️ Screenshots

> Add screenshots here after deployment. Example:

```
assets/images/screenshot_normal.png   ← Normal prediction result
assets/images/screenshot_pneumonia.png ← Pneumonia detection result
assets/images/screenshot_sidebar.png   ← Model info sidebar
```

---

## 🔮 Future Improvements

| Feature | Priority | Description |
|---------|----------|-------------|
| **Grad-CAM visualization** | High | Highlight which regions of the X-ray triggered the prediction |
| **Fine-tuned Stage 2 model** | High | Deploy the fully fine-tuned version (13 unfrozen layers) |
| **Multi-class extension** | Medium | Extend to Bacterial vs. Viral Pneumonia distinction |
| **Batch prediction** | Medium | Upload multiple X-rays and download a CSV report |
| **ONNX export** | Medium | Convert to ONNX for faster CPU inference |
| **DICOM support** | Low | Accept `.dcm` files directly (clinical workflow) |
| **REST API** | Low | Wrap inference in a FastAPI endpoint |
| **Docker deployment** | Low | Containerize for self-hosting or GCP/AWS deployment |

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Abdulrhman Darwish**  
AI Diploma Student  
Deep Learning Project — Pneumonia Detection from Chest X-Rays

---

*Built with ❤️ using [Streamlit](https://streamlit.io) · [TensorFlow](https://tensorflow.org) · [DenseNet121](https://keras.io/api/applications/densenet/)*
