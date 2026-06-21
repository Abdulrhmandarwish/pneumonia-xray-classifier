# Pneumonia Chest X-Ray Classifier

A Streamlit web app that uses a DenseNet121 deep learning model to classify
chest X-ray images as **NORMAL** or **PNEUMONIA**.

## ⚠️ Disclaimer
This is an **educational / student project** and is **NOT** a medical diagnostic tool.

## How it works
1. Upload a chest X-ray image (JPG, JPEG, or PNG)
2. The image is preprocessed (resize to 224×224, RGB, DenseNet preprocessing)
3. A trained DenseNet121 model predicts the class
4. The result and confidence score are displayed

## Model Details
- **Architecture**: DenseNet121 (transfer learning from ImageNet)
- **Training**: Binary classification with sigmoid output
- **Classes**: NORMAL (0), PNEUMONIA (1)
- **Threshold**: 0.5

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
