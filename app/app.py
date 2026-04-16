import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
import torch
import json
import numpy as np

from streamlit_drawable_canvas import st_canvas   # ⭐ IMPORTANT
from predict import load_model, predict

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="MNIST Classifier", layout="centered")

st.title("🧠 MNIST Digit Classifier")

# -------------------------------
# LOAD MODEL
# -------------------------------
model = load_model()

# -------------------------------
# 📊 LOAD REAL METRICS
# -------------------------------
st.subheader("📊 Model Performance")

try:
    with open("outputs/metrics.json", "r") as f:
        metrics = json.load(f)

    accuracy = metrics["accuracy"]
    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]

except:
    st.warning("⚠️ metrics.json not found. Using default values.")
    accuracy = 0.96
    precision = 0.95
    recall = 0.95
    f1 = 0.95

col1, col2 = st.columns(2)

col1.metric("Accuracy", f"{accuracy*100:.2f}%")
col1.metric("Precision", f"{precision*100:.2f}%")

col2.metric("Recall", f"{recall*100:.2f}%")
col2.metric("F1 Score", f"{f1*100:.2f}%")

# -------------------------------
# 📉 CONFUSION MATRIX
# -------------------------------
st.image("outputs/confusion_matrix.png", use_container_width=True)



# -------------------------------
# 📈 LOSS GRAPH
# -------------------------------
st.image("outputs/loss_graph.png", use_container_width=True)


# -------------------------------
# 📂 IMAGE UPLOAD
# -------------------------------
st.subheader("📂 Upload Handwritten Digit")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    image_tensor = transform(image).unsqueeze(0)

    result = predict(image_tensor, model)

    st.success(f"🎯 Prediction: {result}")
# -------------------------------
# ✏️ DRAW DIGIT CANVAS
# -------------------------------
st.subheader("✏️ Draw Digit")

canvas_result = st_canvas(
    fill_color="white",
    stroke_width=15,
    stroke_color="black",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = canvas_result.image_data

    # Convert to grayscale
    img = np.mean(img[:, :, :3], axis=2)

    # Convert to PIL
    img = Image.fromarray(img).convert('L')

    # Resize
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    img_tensor = transform(img).unsqueeze(0)

    # Predict
    result = predict(img_tensor, model)

    st.success(f"🖌️ Drawn Digit Prediction: {result}")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Built using PyTorch + Streamlit 🚀")