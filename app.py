import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Wildfire AIoT System", layout="wide")

st.title("🔥 AIoT Wildfire Detection System")
st.write("Real-time Fire & Smoke Detection using YOLOv8")

# Load model
model = YOLO("runs/detect/train/weights/best.pt")

# Sidebar
st.sidebar.title("Options")
conf = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns(2)

    image = Image.open(uploaded_file)

    with col1:
        st.image(image, caption="Original Image")

    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    results = model(img, conf=conf)

    result_img = results[0].plot()
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    with col2:
        st.image(result_img, caption="Detected Output")

    # Detection status
    if len(results[0].boxes) > 0:
        st.error("🔥 Fire/Smoke Detected!")
    else:
        st.success("✅ No Fire Detected")