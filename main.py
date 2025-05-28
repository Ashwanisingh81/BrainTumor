import streamlit as st

# Must be the first Streamlit command
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

from ultralytics import YOLO
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

@st.cache_resource
def load_model():
    return YOLO("best.pt")  # Replace with your trained model path

model = load_model()

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "upload"
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# PAGE 1: Image Upload
if st.session_state.page == "upload":
    st.title("🧠 Brain Tumor Detection using YOLOv11")
    uploaded_file = st.file_uploader("Upload an MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.session_state.uploaded_image = image
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Show "Next" button after image upload
        if st.button("➡️ Next"):
            st.session_state.page = "result"
            # Save image temporarily
            image.save("temp_image.jpg")
            st.rerun()

# PAGE 2: Result
elif st.session_state.page == "result":
    st.title("🧠 Detection Result")

    conf_threshold = st.slider("Detection Confidence Threshold", 0.0, 1.0, 0.2, 0.05)

    with st.spinner("Detecting brain tumor(s)..."):
        results = model.predict(source="temp_image.jpg", save=False, imgsz=640, conf=conf_threshold)
        result = results[0]
        result_array = result.plot()
        result_image = Image.fromarray(result_array)
        st.image(result_image, caption="Detected Tumor(s)", use_container_width=True)

    with st.expander("🔍 Detection Details"):
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                st.write(f"🟦 Detection {i + 1}:")
                st.write(f"- Class ID: {cls_id}")
                st.write(f"- Confidence: {conf:.2f}")
                st.write(f"- Bounding Box: {xyxy}")
        else:
            st.write("No detections found.")

    with st.expander("📊 Confidence Score Chart"):
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            confidences = boxes.conf.cpu().numpy().tolist()
            x = list(range(1, len(confidences) + 1))

            fig, ax = plt.subplots()
            ax.bar(x, confidences, color='skyblue', edgecolor='black')
            ax.scatter(x, confidences, color='red', zorder=5, label='Confidence Point')
            for i, conf in zip(x, confidences):
                ax.text(i, conf + 0.03, f"{conf:.2f}", ha='center', va='bottom', fontsize=8)
            ax.set_xlabel("Detection Index")
            ax.set_ylabel("Confidence Score")
            ax.set_title("Model Confidence per Detection")
            ax.set_xticks(x)
            ax.set_ylim(0, 1.05)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            st.pyplot(fig)
        else:
            st.write("No confidence scores to display.")

    # Go back to upload page
    if st.button("🔁 Upload Another Image"):
        if os.path.exists("temp_image.jpg"):
            os.remove("temp_image.jpg")
        st.session_state.page = "upload"
        st.session_state.uploaded_image = None
        st.rerun()
