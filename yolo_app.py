
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Bahrain Currency Detector",
    page_icon="💵",
    layout="wide"
)

st.title("🇧🇭 Bahrain Currency Detection")
st.write("AI-powered Bahrain banknote detection using YOLO.")

# -----------------------------
# Load YOLO model
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load YOLO model: {e}")
    st.stop()

st.success("YOLO model loaded successfully!")

# -----------------------------
# Detection settings
# -----------------------------
confidence = st.slider(
    "Detection confidence",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.05
)

# -----------------------------
# Input selection
# -----------------------------
option = st.radio(
    "Choose input method:",
    ["Upload Image", "Take a Picture"]
)

image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload a Bahrain currency image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

elif option == "Take a Picture":
    camera_image = st.camera_input("Capture Bahrain currency")

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")

# -----------------------------
# YOLO prediction
# -----------------------------
if image is not None:

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    if st.button("Detect Currency", type="primary"):

        with st.spinner("Detecting Bahrain currency..."):

            image_array = np.array(image)

            results = model.predict(
                source=image_array,
                conf=confidence,
                verbose=False
            )

            result = results[0]

            # YOLO returns annotated image in BGR format
            annotated_image = result.plot()
            annotated_image = annotated_image[:, :, ::-1]

        st.subheader("Detection Results")

        st.image(
            annotated_image,
            caption="YOLO Detection Output",
            use_container_width=True
        )

        # -----------------------------
        # Display detected objects
        # -----------------------------
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:

            st.success(f"Detected {len(boxes)} currency object(s)!")

            for box in boxes:
                class_id = int(box.cls.item())
                class_name = model.names[class_id]
                score = float(box.conf.item())

                st.write(
                    f"💵 **{class_name}** — "
                    f"Confidence: {score:.2%}"
                )

        else:
            st.warning("No currency detected. Try another image.")

st.divider()

st.caption(
    "Bahrain Currency Detector | Powered by YOLO and Streamlit"
)
