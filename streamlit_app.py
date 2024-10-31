import streamlit as st
import mmcv
import cv2
import torch
from mmdet.apis import init_detector, inference_detector
import numpy as np
from PIL import Image

# Model configuration and checkpoint
CONFIG_PATH = '/home/jovyan/CRANE/model/crane_1class_fasterrcnn_10/22/24/config.py'
CHECKPOINT_PATH ='/home/jovyan/CRANE/model/crane_1class_fasterrcnn_10/22/24/epoch_300.pth'

# Initialize the model
@st.cache_resource
def load_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = init_detector(CONFIG_PATH, CHECKPOINT_PATH, device=device)
    return model

model = load_model()

def detect_cranes(image: np.ndarray):
    # Run inference
    result = inference_detector(model, image)
    return result

def visualize_results(image: np.ndarray, result, score_threshold=0.3):
    # Visualize detection results directly with model.show_result
    visualized_image = model.show_result(
        image,
        result,
        score_thr=score_threshold,
        show=False
    )
    return Image.fromarray(cv2.cvtColor(visualized_image, cv2.COLOR_BGR2RGB))

# Streamlit interface
st.title("Crane Detection in Satellite Images")
st.write("Upload a satellite image to detect cranes.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Detect cranes
    st.write("Detecting cranes...")
    result = detect_cranes(image_np)

    # Visualize the detection results
    st.write("Detection Results:")
    result_image = visualize_results(image_np, result)
    st.image(result_image, caption="Detected Cranes", use_column_width=True)
