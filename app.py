import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.Sequential([
        tf.keras.layers.TFSMLayer("model/model.savedmodel", call_endpoint="serving_default")
    ])

model = load_model()
class_names = ['Normal', 'Anomaly']

def preprocess_image(image, target_size=(224, 224)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        min-width: 250px !important;
        max-width: 250px !important;
    }

    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
        margin-top: 0 !important;
    }

    .main > div:first-child {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }

    h1 {
        margin-top: 0px !important;
        padding-top: 0 !important;
    }

    .sidebar-button {
        display: block;
        width: 100%;
        background-color: #f0f2f6;
        border: none;
        padding: 10px 0;
        margin-bottom: 10px;
        text-align: center;
        border-radius: 6px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .sidebar-button:hover {
        background-color: #d6dae1;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Navigation with Session State ---
if "page" not in st.session_state:
    st.session_state.page = "predict"

with st.sidebar:
    st.markdown("## Navigation")
    if st.button("🔍 Predict", use_container_width=True):
        st.session_state.page = "predict"
    if st.button("📊 Model Metrics", use_container_width=True):
        st.session_state.page = "metrics"

# ---------------- Page 1: Predict ----------------
if st.session_state.page == "predict":
    st.title('🛠️ Anomaly Detection System')
    st.markdown("<h1>🛠️ Anomaly Detection System</h1>", unsafe_allow_html=True)
    st.write("Select a product image to check if it is Normal or Anomaly.")

    selected_dir = st.selectbox('Select Directory',['normal', 'anomaly'])

    if selected_dir:
        image_folder = f"sample_images/{selected_dir}"  # <-- your local image folder
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    selected_file = st.selectbox("Select an image", ["-- Choose an image --"] + image_files)

    if selected_file != "-- Choose an image --":
        image_path = os.path.join(image_folder, selected_file)
        image = Image.open(image_path)

        img_tensor = preprocess_image(image)
        prediction = model.predict(img_tensor, verbose=0)['sequential_15'][0]
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        st.markdown('---')
        if predicted_class == 'Normal':
            st.success(f"Prediction: **{predicted_class}**")
        else:
            st.error(f"Prediction: **{predicted_class}**")
        st.info(f"Confidence: **{confidence:.2f}%**")

        st.image(image, caption="Selected Image")

# ---------------- Page 2: Metrics ----------------
elif st.session_state.page == "metrics":
    st.markdown("<h1>📈 Model Training Metrics</h1>", unsafe_allow_html=True)
    st.image("metrics/acc.png")
    st.image("metrics/loss.png")
    st.image("metrics/metri.png")
    st.markdown("---")
    st.caption("These metrics are generated during training. Helpful to evaluate model performance.")
