import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import json
import pickle
from pathlib import Path

pip install tensorflow[and-cuda]


# ─── HACK: Override Keras’s InputLayer to swallow legacy `batch_shape` ─────────
from tensorflow.keras.layers import InputLayer as _OrigInputLayer

class InputLayer(_OrigInputLayer):
    def __init__(self, *args, batch_shape=None, **kwargs):
        if batch_shape is not None:
            # Keras InputLayer accepts batch_input_shape, not "shape"
            kwargs['batch_input_shape'] = batch_shape
        super().__init__(*args, **kwargs)

# Patch it into both modules where Keras may look for it
import tensorflow.keras.layers              as _layers_mod
_layers_mod.InputLayer                     = InputLayer

try:
    # TF ≥2.11 layout
    import tensorflow.keras.engine.input_layer as _input_mod
except ModuleNotFoundError:
    # fallback to the python package path
    import tensorflow.python.keras.engine.input_layer as _input_mod

_input_mod.InputLayer = InputLayer


# ─── PATH SETUP ────────────────────────────────────────────────────────────────
APP_DIR       = Path(__file__).resolve().parent        # .../Zohaib Web App/App
ROOT_DIR      = APP_DIR.parent                         # .../Zohaib Web App
MODELS_DIR    = ROOT_DIR / "Models"                    # .../Zohaib Web App/Models
MOBILENET_DIR = MODELS_DIR / "MobileNet"               # .../Zohaib Web App/Models/MobileNet


# ─── LOAD MODEL & ASSETS ───────────────────────────────────────────────────────
@st.cache_resource
def load_assets_for_plant(plant_name: str):
    model_dir = MOBILENET_DIR / plant_name

    model = tf.keras.models.load_model(
        model_dir / "plant_disease_model.h5",
        custom_objects={"InputLayer": InputLayer},
        compile=False
    )

    with open(model_dir / "label_mappings.json", "r") as f:
        mappings = json.load(f)
    index_to_label = {int(k): v for k, v in mappings["index_to_label"].items()}

    with open(model_dir / "model_info.pkl", "rb") as f:
        info = pickle.load(f)
    image_size = tuple(info["image_size"])
    mode       = info["mode"]

    return model, index_to_label, image_size, mode


# ─── IMAGE PREPROCESSING ────────────────────────────────────────────────────────
def preprocess_image(img: Image.Image, image_size: tuple[int, int]):
    img = img.resize(image_size)
    arr = np.array(img)
    if arr.shape[-1] == 4:      # drop alpha channel if exists
        arr = arr[..., :3]
    arr = arr.astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


# ─── PREDICTION ────────────────────────────────────────────────────────────────
def predict_disease(model, img_array, mode, index_to_label):
    preds = model.predict(img_array)
    if mode == "binary":
        score      = float(preds[0][0])
        result     = int(score > 0.5)
        confidence = score if result == 1 else 1 - score
    else:
        result     = int(np.argmax(preds, axis=1)[0])
        confidence = float(np.max(preds))
    return index_to_label[result], confidence


# ─── STREAMLIT APP ────────────────────────────────────────────────────────────
st.set_page_config(page_title="CropVision 🌿", layout="centered")
st.title("🌱 CropVision - Plant Disease Classifier (MobileNet)")

available_plants = sorted([p.name for p in MOBILENET_DIR.iterdir() if p.is_dir()])
plant_choice     = st.selectbox("Select Plant Type", available_plants)

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file and plant_choice:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Leaf Image", width=300)

    model, index_to_label, image_size, mode = load_assets_for_plant(plant_choice)
    img_array = preprocess_image(img, image_size)
    label, conf = predict_disease(model, img_array, mode, index_to_label)

    st.success(f"✅ Predicted Disease: **{label}**")
    st.info(f"🧠 Confidence: `{conf * 100:.2f}%`")
