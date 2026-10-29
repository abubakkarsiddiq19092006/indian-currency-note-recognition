
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Indian Currency Recognition",
    page_icon="💵"
)

CLASS_NAMES = [
    "1Hundrednote",
    "2Hundrednote",
    "2Thousandnote",
    "5Hundrednote",
    "Fiftynote",
    "Tennote",
    "Twentynote"
]

DENOMINATIONS = {
    "1Hundrednote": "₹100",
    "2Hundrednote": "₹200",
    "2Thousandnote": "₹2000",
    "5Hundrednote": "₹500",
    "Fiftynote": "₹50",
    "Tennote": "₹10",
    "Twentynote": "₹20"
}

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "currency_cnn_improved.keras"
    )

st.title("🇮🇳 Indian Currency Note Recognition")
st.write("Upload an Indian currency-note image to predict its denomination.")

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a note image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded note", width=350)

    if st.button("Predict Denomination"):
        resized = img.resize((160, 160))
        arr = np.array(resized, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)

        predictions = model.predict(arr, verbose=0)[0]
        index = int(np.argmax(predictions))
        name = CLASS_NAMES[index]

        st.subheader(f"Prediction: {DENOMINATIONS[name]}")
        st.write(f"Model confidence: {predictions[index] * 100:.1f}%")
        st.caption(
            "Educational prototype only. Predictions may be wrong; "
            "this app does not verify whether a note is genuine."
        )
