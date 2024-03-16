import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import streamlit as st
import os

from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image


class_names = ["bajaj", "mobil", "motor", "sepeda", "taksi", "truk"]
MODEL_PATH = "model/vehicle_efficientnetV2"


loaded_model = tf.saved_model.load(MODEL_PATH)

infer = loaded_model.signatures["serving_default"]


def process_and_predict(image):
    img = image.convert("RGB").resize((224, 224))  # Convert image to RGB and resize
    img_tensor = img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)  # Shape -> (1, 224, 224, 3)
    img_tensor = tf.convert_to_tensor(img_tensor)  # Convert to Tensor
    img_tensor /= 255.0  # Normalize the image similarly to the model's training

    pred = infer(tf.constant(img_tensor))[
        list(infer.structured_outputs.keys())[0]
    ].numpy()
    pred_class = class_names[np.argmax(pred)]
    return pred_class


st.title("Vehicle Image Classification 🚗")
st.write(
    "Upload an image of the vehicle for classification or select a sample image below."
)

uploaded_image = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png"])

sample_images_dir = "assets/sample_image"  # Ensure this directory path is correct
sample_images = os.listdir(sample_images_dir)
selected_sample = st.selectbox("Or choose a sample image:", sample_images)

if st.button("Classify Selected Sample"):
    image_path = os.path.join(sample_images_dir, selected_sample)
    image = Image.open(image_path)
    st.image(image, caption="Selected Sample Image", use_column_width=True)
    st.write("Classifying...")
    predicted_class = process_and_predict(image)
    st.write(f"Predicted Class: {predicted_class}")

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")
    predicted_class = process_and_predict(image)
    st.write(f"Predicted Class: {predicted_class}")
