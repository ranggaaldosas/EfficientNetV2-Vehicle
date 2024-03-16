from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import img_to_array
import streamlit as st

class_names = ["bajaj", "mobil", "motor", "sepeda", "taksi", "truk"]
MODEL_PATH = "model/vehicle_efficientnetV2"


loaded_model = tf.saved_model.load(MODEL_PATH)

infer = loaded_model.signatures["serving_default"]


def process_and_predict(image):
    # Adjust the target_size to match the model's expected input
    img = image.resize((224, 224))
    img_tensor = img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)  # Shape -> (1, 224, 224, 3)
    img_tensor = tf.convert_to_tensor(img_tensor)  # Convert to Tensor
    img_tensor /= 255.0  # Normalize the image similarly to the model's training

    # Predict using the loaded model
    pred = infer(tf.constant(img_tensor))[
        list(infer.structured_outputs.keys())[0]
    ].numpy()
    pred_class = class_names[np.argmax(pred)]
    return pred_class


st.title("Vehicle Image Classification")
st.write("Upload an image of the vehicle for classification.")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    # Lakukan prediksi menggunakan fungsi yang telah disesuaikan
    predicted_class = process_and_predict(image)

    # Tampilkan hasil prediksi
    st.write(f"Predicted Class: {predicted_class}")
