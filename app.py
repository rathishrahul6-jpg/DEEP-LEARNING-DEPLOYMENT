
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("mnist_cnn.keras")

st.title("MNIST Digit Classifier")
st.write("Upload a handwritten digit image")

uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.convert("L")
    image = image.resize((28, 28), Image.Resampling.LANCZOS)

    st.image(image, caption="Uploaded Image")

    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)

    prediction = model.predict(image_array)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"Predicted digit: {predicted_digit}")
    st.write(f"Confidence: {confidence:.2%}")
