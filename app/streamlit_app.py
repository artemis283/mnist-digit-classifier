# app/streamlit_app.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import requests
import numpy as np
import psycopg2
from datetime import datetime

# Database Connection
DB_PARAMS = {
    "host": "db",
    "database": "mnist_db",
    "user": "postgres",
    "password": "password"
}

def log_prediction(predicted_digit, true_label):
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO predictions (timestamp, predicted_digit, true_label) VALUES (%s, %s, %s)",
            (datetime.now(), predicted_digit, true_label),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        st.error(f"Database error: {e}")

# Streamlit UI
st.title("🧮 MNIST Digit Classifier")
st.write("Draw a digit below:")

canvas = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas.image_data is not None:
    image = np.mean(canvas.image_data[:, :, :3], axis=2)  # Convert to grayscale
    image = np.resize(image, (28, 28)).astype(np.float32) / 255.0
    if st.button("Predict"):
        response = requests.post(
            "http://api:8000/predict",
            json={"image": image.tolist()},
        )
        if response.status_code == 200:
            result = response.json()
            st.write(f"Predicted Digit: {result['predicted_digit']}")
            st.write(f"Confidence: {result['confidence']:.4f}")

            true_label = st.number_input("Enter the correct digit:", min_value=0, max_value=9, step=1)
            if st.button("Submit Feedback"):
                log_prediction(result['predicted_digit'], true_label)
                st.success("Feedback submitted.")
        else:
            st.error("Prediction service error.")
