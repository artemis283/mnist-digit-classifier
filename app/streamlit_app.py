# app/streamlit_app.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import requests
import numpy as np
import psycopg2
from datetime import datetime
from PIL import Image

# Database Connection
DB_PARAMS = {
    "host": "db",
    "database": "mnist_db",
    "user": "postgres",
    "password": "password"
}

def log_prediction(predicted_digit, true_label):
    """Logs predictions into PostgreSQL database."""
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

def fetch_history():
    """Fetches all past predictions from the database."""
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        cur.execute("SELECT timestamp, predicted_digit, true_label FROM predictions ORDER BY timestamp DESC;")
        history = cur.fetchall()
        cur.close()
        conn.close()
        return history
    except Exception as e:
        st.error(f"Error fetching history: {e}")
        return []

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
    # Convert to grayscale (only RGB channels, ignore alpha)
    image = np.mean(canvas.image_data[:, :, :3], axis=2).astype(np.uint8)

    # Print min/max before processing
    st.write(f"Image Range after Grayscale Conversion: Min = {image.min()}, Max = {image.max()}")

    # Convert to PIL image and resize
    image = Image.fromarray(image)
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32)

    # Convert grayscale image to range [0, 1]
    image = image / 255.0

    # Ensure min/max are within [0, 1] (just in case)
    image = np.clip(image, 0.0, 1.0)

    st.write(f"Image Range after Normalizing to [0, 1]: Min = {image.min()}, Max = {image.max()}")

    # Normalize to [-1, 1] for model input
    image = 2 * image - 1  # This keeps it between -1 and 1

    # Ensure final normalization is within expected range
    image = np.clip(image, -1.0, 1.0)

    # Final debug info
    st.write(f"Image Range after Normalizing to [-1, 1]: Min = {image.min()}, Max = {image.max()}")


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

# Display Prediction History
st.subheader("📜 Prediction History")

history = fetch_history()

if history:
    st.table(
        [{"Timestamp": row[0], "Prediction": row[1], "True Label": row[2]} for row in history]
    )
else:
    st.write("No history available yet.")



