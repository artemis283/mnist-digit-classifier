
# app/streamlit_app.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import requests
import numpy as np
import psycopg2
from datetime import datetime
from PIL import Image
from scipy.ndimage import zoom
import pandas as pd

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
        return True
    except Exception as e:
        st.error(f"Database error: {e}")
        return False

def process_image(image_data, size=28):
    """Convert drawn image to grayscale and resize to 28x28."""
    # Convert image to grayscale
    grayscale_image = np.mean(image_data[:, :, :3], axis=2)
    
    # Resize image
    resized_image = zoom(grayscale_image, size / grayscale_image.shape[0])
    
    # Normalize pixel values
    normalized_image = resized_image.astype(np.float32) / 255
    
    # Return image as a 2D array (not reshaped to a single row)
    return normalized_image

def fetch_history():
    """Fetches all past predictions from the database."""
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        cur.execute("SELECT timestamp, predicted_digit, true_label FROM predictions ORDER BY timestamp DESC LIMIT 50;")
        history = cur.fetchall()
        cur.close()
        conn.close()
        return history
    except Exception as e:
        st.error(f"Error fetching history: {e}")
        return []

# Streamlit UI
def main():
    st.title("🧮 MNIST Digit Classifier")
    st.write("Draw a digit below:")

    canvas = st_canvas(
        fill_color="black",
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    # Use a session state to track prediction result
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None

    if canvas.image_data is not None:
        # Process the image
        image = process_image(canvas.image_data)
        
        # Display the processed image
        st.write(f"Image Range: Min = {image.min()}, Max = {image.max()}")
        
        # Create a larger version for display
        display_img = Image.fromarray((image * 255).astype(np.uint8)).resize((140, 140), Image.NEAREST)
        st.image(display_img, caption="Processed Image (28x28)")
        
        # Predict button
        if st.button("Predict"):
            try:
                response = requests.post(
                    "http://api:8000/predict",
                    json={"image": image.tolist()},
                )
                if response.status_code == 200:
                    st.session_state.prediction_result = response.json()
                    st.write(f"Predicted Digit: {st.session_state.prediction_result['predicted_digit']}")
                    st.write(f"Confidence: {st.session_state.prediction_result['confidence']:.4f}")
                else:
                    st.error("Failed to get prediction")
            except Exception as e:
                st.error(f"Prediction error: {e}")

        # Feedback section (only show if prediction exists)
        if st.session_state.prediction_result:
            true_label = st.number_input(
                "Enter the correct digit:", 
                min_value=0, 
                max_value=9, 
                step=1,
                key="true_label_input"
            )
            
            if st.button("Submit Feedback"):
                predicted_digit = st.session_state.prediction_result['predicted_digit']
                if log_prediction(predicted_digit, true_label):
                    st.success("✅ Feedback submitted successfully!")
                    # Clear the prediction result after submission
                    st.session_state.prediction_result = None
                else:
                    st.error("❌ Failed to submit feedback")

    # Prediction History Section
    st.subheader("📜 Prediction History")

    history = fetch_history()

    if history:
        # Create a DataFrame for better display
        history_df = pd.DataFrame(
            history, 
            columns=["timestamp", "pred", "label"]
        )
        
        # Format the timestamp
        history_df["timestamp"] = history_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        
        # Add accuracy indicator
        history_df["correct"] = history_df["pred"] == history_df["label"]
        
        # Create a styled table
        st.dataframe(
            history_df.drop("correct", axis=1),
            column_config={
                "timestamp": st.column_config.TextColumn("Timestamp"),
                "pred": st.column_config.NumberColumn("Prediction"),
                "label": st.column_config.NumberColumn("True Label")
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Display accuracy statistics
        correct_count = history_df["correct"].sum()
        total_count = len(history_df)
        if total_count > 0:
            accuracy = (correct_count / total_count) * 100
            st.metric("Accuracy", f"{accuracy:.1f}%", f"{correct_count}/{total_count} correct")
    else:
        st.write("No history available yet.")

if __name__ == "__main__":
    main()