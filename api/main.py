import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import psycopg2
import random
import os

# Database Connection
DB_PARAMS = {
    "host": "db",
    "database": "mnist_db",
    "user": "postgres",
    "password": "password"
}

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed()

# Define the CNN-based classifier
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


try:
    model = CNNClassifier()
    state_dict = torch.load("/app/mnist_cnn.pth", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")

# Initialize FastAPI app
app = FastAPI()

# Define input schema
class DigitRequest(BaseModel):
    image: list  # Expect a 2D list (28x28)

# Image Preprocessing
def preprocess_image(image_array):
    try:
        image_tensor = torch.tensor(image_array, dtype=torch.float32)
        if image_tensor.shape != (28, 28):
            raise ValueError("Input image must be 28x28 pixels.")
        image_tensor = (image_tensor - 0.5) / 0.5
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        return image_tensor
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

@app.post("/predict")
def predict(request: DigitRequest):
    try:
        image_tensor = preprocess_image(request.image)
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        return {"predicted_digit": predicted.item(), "confidence": round(confidence.item(), 4)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

@app.get("/history")
def get_history():
    """Fetches all past predictions from the database."""
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        cur.execute("SELECT timestamp, predicted_digit, true_label FROM predictions ORDER BY timestamp DESC;")
        history = cur.fetchall()
        cur.close()
        conn.close()
        return [{"timestamp": row[0], "predicted_digit": row[1], "true_label": row[2]} for row in history]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

