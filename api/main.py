import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from model.train import DigitClassifier
model = DigitClassifier()
model.load_state_dict(torch.load("mnist_cnn.pth"))
model.eval()

app = FastAPI()

class DigitRequest(BaseModel):
    image: list

@app.post("/predict")
def predict(request: DigitRequest):
    image = torch.tensor(request.image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return {
        "predicted_digit": predicted.item(),
        "confidence": confidence.item()
    }