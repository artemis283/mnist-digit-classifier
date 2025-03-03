import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import torch.nn as nn

class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 24 * 24, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
state_dict = torch.load("mnist_cnn.pth")  
print(state_dict.keys()) 

model = DigitClassifier()
model.load_state_dict(state_dict)
model.eval()

app = FastAPI()

class DigitRequest(BaseModel):
    image: list

@app.post("/predict")
def predict(request: DigitRequest):
    image = torch.tensor(request.image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    print(f"Input shape: {image.shape}") 
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return {
        "predicted_digit": predicted.item(),
        "confidence": confidence.item()
    }