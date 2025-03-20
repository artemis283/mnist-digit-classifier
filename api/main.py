import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import os
import base64
import io
from PIL import Image

# Set seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed()

class ImprovedCNNClassifier(nn.Module):
    def __init__(self):
        super(ImprovedCNNClassifier, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # First block
        x = self.bn1(self.conv1(x))
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)  # 28x28 -> 14x14
        
        # Second block
        x = self.bn2(self.conv2(x))
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)  # 14x14 -> 7x7
        
        # Third block
        x = self.bn3(self.conv3(x))
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)  # 7x7 -> 3x3
        
        # Flatten and fully connected
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Initialize FastAPI app
app = FastAPI()

# Define input schema
class DigitRequest(BaseModel):
    image: list  # Expect a 2D list (28x28)

def preprocess_image(image_array):
    try:
        # Import necessary libraries
        import numpy as np
        import torch
        import os
    
        if isinstance(image_array, list):
            image_array = np.array(image_array, dtype=np.float32)

        image_tensor = torch.tensor(image_array, dtype=torch.float32)
        

        print(f"Input min: {image_tensor.min()}, max: {image_tensor.max()}, mean: {image_tensor.mean()}")
        
  
        if image_tensor.min() < 0:
            image_tensor = (image_tensor + 1) / 2.0
        

        try:
            os.makedirs("/app/debug", exist_ok=True)
            
            # Save as a simple text file for debugging (no matplotlib required)
            with open("/app/debug/pixel_values.txt", "w") as f:
                f.write(f"Min: {image_tensor.min()}, Max: {image_tensor.max()}, Mean: {image_tensor.mean()}\n")
                f.write("Sample of tensor values (first 5x5 pixels):\n")
                sample = image_tensor[:5, :5].numpy()
                for row in sample:
                    f.write(" ".join([f"{val:.4f}" for val in row]) + "\n")
        except Exception as e:
            print(f"Failed to save debug info: {e}")
        
        # Apply MNIST normalization
        image_tensor = (image_tensor - 0.1307) / 0.3081
        
        print(f"After normalization - min: {image_tensor.min()}, max: {image_tensor.max()}, mean: {image_tensor.mean()}")
        
        # Reshape for model
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        print(f"Preprocessed tensor shape: {image_tensor.shape}")
        
        return image_tensor
    except Exception as e:
        print(f"Preprocessing error: {e}")
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(e)}")



model = None
try:
    model_path = "/app/best_mnist_cnn.pth"
    if os.path.exists(model_path):
        model = ImprovedCNNClassifier()
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        print("Model loaded successfully")
        
        # Test model with zeros and ones
        test_zeros = torch.zeros((1, 1, 28, 28))
        test_ones = torch.ones((1, 1, 28, 28))
        with torch.no_grad():
            zeros_output = model(test_zeros)
            ones_output = model(test_ones)
        print(f"Test with zeros - output: {zeros_output}")
        print(f"Test with ones - output: {ones_output}")
    else:
        print(f"Warning: Model file not found at {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")

# Prediction endpoint
@app.post("/predict")
def predict(request: DigitRequest):
    try:
        image_tensor = preprocess_image(request.image)
        
        
        # Check if model is loaded
        if model is not None:
            with torch.no_grad():
                output = model(image_tensor)
                print(f'output: {output}')
                
                # Try different ways to convert logits to probabilities
                probabilities = F.softmax(output, dim=1)
                print(f'probabilities (standard softmax): {probabilities}')
                
                # Alternative: Scale outputs to reduce extreme values
                scaled_output = output / 2.0  # Soften extreme values
                alt_probabilities = F.softmax(scaled_output, dim=1)
                print(f'probabilities (scaled softmax): {alt_probabilities}')
                
                # Get predictions
                confidence, predicted = torch.max(probabilities, 1)
                alt_confidence, alt_predicted = torch.max(alt_probabilities, 1)
                
                print(f'confidence: {confidence}, predicted: {predicted}')
                print(f'alt_confidence: {alt_confidence}, alt_predicted: {alt_predicted}')
            
            return {
                "predicted_digit": int(predicted.item()),
                "confidence": float(confidence.item()),
                "probabilities": probabilities[0].tolist(),
                "alt_prediction": int(alt_predicted.item()),
                "alt_confidence": float(alt_confidence.item())
            }
        else:
            # Return a default response if model is not available
            return {
                "predicted_digit": -1,
                "confidence": 0.0,
                "error": "Model not loaded. Please check server logs."
            }
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")



    


