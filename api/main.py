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

# Define the CNN-based classifier
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
        
        # Convert to numpy if it's a list
        if isinstance(image_array, list):
            image_array = np.array(image_array, dtype=np.float32)
        
        # Convert to tensor
        image_tensor = torch.tensor(image_array, dtype=torch.float32)
        
        # Log input statistics
        print(f"Input min: {image_tensor.min()}, max: {image_tensor.max()}, mean: {image_tensor.mean()}")
        
        # Normalize to [0,1] range if coming in as [-1,1]
        if image_tensor.min() < 0:
            image_tensor = (image_tensor + 1) / 2.0
        
        # Optional: Save debug image without requiring matplotlib
        try:
            # Create debug directory if it doesn't exist
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


# Load the model once on startup
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
        # Process the image
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

# Health check endpoint
@app.get("/health")
def health_check():
    # List all files in the /app directory to help locate model files
    files_in_app = []
    try:
        if os.path.exists("/app"):
            files_in_app = [f for f in os.listdir("/app") if f.endswith(".pth")]
    except Exception as e:
        files_in_app = ["Error listing files: " + str(e)]
    
    # Also check current directory
    current_dir_files = []
    try:
        current_dir_files = [f for f in os.listdir(".") if f.endswith(".pth")]
    except Exception as e:
        current_dir_files = ["Error listing files: " + str(e)]
    
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "files_in_app_dir": files_in_app,
        "files_in_current_dir": current_dir_files,
        "current_working_dir": os.getcwd()
    }

# New endpoint to test the model with sample digits
@app.get("/test_model")
def test_model():
    if model is None:
        return {"error": "Model not loaded"}
    
    results = []
    
    # Create simple test patterns for digits 0-9
    test_patterns = []
    
    # Digit 0: Circle
    pattern0 = np.zeros((28, 28), dtype=np.float32)
    for i in range(28):
        for j in range(28):
            if ((i-14)**2 + (j-14)**2 < 100) and ((i-14)**2 + (j-14)**2 > 36):
                pattern0[i, j] = 1.0
    test_patterns.append(("digit_0", pattern0))
    
    # Digit 1: Vertical line
    pattern1 = np.zeros((28, 28), dtype=np.float32)
    pattern1[5:23, 14:16] = 1.0
    test_patterns.append(("digit_1", pattern1))
    
    # Digit 2: Simple 2 shape
    pattern2 = np.zeros((28, 28), dtype=np.float32)
    pattern2[5:7, 8:20] = 1.0     # Top horizontal
    pattern2[5:14, 18:20] = 1.0   # Right vertical
    pattern2[12:14, 8:20] = 1.0   # Middle horizontal
    pattern2[14:22, 8:10] = 1.0   # Bottom left vertical
    pattern2[20:22, 8:20] = 1.0   # Bottom horizontal
    test_patterns.append(("digit_2", pattern2))
    
    # Process each test pattern
    for name, pattern in test_patterns:
        try:
            # Convert to tensor and preprocess
            raw_tensor = torch.from_numpy(pattern)
            # Standard MNIST normalization
            input_tensor = ((raw_tensor - 0.1307) / 0.3081).unsqueeze(0).unsqueeze(0)
            
            # Get prediction
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = F.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                results.append({
                    "pattern": name,
                    "predicted": int(predicted.item()),
                    "confidence": float(confidence.item()),
                    "probabilities": [float(p) for p in probabilities[0].tolist()]
                })
        except Exception as e:
            results.append({
                "pattern": name,
                "error": str(e)
            })
    
    return {"test_results": results}
    

    


