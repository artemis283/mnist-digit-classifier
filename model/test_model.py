import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2

# Define CNN-based model (matches a strong TensorFlow model)
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define CNN-based model with batch normalization (to match training model)
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Add batch normalization
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Add batch normalization
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # Apply batch norm
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))  # Apply batch norm
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load trained model
model = CNNClassifier()
model.load_state_dict(torch.load("mnist_cnn.pth"))
model.eval()


# Improved image processing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Additional preprocessing: binarization + contrast adjustment
def preprocess_image(img):
    """Ensures the image is correctly formatted without unnecessary alterations."""
    # Convert to grayscale (if not already)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Ensure digits are black on white background (invert only if needed)
    if np.mean(img) > 127:  # If background is white, invert it
        img = cv2.bitwise_not(img)
    
    # Apply a slight Gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    return img

# Make multiple predictions with slight variations
def ensemble_prediction(img_transformed):
    """Averages predictions from slightly modified versions of the input."""
    img_batch = torch.cat([
        img_transformed,  # Original
        torch.rot90(img_transformed, 1, [2, 3]),  # Rotated 90 degrees
        torch.flip(img_transformed, [2]),  # Flipped horizontally
        torch.flip(img_transformed, [3]),  # Flipped vertically
    ], dim=0)

    with torch.no_grad():
        outputs = model(img_batch)
        probs = torch.softmax(outputs, dim=1)
        avg_probs = torch.mean(probs, dim=0)  # Average over augmented versions
        predicted_class = torch.argmax(avg_probs).item()

    return predicted_class, avg_probs.max().item()  # Return predicted digit and confidence score

# Testing custom images
image_number = 1
total_images = 10
correct_predictions = 0

while os.path.isfile(f"/Users/artemiswebster/source/mnist-digit-classifier/digit/digit{image_number}.png"):
    try:
        img_path = f"/Users/artemiswebster/source/mnist-digit-classifier/digit/digit{image_number}.png"
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Error reading image {img_path}, skipping...")
            image_number += 1
            continue

        img = preprocess_image(img)  # Apply preprocessing
        img_transformed = transform(Image.fromarray(img)).unsqueeze(0)  # Add batch dimension

        # Use ensemble prediction method
        predicted, confidence = ensemble_prediction(img_transformed)

        print(f"Image {image_number}: Predicted {predicted} (Confidence: {confidence:.4f})")

        # Show image with prediction label
        plt.imshow(img, cmap="gray")
        plt.title(f"Predicted: {predicted}, Confidence: {confidence:.4f}")
        plt.show()

        # Track accuracy
        if predicted == int(img_path[-5]):  # Compare with actual label (assuming last digit in filename is true label)
            correct_predictions += 1

        image_number += 1
    except Exception as e:
        print(f"Error processing image {image_number}: {e}")
        image_number += 1

# Print final accuracy
print(f"Final Accuracy: {correct_predictions}/{total_images} ({(correct_predictions / total_images) * 100:.2f}%)")
