import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2

class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        # Flatten layer is implicit in PyTorch; we use Linear layers
        self.fc1 = nn.Linear(28 * 28, 128)  # First fully connected layer (input is 28x28)
        self.fc2 = nn.Linear(128, 128)      # Second fully connected layer
        self.fc3 = nn.Linear(128, 10)       # Output layer (10 classes for digits 0-9)

    def forward(self, x):
        # Flatten the image (28x28 to 784) and apply ReLU activations
        x = torch.flatten(x, 1)  # Flattening all dimensions except batch
        x = torch.relu(self.fc1(x))  # First hidden layer
        x = torch.relu(self.fc2(x))  # Second hidden layer
        x = self.fc3(x)  # Output layer (no activation here, we'll apply softmax in the loss function)
        return x

# Load the trained model
model = DigitClassifier()
model.load_state_dict(torch.load("mnist_cnn.pth"))
model.eval()  # Set to evaluation mode

# Print model parameters after loading the weights
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Layer {name}, Shape: {param.shape}")


# Image transformation (to match model's input format)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure the image is in grayscale
    transforms.Resize((28, 28)),  # Resize the image to 28x28
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize to the same range as training data
])

# Load and test hand-drawn images
image_number = 1
correct = 0
total = 11

# Load custom images and predict them
while os.path.isfile('/Users/artemiswebster/source/mnist-digit-classifier/digit/digit{}.png'.format(image_number)):
    try:
        # Read the image
        img = cv2.imread('/Users/artemiswebster/source/mnist-digit-classifier/digit/digit{}.png'.format(image_number), cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print("Error reading image! Proceeding with next image...")
            image_number += 1
            continue

        # Apply the transformations (resize, normalize, convert to tensor)
        img_transformed = transform(Image.fromarray(img))
        
        # Add batch dimension
        img_transformed = img_transformed.unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            output = model(img_transformed)
            _, predicted = torch.max(output, 1)

        print(f"The number is probably a {predicted.item()}")

        # Display the image
        plt.imshow(img, cmap=plt.cm.binary)
        plt.show()

        image_number += 1
    except Exception as e:
        print(f"Error: {e}. Proceeding with next image...")
        image_number += 1
