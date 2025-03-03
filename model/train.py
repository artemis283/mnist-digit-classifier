import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the PyTorch model (similar to the TensorFlow model)
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

# Data preprocessing and augmentation
transform_train = transforms.Compose([
    transforms.RandomRotation(10),  # Randomly rotate digits by up to 10 degrees
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Randomly translate
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize with mean and std
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize with mean and std
])

# Load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Create the model instance
model = DigitClassifier()

# Set up the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss includes softmax internally
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(images)  # Get model predictions
        loss = criterion(outputs, labels)  # Compute the loss
        loss.backward()  # Backpropagate the gradients
        optimizer.step()  # Update the weights

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Save the trained model
torch.save(model.state_dict(), 'mnist_cnn.pth')
print("Model trained and saved as mnist_cnn.pth")

# Evaluating the model
model.eval()  # Set to evaluation mode
correct = 0
total = 0
with torch.no_grad():  # No need to compute gradients during inference
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # Get the predicted class
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test set: {100 * correct / total:.2f}%')

