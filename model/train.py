import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the CNN-based classifier
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 10)  # Output layer

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.max_pool2d(x, 2)  # Downsample to 14x14
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.max_pool2d(x, 2)  # Downsample to 7x7
        x = torch.flatten(x, 1)  # Flatten feature maps
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # No activation (softmax applied in loss function)
        return x

# Data augmentation for training
transform_train = transforms.Compose([
    transforms.RandomRotation(10),  # Rotate by ±10 degrees
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Shift by ±10%
    transforms.RandomPerspective(distortion_scale=0.1, p=0.5),  # Random perspective distortions
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Normal transformation for test data (no augmentation)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load datasets
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Create model, loss function, and optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler (adjusts LR over epochs)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)

# Training loop with accuracy tracking
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Track training accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Update learning rate
    scheduler.step()

    # Print epoch results
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

# Save the trained model
torch.save(model.state_dict(), 'mnist_cnn.pth')
print("Model trained and saved as mnist_cnn.pth")

# Evaluate model on test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Final Test Accuracy: {100 * correct / total:.2f}%')


