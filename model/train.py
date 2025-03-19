import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Define the improved CNN-based classifier
class ImprovedCNNClassifier(nn.Module):
    def __init__(self):
        super(ImprovedCNNClassifier, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third convolutional block (additional)
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

# Data augmentation for training
transform_train = transforms.Compose([
    # Spatial transformations
    transforms.RandomRotation(15),  # Increased rotation range to ±15 degrees
    transforms.RandomAffine(
        degrees=10,  
        translate=(0.15, 0.15),  # Increased translation range to 15%
        scale=(0.85, 1.15),  # Add random scaling between 85% and 115%
        shear=10  # Add shearing transformation
    ),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Increased perspective distortion
    
    # Elastic transformations (simulates hand-writing variations)
    transforms.ElasticTransform(alpha=1.0, sigma=0.5),
    
    # Noise and blur to simulate different writing instruments
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    
    # Convert to tensor and normalize
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    
    # Add random noise occasionally 
    transforms.RandomApply([
        transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x))
    ], p=0.3)
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
model = ImprovedCNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Added weight decay

# Better learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# Training loop with accuracy tracking
epochs = 5  # Increased epochs
best_acc = 0

for epoch in range(epochs):
    # Training phase
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
    
    train_acc = 100 * correct / total
    avg_loss = total_loss / len(train_loader)
    
    # Print training results
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
    
    # Validation phase
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    val_acc = 100 * correct / total
    avg_val_loss = val_loss / len(test_loader)
    
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
    
    # Update scheduler based on validation loss
    scheduler.step(avg_val_loss)
    
    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_mnist_cnn.pth')
        torch.save(model, 'best_mnist_cnn_full.pth')
        print(f"New best model saved with accuracy: {best_acc:.2f}%")

print(f"Training complete. Best validation accuracy: {best_acc:.2f}%")

# Load the best model for final evaluation
model.load_state_dict(torch.load('best_mnist_cnn.pth'))

# Final evaluation on test set
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

final_acc = 100 * correct / total
print(f'Final Test Accuracy: {final_acc:.2f}%')

