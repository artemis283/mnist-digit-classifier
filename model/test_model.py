import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn

# Load trained model
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

# Load model
model = DigitClassifier()
model.load_state_dict(torch.load("mnist_cnn.pth"))
model.eval()  # Set to evaluation mode

# Load test data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# Test on 100 samples
correct = 0
total = 100

for i, (image, label) in enumerate(test_loader):
    if i >= total:  # Stop after 100 runs
        break
    output = model(image)
    predicted_label = torch.argmax(output, dim=1).item()
    
    print(f"Sample {i+1}: Actual label: {label.item()}, Predicted label: {predicted_label}")

    if predicted_label == label.item():
        correct += 1

# Print accuracy
accuracy = (correct / total) * 100
print(f"Accuracy over {total} samples: {accuracy:.2f}%")
