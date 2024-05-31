import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch import Tensor

if not torch.cuda.is_available():
    print("WARNING: Cuda not available! Calculations will be slower!")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

class ConvNeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super(ConvNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1,32,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64*7*7,128)
        self.fc2 = nn.Linear(128,10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1,64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Pre-process the data
transform = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

# Load MNIST
train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
# Place into loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Create model, put on GPU (if available)
model = ConvNeuralNetwork().to(DEVICE)

# Loss function
criterion = nn.CrossEntropyLoss()
# Optimizer
optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)

n_total_steps = len(train_loader)
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images: Tensor = images.to(DEVICE)
        labels: Tensor = labels.to(DEVICE)

        # Forward pass
        outputs = model(images)
        # Calculate the loss
        loss: Tensor = criterion(outputs,labels)

        # Backwards pass
        optimizer.zero_grad() # Clear gradients
        loss.backward() # Calculate gradients
        optimizer.step() # Update weights

        running_loss += loss.item()

    # Divide by to calculate average
    print(f'[{epoch + 1}] loss: {running_loss / n_total_steps:.3f}')