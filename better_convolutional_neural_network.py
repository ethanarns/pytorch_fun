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

# Input data
COLOR_CHANNELS = 1 # Grayscale means only 1 channel
POOL_DIMS = 2 # Means 2x2
IMG_DIMS = 28 # Images are 28x28 pixels
CLASSES = 10 # 0-9
IMG_POOLED_DIMS = int(IMG_DIMS / POOL_DIMS / POOL_DIMS) # The image is 2x2 pooled twice

# Hyper-parameters
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
FILTERS_1 = 32 # Can be adjusted custom
FILTERS_2 = 64 # Can be adjusted custom
FC_FEATURES = 128 # Can be adjusted custom
FILTER_DIMENSION = 3 # 3x3 filter walked across image
FILTER_PADDING = 1

class ConvNeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super(ConvNeuralNetwork, self).__init__()
        self.pool = nn.MaxPool2d(POOL_DIMS,POOL_DIMS)
        self.relu = nn.ReLU()

        # Conv2d is a Convolutional Layer
        self.conv1 = nn.Conv2d(COLOR_CHANNELS,FILTERS_1,kernel_size=FILTER_DIMENSION,padding=FILTER_PADDING)
        self.conv2 = nn.Conv2d(FILTERS_1, FILTERS_2, kernel_size=FILTER_DIMENSION, padding=FILTER_PADDING)
        self.fc1 = nn.Linear(FILTERS_2*IMG_POOLED_DIMS*IMG_POOLED_DIMS,FC_FEATURES)
        self.fc2 = nn.Linear(FC_FEATURES,CLASSES)
    
    def forward(self, x):
        # x's shape starts as [32, 1, 28, 28]
        # The MNIST images are 28x28 px

        # The 2x2 pool samples down the 28x28 pixel images into 14x14
        x = self.pool(self.relu(self.conv1(x))) # Makes shape [32, 32, 14, 14]
        # The 2x2 pool samples down the 14x14 pixel images into 7x7
        x = self.pool(self.relu(self.conv2(x))) # Makes shape [32, 64, 7, 7]
        x = x.view(-1,FILTERS_2 * IMG_POOLED_DIMS * IMG_POOLED_DIMS) # Changes the shape to [32, 3136]
        x = self.relu(self.fc1(x)) # Changes the shape to [32, 128]
        x = self.fc2(x) # Changes the shape to [32, 10]
        return x
    
# Pre-process the data
# transform will be used to make the loaded datasets easier to use
transform = transforms.Compose([
    transforms.Resize((IMG_DIMS,IMG_DIMS)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

print("Loading datasets...")
# Load MNIST (28x28 pix images)
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

print("Starting training...")

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

# Now test it
model.eval() # Switch to evaluation mode, which is better for testing than training
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images: Tensor = images.to(DEVICE)
        labels: Tensor = labels.to(DEVICE)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the 10000 test images: {100 * correct / total}%')