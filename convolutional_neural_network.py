import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch import Tensor
import numpy as np

if not torch.cuda.is_available():
    print("WARNING: Cuda not available! Calculations will be slower!")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# dataset has PILImage images of range [0, 1]. 
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# first 0.5s are mean, second is standard deviation for all 3 color channels

# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                         shuffle=False)

# With MNIST, classes were digits
# Here, they are words
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# def imshow(imgs):
#     imgs = imgs / 2 + 0.5 # denormalize
#     npimgs = imgs.numpy()
#     plt.imshow(np.transpose(npimgs, (1, 2, 0)))
#     plt.show()

# # Data test
# dataiter = iter(train_loader)
# images, labels = next(dataiter)
# img_grid = torchvision.utils.make_grid(images[0:25],nrow=5)
# imshow(img_grid)

CONV_CHANNELS_1 = 32
CONV_CHANNELS_2 = 64
COLOR_CHANNELS = 3
# https://stats.stackexchange.com/questions/296679/what-does-kernel-size-mean
KERNEL_SIZE = 3

class ConvNeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Activation function
        self.relu = nn.ReLU()
        # Conv2d: in channels, out channels, kernel size
        # 3 color channels
        # Kernel size is the convolution window, here a 3x3
        self.conv1 = nn.Conv2d(COLOR_CHANNELS,CONV_CHANNELS_1,KERNEL_SIZE)
        # This takes the highest value in pools of 2x2
        # https://computersciencewiki.org/index.php/Max-pooling_/_Pooling
        # This reduces overfitting and increases generalization
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(CONV_CHANNELS_1,CONV_CHANNELS_2,KERNEL_SIZE)
        self.conv3 = nn.Conv2d(CONV_CHANNELS_2,64,KERNEL_SIZE)
        # Linear Layer 1
        self.fc1 = nn.Linear(64*4*4,64)
        # Linear Layer 2
        self.fc2 = nn.Linear(64,10)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))   # -> N, 32, 30, 30
        x = self.pool(x)            # -> N, 32, 15, 15
        x = self.relu(self.conv2(x))   # -> N, 64, 13, 13
        x = self.pool(x)            # -> N, 64, 6, 6
        x = self.relu(self.conv3(x))   # -> N, 64, 4, 4
        x = torch.flatten(x, 1)     # -> N, 1024
        x = self.relu(self.fc1(x))     # -> N, 64
        x = self.fc2(x)             # -> N, 10
        return x

model = ConvNeuralNetwork().to(DEVICE)

# Loss function
criterion = nn.CrossEntropyLoss()
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

n_total_steps = len(train_loader)
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images: Tensor = images.to(DEVICE)
        labels: Tensor = labels.to(DEVICE)

        # Forward pass
        outputs = model(images)
        # Calculate loss
        loss: Tensor = criterion(outputs,labels)

        # backwards pass
        optimizer.zero_grad() # Clear gradients
        loss.backward() # Calculate gradients
        optimizer.step() # Update weights

        running_loss += loss.item()

    # Divide by to calculate average
    print(f'[{epoch + 1}] loss: {running_loss / n_total_steps:.3f}')

print("Training complete!")

# PyTorcH save file
PATH = "./cnn.pth"
torch.save(model.state_dict(), PATH)

# We can load it up again, but it will need to be a new model
loaded_model = ConvNeuralNetwork()
loaded_model.load_state_dict(torch.load(PATH))
loaded_model.to(DEVICE)
loaded_model.eval() # Setup for evaluation, not training

with torch.no_grad():
    n_correct = 0
    n_correct2 = 0
    n_samples = len(test_loader.dataset)

    for images, labels in test_loader:
        images: Tensor = images.to(DEVICE)
        labels: Tensor = labels.to(DEVICE)
        outputs = model(images)

        # Max returns
        # 1 = dimensions
        _, predicted = torch.max(outputs, 1)
        n_correct += (predicted == labels).sum().item()

        outputs2 = loaded_model(images)
        _, predicted2 = torch.max(outputs2,1)
        n_correct2 += (predicted2 == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the model: {acc} %')

    acc = 100.0 * n_correct2 / n_samples
    print(f'Accuracy of the loaded model: {acc} %')