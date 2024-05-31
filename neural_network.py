import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

if not torch.cuda.is_available():
    print("WARNING: Cuda not available! Calculations will be slower!")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 784 # images are 28x28px
hidden_size = 500 # ?
num_classes = 10 # 10 different digits?
num_epochs = 2
batch_size = 100
learning_rate = 0.001 # Speed of change?

train_dataset = torchvision.datasets.MNIST(root="./data", # Data save location
                                           train=True, # Make it use the training dataset
                                           transform=transforms.ToTensor(), # Auto convert data to Tensors
                                           download=True) # Download it if not found

test_dataset = torchvision.datasets.MNIST(root="./data", # Location of data
                                          train=False, # Make it use the test dataset
                                          transform=transforms.ToTensor()) # Auto convert data to Tensors
# No need to use download again, it was done above

# Data loaders allow easier iteration and more
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, # Load training dataset
                                           batch_size=batch_size, 
                                           shuffle=True) # Important to prevent bad pattern seeking

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, # Load testing dataset
                                          batch_size=batch_size,
                                          shuffle=False) # Uneeded
examples = iter(test_loader)
# Formerly "example_data, example_targets = examples.next()"
example_data, example_targets = next(examples)
print(example_data)