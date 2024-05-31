import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch import Tensor

if not torch.cuda.is_available():
    print("WARNING: Cuda not available! Calculations will be slower!")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
INPUT_SIZE = 784 # images are 28x28px
HIDDEN_SIZE = 500 # 
NUM_CLASSES = 10 # 10 different digits that can be returned
NUM_EPOCHS = 3
BATCH_SIZE = 100
LEARNING_RATE = 0.001 # Speed of learning

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
                                           batch_size=BATCH_SIZE, 
                                           shuffle=True) # Important to prevent bad pattern seeking

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, # Load testing dataset
                                          batch_size=BATCH_SIZE,
                                          shuffle=False) # Uneeded
examples = iter(test_loader)
# Formerly "example_data, example_targets = examples.next()"
# Example data is X, targets is Y
example_data, example_targets = next(examples)

# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.imshow(example_data[i][0], cmap='gray')
# plt.show()

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes) -> None:
        super(NeuralNet, self).__init__() # Do the stuff in the module probably
        # Linear layer (input) (in features, out feathers)
        self.l1 = nn.Linear(input_size, hidden_size)
        # Activation function (REctified Linear Unit)
        self.relu = nn.ReLU()
        # Second Linear layer (hidden)
        # num classes is 10, because 10 digits
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # No activation or softmax
        # "No activation because CrossEntropyLoss needs raw values"
        # softmax: normalizing?
        return out
    
model = NeuralNet(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(DEVICE)

# Loss function. Criterions "compute a gradient according to a given loss function"
criterion = nn.CrossEntropyLoss()
# Very typical optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

## Training ##
# We usually have two for loops: the first iterates over epochs, the second iterates over the training loader
n_total_steps = len(train_loader)
for epoch in range(NUM_EPOCHS):
    # Images are X, labels are Y
    for i, (images, labels) in enumerate(train_loader):
        # We want to reshape the images
        # origin shape: [100, 1, 28, 28]
        # resized: [100, 784]
        images: Tensor = images.reshape(-1, 28*28).to(DEVICE)
        labels: Tensor = labels.to(DEVICE)

        # forward pass
        outputs = model(images)
        # calculate loss
        loss: Tensor = criterion(outputs,labels)

        # backwards pass
        loss.backward()

        # optimize
        optimizer.step()
        # zero the gradients to prevent stacking
        optimizer.zero_grad()

        # debug
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{n_total_steps}], Loss [{loss.item():.4f}]')

# Now test the model
with torch.no_grad():
    n_correct = 0
    n_samples = len(test_loader.dataset)

    for images, labels in test_loader:
        # flatten again
        images: Tensor = images.reshape(-1, 28*28).to(DEVICE)
        labels: Tensor = labels.to(DEVICE)

        # Call predictions from the model, aka another forward pass
        outputs = model(images)

        # max returns (output_value, index)
        # 1 = dimensions
        _, predicted = torch.max(outputs, 1)
        n_correct += (predicted == labels).sum().item()
    
    # Calculate accuracy
    acc = n_correct / n_samples
    print(f'Accuracy of the network on the {n_samples} test images: {100*acc} %')
