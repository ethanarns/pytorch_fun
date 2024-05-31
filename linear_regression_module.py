import torch
import torch.nn as nn # neural network

if not torch.cuda.is_available():
    print("WARNING: Cuda not available! Calculations will be slower!")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Linear regression
# f = w * x 
# here : f = 2 * x

X = torch.tensor([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8], [10], [12], [14], [16]], dtype=torch.float32)
# Note that this is just one long matrix, it has to be that way

n_samples, n_features = X.shape
# print(n_samples) # 8
# print(n_features) # 1

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super(LinearRegression,self).__init__()
        # define layers, in this case, the linear regression
        self.lin = nn.Linear(input_dim,output_dim)
    
    def forward(self, x):
        return self.lin(x)
    
input_size, output_size = n_features, n_features

model = LinearRegression(input_size,output_size)

X_test = torch.tensor([5.0], dtype=torch.float32)

# Terrible first prediction
# print(model(X_test).item())

# Now define the loss function and optimizer
learning_rate = 0.01
n_epochs = 1000

# Mean Squared Error loss, used in linear regression
loss = nn.MSELoss()
# Stochastic Gradient Descent
# Parameters is a built-in thing
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# The training loop
for epoch in range(n_epochs):
    # do a forward pass prediction with the input array
    y_predicted = model(X)

    # loss finds the difference, how bad it was
    # Actual vs prediction
    l = loss(Y,y_predicted)

    #calculate the gradients with a backwards pass
    l.backward()

    # Update the weights
    optimizer.step()

    # zero the gradients so it doesn't stack
    optimizer.zero_grad()

    # debug print
    if (epoch+1) % 10 == 0:
        w, b = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l.item():.3f}')

print(model(X_test).item())