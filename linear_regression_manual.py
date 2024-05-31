import torch

if not torch.cuda.is_available():
    print("WARNING: Cuda not available! Calculations will be slower!")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Linear regression (no bias)

X = torch.tensor([1,2,3,4,5,6,7,8],dtype=torch.float32, device=DEVICE)
Y = torch.tensor([2,4,6,8,10,12,14,16], device=DEVICE)

# Weight
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=DEVICE)

# Get model output, or predict
def forward(x):
    # Multiply the weight by the input
    return w * x

# A simple loss function
# y = actual value
# y_pred = prediction
def loss(y, y_pred):
    return ((y_pred-y)**2).mean()

x_test = 5.0

learning_rate = 0.01
n_epochs = 100

for epoch in range(n_epochs):
    # predict Y
    y_pred = forward(X)

    # calculate the loss
    l = loss(Y, y_pred)

    # calculate the gradients with a backwards pass
    l.backward()

    # update the weights
    with torch.no_grad():
        # Move it closer by whatever the learning rate is
        w -= learning_rate * w.grad

    # after the weight update, zero its gradient
    w.grad.zero_()

    # debug print
    if (epoch+1) % 10 == 0:
        print(f'epoch {epoch+1}: w = {w.item():.3f}, loss = {l.item():.3f}')

print(forward(x_test).item())