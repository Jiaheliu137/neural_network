import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from utils import *

# Define the neural network model
class ParityModel(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super(ParityModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size_1) # layer1 is the first hidden layer
        self.layer2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.layer3 = nn.Linear(hidden_size_2, output_size)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.layer1(x)) 
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        return x

# Set hyperparameters
input_size = 5
hidden_size_1 = 16
hidden_size_2 = 16
output_size = 1
learning_rate = 2
num_epochs = 5000

# Load dataset
inputs, outputs = generate_parity_dataset(4)
inputs = torch.tensor(inputs, dtype=torch.float32)
outputs = torch.tensor(outputs, dtype=torch.float32)

# Initialize the model, loss function, and optimizer
model = ParityModel(input_size, hidden_size_1, hidden_size_2, output_size)
criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
loss_list = []
for epoch in range(num_epochs):
    # Forward pass
    predictions = model(inputs)
    
    # Compute loss
    loss = criterion(predictions, outputs)
    loss_list.append(loss.item())

    # Backward pass

    model.zero_grad()
    # optimizer.zero_grad() # Set the existing gradients as 0
    loss.backward() # Automatically compute the gradient of the loss function with respect to all trainable parameters.
    # optimizer.step() # Update the model's parameters based on the calculated gradient
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

    # Print loss
    if (epoch+1) % 1000 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")


# Test the model
with torch.no_grad():
    test_inputs = inputs
    test_outputs = outputs
    predictions = model(test_inputs)
    predictions_rounded = torch.round(predictions)
    accuracy = torch.mean((predictions_rounded == test_outputs).float())


    for i in range(len(test_inputs)):
        print(f"Input: {test_inputs[i][:4].int().tolist()} | Desired Output: {test_outputs[i].item()} | Predicted: {predictions[i].item()}, Rounded: {predictions_rounded[i].item()}")

    print(f"Accuracy: {accuracy.item() * 100:.2f}%")


# Plot loss curve
plt.plot(range(num_epochs), loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
