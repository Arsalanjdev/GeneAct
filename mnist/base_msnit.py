from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 784
hidden_size = 128
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5


class MLP(nn.Module):
    def __init__(
        self,
        input_size=784,
        hidden_size=128,
        num_classes=10,
        activation_fn: Callable = None,
    ):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation_fn = activation_fn
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x


#
# model = MLP(input_size, hidden_size, num_classes).to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
# # Training loop
# for epoch in range(num_epochs):
#     for batch_idx, (data, targets) in enumerate(train_loader):
#         # Move data to device
#         data = data.to(device=device)
#         targets = targets.to(device=device)
#
#         # Reshape data (flatten images)
#         data = data.reshape(data.shape[0], -1)
#
#         # Forward pass
#         scores = model(data)
#         loss = criterion(scores, targets)
#
#         # Backward pass
#         optimizer.zero_grad()
#         loss.backward()
#
#         # Gradient descent
#         optimizer.step()
#
#     print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")
#
#
# # Evaluate model
# def check_accuracy(loader, model):
#     num_correct = 0
#     num_samples = 0
#     model.eval()
#
#     with torch.no_grad():
#         for x, y in loader:
#             x = x.to(device=device)
#             y = y.to(device=device)
#             x = x.reshape(x.shape[0], -1)
#
#             scores = model(x)
#             _, predictions = scores.max(1)
#             num_correct += (predictions == y).sum()
#             num_samples += predictions.size(0)
#
#     accuracy = float(num_correct) / float(num_samples) * 100
#     print(f"Accuracy: {accuracy:.2f}%")
#
#
# check_accuracy(train_loader, model)
# check_accuracy(test_loader, model)
