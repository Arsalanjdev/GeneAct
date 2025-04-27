from abc import ABC, abstractmethod
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader
from msnit.base_msnit import MLP


class BaseNeuralNetwork(ABC):
    def __init__(
            self,
            train_loader: DataLoader,
            test_loader: DataLoader,
            activation_function: Callable,
    ):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model: nn.Module = None
        self.activation_function = activation_function
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def build_model(self, activation_fn: callable) -> nn.Module:
        """Constructs and returns a fresh nn.Module, using the given activation."""
        pass

    @abstractmethod
    def train(self, epochs: int = 1) -> None:
        """Trains self.model on self.train_loader."""
        pass

    @abstractmethod
    def evaluate(self) -> float:
        """Evaluates self.model on self.test_loader, returns accuracy (0–1)."""
        pass


class MLPModel(BaseNeuralNetwork):
    def __init__(self, train_loader: DataLoader, test_loader: DataLoader):
        super().__init__(train_loader, test_loader)
        self.model = MLP(activation_fn=self.activation_function).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    # XXX:use factory method?
    def train(self, epochs: int = 3) -> None:
        for epoch in range(epochs):
            for batch_idx, (data, targets) in enumerate(self.train_loader):
                # Move data to device
                data = data.to(device=self.device)
                targets = targets.to(device=self.device)

                # Reshape data (flatten images)
                data = data.reshape(data.shape[0], -1)

                # Forward pass
                scores = self.model(data)
                loss = self.criterion(scores, targets)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient descent
                self.optimizer.step()

    def evaluate(self) -> float:
        num_correct = 0
        num_samples = 0
        self.model.eval()

        with torch.no_grad():
            for x, y in self.train_loader:
                x = x.to(device=self.device)
                y = y.to(device=self.device)
                x = x.reshape(x.shape[0], -1)

                scores = self.model(x)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)

        accuracy = float(num_correct) / float(num_samples) * 100
        print(f"Accuracy: {accuracy:.2f}%")
