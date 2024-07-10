# alt_model.py
# Copyright (c) 2023 Sirui Li (sirui.li@murdoch.edu.au) and Kevin Wong (K.Wong@murdoch.edu.au)
# ICT203 - Artificial Intelligence and Intelligent Agents
# Murdoch University

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

class ALTModel(nn.Module):
    """
    A custom PyTorch model for your alternative neural network-based model.
    """
    def __init__(self):
        super(ALTModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)


    def forward(self, x):
        # x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def train_model(self, dataloader, epochs, criterion, optimiser):
        self.train()
        
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in tqdm(dataloader):
                optimiser.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimiser.step()
                running_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")
            
    def predict(self, dataloader):
        self.eval()
        predictions = []
        with torch.no_grad():
            for inputs, _ in dataloader:
                # inputs = inputs.view(inputs.size(0), -1)
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.tolist())
        return predictions
    
    
    def test_model(self, dataloader):
        self.eval()
        correct = 0
        total = 0
        predicted_values = []
        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                # _, predicted = torch.max(outputs, 1)
                predicted_values.append(predicted)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Accuracy of the network on the test images: {100 * correct/total}%")
        return predicted_values