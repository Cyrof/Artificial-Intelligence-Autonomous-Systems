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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(self.device)
        self.feature = nn.Sequential(
            # 1
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2), # 28*28-->32*32-->28*28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 14*14
            
            # 2
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1), #10*10
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 5*5
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16*5*5, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10),
        )
    
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        return self.classifier(self.feature(x))
    
    def train_model(self, train_dataloader, test_dataloader, epochs, criterion, optimiser, accuracy):
        accuracy = accuracy.to(self.device)
        self = self.to(self.device)
        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = 0.0, 0.0
            for x, y in train_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                
                self.train()
                y_pred = self(x)
                loss = criterion(y_pred, y)
                train_loss += loss.item()

                acc = accuracy(y_pred, y)
                train_acc += acc
                
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            train_loss /= len(train_dataloader)
            train_acc /= len(train_dataloader)

            # validation loop
            val_loss, val_acc = 0.0, 0.0
            self.eval()
            with torch.inference_mode():
                for x, y in test_dataloader:
                    x, y = x.to(self.device), y.to(self.device)
                    
                    y_pred = self(x)
                    
                    loss = criterion(y_pred, y)
                    val_loss += loss.item()
                    
                    acc = accuracy(y_pred, y)
                    val_acc += acc

                val_loss /= len(test_dataloader)
                val_acc /= len(test_dataloader)
            
            print(f"Epoch: {epoch} | Train loss: {train_loss: .5f} | Train acc: {train_acc: .5f} | Val loss: {val_loss: .5f} | Val acc: {val_acc: .5f}")
            
        
