# alt_model.py
# Copyright (c) 2023 Sirui Li (sirui.li@murdoch.edu.au) and Kevin Wong (K.Wong@murdoch.edu.au)
# ICT203 - Artificial Intelligence and Intelligent Agents
# Murdoch University

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from earlystop import *
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
import numpy as np

class ALTModel(nn.Module):
    """
    A custom PyTorch model for your alternative neural network-based model.

    Attributes: 
    device (str): The device to use ('cuda' or 'cpu')
    feature (nn.Sequential): Sequential module for feature extraction.
    classifier (nn.Sequential): Sequential module for classification.
    """
    def __init__(self):
        """
        Initialise the ALTModel object.
        """
        super(ALTModel, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        """
        Defines the forward pass of the model.

        Parameters: 
        x (torch.Tensor): Input tensor.

        Returns: 
        y_pred (torch.Tensor): Predicted labels.
        """
        x = x.view(-1, 1, 28, 28)
        return self.classifier(self.feature(x))
    
    def train_model(self, train_dataloader, test_dataloader, epochs, criterion, optimiser, accuracy):
        """
        Trains the ALTModel using the provided data and evaluation criteria.

        Parameters:
        train_dataloader (DataLoader): DataLoader for training data.
        test_dataloader (DataLoader): DataLoader for testing data.
        epoch (int): Number of training epochs.
        criterion (nn.CrossEntropyLoss): Loss function.
        optimiser (torch.optim.Optimizer): Optimiser for training.
        accuracy (torch.nn.Module): Accuracy function for evaluation.
        """
        early_stopping = EarlyStopping(patience=5, min_delta=0.01)
        accuracy = accuracy.to(self.device)
        self = self.to(self.device)
        scaler = GradScaler()

        for epoch in range(epochs):
            train_loss, train_acc = 0.0, 0.0
            for x, y in train_dataloader:
                x, y = x.to(self.device), y.to(self.device)

                self.train()
                optimiser.zero_grad()

                with autocast():
                    y_pred = self(x)
                    loss = criterion(y_pred, y)

                scaler.scale(loss).backward()                    
                scaler.step(optimiser)
                scaler.update()

                train_loss += loss.item()
                train_acc += accuracy(y_pred, y)

            train_loss /= len(train_dataloader)
            train_acc /= len(train_dataloader)

            # validation loop 
            val_loss, val_acc = 0.0, 0.0
            self.eval()
            with torch.inference_mode():
                for x, y in test_dataloader:
                    x, y = x.to(self.device), y.to(self.device)

                    with autocast():
                        y_pred = self(x)
                        loss = criterion(y_pred, y)
                    
                    val_loss += loss.item()
                    val_acc += accuracy(y_pred, y)

                    if early_stopping(val_loss):
                        break
                
                val_loss /= len(test_dataloader)
                val_acc /= len(test_dataloader)

                
            
            print(f"\nEpoch: {epoch+1} | Train loss: {train_loss: .5f} | Train acc: {train_acc: .5f} | Val loss: {val_loss: .5f} | Val acc: {val_acc: .5f}")

    
    def test_model(self, test_dataloader, criterion, accuracy):
        """
        Evaluate the ALTModel on test data and print performance metrics.

        Parameters:
        test_dataloader (DataLoader): DataLoader for testing data.
        criterion (nn.CrossEntropyLoss): Loss function.
        accuracy (torch.nn.Module): Accuracy function for evaluation.

        Returns: 
        tuple: Tuple containing test accuracy, F1 score, confusion matrix, precision, and recall.
        """
        test_loss, test_acc = 0, 0

        accuracy.to(self.device)
        self.to(self.device)
        all_predictions = []
        all_labels = []
        self.eval()
        with torch.inference_mode():
            for x, y in test_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                
                y_pred = self(x)
                
                test_loss += criterion(y_pred, y)
                test_acc += accuracy(y_pred, y)

                all_predictions.append(torch.argmax(y_pred, dim=1).cpu().numpy())
                all_labels.append(y.cpu().numpy())
            
            test_loss /= len(test_dataloader)
            test_acc /= len(test_dataloader)
        
        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        
        print(f"\nTest loss: {test_loss: .5f}| Test acc:{test_acc: .5f}%")
        print(f"Test F1 Score: {f1:.2f}%")
        print(f"Test Precision: {precision:.2f}%")
        print(f"Test Recall: {recall:.2f}%")
        print(f"Test Confusion Matrix:\n{conf_matrix}\n")

        return test_acc, f1, conf_matrix, precision, recall

    def predict(self, dataloader):
        """
        Generate perfictions and scores for data using the ALTModel. 

        Parameters: 
        dataloader (DataLoader): DataLoader for input data.

        Returns: 
        tuple: Tuple containing predicted classes and scores.
        """
        self.eval()
        all_predictions = []
        all_scores = []
        with torch.inference_mode():
            for x, _ in dataloader:
                x = x.to(self.device)
                y_pred = self(x)
                predicted_class = torch.argmax(y_pred, dim=1)
                all_predictions.append(predicted_class.cpu())
                all_scores.append(y_pred.cpu())
                
        return torch.cat(all_predictions, dim=0), torch.cat(all_scores, dim=0)
