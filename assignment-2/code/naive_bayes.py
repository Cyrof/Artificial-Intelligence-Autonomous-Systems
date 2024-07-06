# naive_bayes.py
# Copyright (c) 2023 Sirui Li (sirui.li@murdoch.edu.au) and Kevin Wong (K.Wong@murdoch.edu.au)
# ICT203 - Artificial Intelligence and Intelligent Agents
# Murdoch University

import numpy as np
from tqdm import tqdm
import torch

class NaiveBayes:
    def __init__(self, smoothing_factor=1.0):
        """
        Args:
        1. smoothing_factor: Laplace smoothing factor to handle zero probabilities.
        2. class_probs: the prior probabilities for each class.
        3. feature_probs: the conditional probabilities for each feature given the class.
        """
        self.class_probs = None
        self.feature_probs = None
        self.smoothing_factor = smoothing_factor

    def calculate_class_probs(self, y_train):
        """
        Calculate the prior probabilities for each class.

        Args:
        y_train: Training labels.
    
        Returns:
        class_probs: Array of prior probabilities for each class.
        """
        num_classes = len(torch.unique(y_train))
        class_probs = torch.zeros(num_classes)

        for cls in range(num_classes):
            class_probs[cls] = torch.sum(y_train == cls) / len(y_train)

        return class_probs


    def calculate_feature_probs(self, x_train, y_train):
        """
        Calculate the conditional probabilities for each feature given the class.
    
        Args:
        x_train: Training features.
        y_train: Training labels.
    
        Returns:
        feature_probs: Array of conditional probabilities for each feature and class.
        """
        num_classes = len(torch.unique(y_train))
        _ , num_features = x_train.shape
        feature_probs = torch.zeros((num_classes, num_features, 2)) # for binary features

        for cls in range(num_classes):
            cls_indices = torch.where(y_train == cls)[0]
            cls_features = x_train[cls_indices]
            feature_probs[cls, :, 0] = (torch.sum(cls_features==0, axis=0) + self.smoothing_factor) / (len(cls_indices) + 2 * self.smoothing_factor)
            feature_probs[cls, :, 1] = (torch.sum(cls_features==1, axis=0) + self.smoothing_factor) / (len(cls_indices) + 2 * self.smoothing_factor)

        return feature_probs


    def train(self, x_train, y_train):
        """
        Train the NaiveBayes classifier. Do not modify this method.
    
        Args:
        x_train: Training features.
        y_train: Training labels.
        """
        self.class_probs = self.calculate_class_probs(y_train)
        self.feature_probs = self.calculate_feature_probs(x_train, y_train)

    def predict(self, x_test):
        """
        Predict the class labels for test sample.
    
        Args:
        x_test: Test features.
    
        Returns:
        predictions: Predicted class labels for test features.
        """
        num_samples, num_features = x_test.shape
        num_classes = len(self.class_probs)
        predictions = torch.zeros(num_samples, dtype=torch.int64)

        for i in tqdm(range(num_samples), desc="Predicting..."):
            log_probs = torch.zeros(num_classes)
            for cls in range(num_classes):
                log_prob = torch.log(self.class_probs[cls])
                for j in range(num_features):
                    feature_val = int(x_test[i, j])
                    log_prob += torch.log(self.feature_probs[cls, j, feature_val])
                log_probs[cls] = log_prob
            predictions[i] = torch.argmax(log_probs)        

        return predictions

        
