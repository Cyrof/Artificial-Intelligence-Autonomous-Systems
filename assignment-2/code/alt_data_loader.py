# alt_data_loader.py
# Copyright (c) 2023 Sirui Li (sirui.li@murdoch.edu.au) and Kevin Wong (K.Wong@murdoch.edu.au)
# ICT203 - Artificial Intelligence and Intelligent Agents
# Murdoch University

import torch
from torch.utils.data import Dataset
import os
import numpy as np

class ALTDataLoader(Dataset):
    """
    Do not modify this file.
    Please note that this dataloader is based on "torch.utils.data import Dataset"
    """
    def __init__(self, data_dir, mode):
        """
        Args:
        data_dir: Directory containing data files.
        mode: 'train', 'valid', or 'test' mode for loading corresponding data.
        """
        self.data_folder = f"{os.path.dirname(os.path.realpath(__file__))}/data"
        self.x = np.load(os.path.join(self.data_folder, data_dir, f'x_{mode}.npy'))
        self.y = np.load(os.path.join(self.data_folder, data_dir, f'y_{mode}.npy'))
        
    def __len__(self):
        """
        Get the length of the dataset.
        """
        return len(self.x)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.
    
        Args:
        idx: Index of the item to retrieve.
    
        Returns:
        x (tensor): Input tensor.
        y (tensor): Label tensor.
        """
        return torch.tensor(self.x[idx].reshape(784), dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

    def get_labels(self):
        return self.y