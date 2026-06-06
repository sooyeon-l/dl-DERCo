import torch
import numpy as np
from torch.utils.data import Dataset

class EEGDataset(Dataset): 
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
        assert len(self.X) == len(self.y), "X and y must have the same number of samples"

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx:int): 
        x = torch.tensor(self.X[idx]).float().unsqueeze(0)
        label = torch.tensor(self.y[idx]).float().unsqueeze(0)
        return x, label