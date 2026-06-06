# PyTorch implementation of the EEGNet feature extractor from Lawhern et al.
# Input: (batch, 1, 32, 201) for 250 Hz, 0–800 ms epochs
# Defaults for this project:
# F1 = 8, D = 2, F2 = 16
# first temporal kernel = 125 samples ≈ 500 ms at 250 Hz
# separable temporal kernel = 31 samples ≈ 124 ms at 250 Hz
# dropout = 0.5
# The model returns raw logits for BCEWithLogitsLoss.

import torch
import torch.nn as nn 

class EEGNetModel(nn.Module):
    def __init__(
        self, 
        dropout_p:float=0.5, 
        kernel_len:int=125, 
        sep_kernel_len:int=31,
        num_timepoints:int=201,
        num_channels:int=32,
        F1:int=8, 
        D:int=2
    ):  
        super().__init__()

        F2 = F1 * D
        self.num_channels = num_channels
        self.padding = (kernel_len - 1) // 2
        self.sep_padding = (sep_kernel_len - 1) // 2
        self.features = nn.Sequential(
            # Temporal conv
            nn.Conv2d(
                1, 
                F1, 
                kernel_size=(1, kernel_len), 
                padding=(0, self.padding), 
                bias=False
            ), 
            nn.BatchNorm2d(F1), 

            # Depthwise spatial
            nn.Conv2d(
                F1, 
                F2, 
                kernel_size=(self.num_channels, 1), 
                groups=F1, 
                bias=False
            ), 
            nn.BatchNorm2d(F2), 
            nn.ELU(), 
            nn.AvgPool2d(kernel_size=(1, 4)), 
            nn.Dropout(dropout_p), 
            
            # Separable conv
            nn.Conv2d(
                F2, 
                F2, 
                kernel_size=(1, sep_kernel_len), 
                padding=(0, self.sep_padding),
                groups=F2, 
                bias=False
            ), 
            nn.Conv2d(F2, F2, kernel_size=(1, 1), bias=False), 
            nn.BatchNorm2d(F2), 
            nn.ELU(), 
            nn.AvgPool2d(kernel_size=(1, 8)), 
            nn.Dropout(dropout_p),

        )

        with torch.no_grad(): 
            dummy = torch.zeros(1, 1, self.num_channels, num_timepoints)
            dummy_out = self.features(dummy)
            flattened_size = dummy_out.shape[1] * dummy_out.shape[2] * dummy_out.shape[3]

        self.classifier = nn.Linear(flattened_size, 1)

    def forward(self, x): 
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (batch, 1, channels, time), got {x.shape}")
        if x.shape[2] != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} EEG channels, got {x.shape[2]}")
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.classifier(x)
        return logits    