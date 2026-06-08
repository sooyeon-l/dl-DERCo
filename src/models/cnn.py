import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(
            self, 
            dropout_p:dict, 
            num_timepoints:int,
            kernel_len:int=31, 
            num_channels:int=32
    ): 
        super().__init__()
        self.num_channels = num_channels
        self.padding = (kernel_len - 1) // 2
        self.dropout_class = dropout_p['classifier']
        self.dropout_conv = dropout_p['conv']
        self.temporal_block = nn.Sequential(
            nn.Conv2d(
                1, 
                8, 
                kernel_size=(1, kernel_len), 
                padding=(0, self.padding), 
                bias=False,
            ),
            nn.BatchNorm2d(8), 
            nn.ELU(), 
            nn.AvgPool2d(kernel_size=(1, 4)), 
            nn.Dropout2d(self.dropout_conv)
        )
        self.spatial_block = nn.Sequential(
            nn.Conv2d(
                8, 
                16, 
                kernel_size=(self.num_channels, 1), 
                bias=False
            ), 
            nn.BatchNorm2d(16), 
            nn.ELU(), 
            nn.Dropout2d(self.dropout_conv)
        )

        with torch.no_grad(): 
            dummy = torch.zeros(1, 1, self.num_channels, num_timepoints)
            dummy_out = self.temporal_block(dummy)
            dummy_out = self.spatial_block(dummy_out)
            flattened_size = dummy_out.shape[1] * dummy_out.shape[2] * dummy_out.shape[3]
            
        self.classifier = nn.Sequential(
            # nn.AdaptiveAvgPool2d((1, 1)), # shape -> (batch, 16, 1, 1)
            nn.Flatten(), # shape -> (batch, flattened_size)
            nn.Linear(flattened_size, 32), 
            nn.ELU(), 
            nn.Dropout(self.dropout_class),
            nn.Linear(32, 1)
        )
        
    def forward(self, x): 
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (batch, 1, 32, T), got {x.shape}")
        if x.shape[2] != self.num_channels: # Electrode dimension
            raise ValueError(f"Expected {self.num_channels} EEG channels, got {x.shape[2]}")
        x = self.temporal_block(x)
        x = self.spatial_block(x)
        logits = self.classifier(x)
        return logits 

    
class CNNV2Model(nn.Module):
    def __init__(
        self,
        dropout_p: dict,
        num_timepoints: int,
        kernel_len: int = 31,
        sep_kernel_len: int = 15,
        num_channels: int = 32,
    ):
        super().__init__()

        padding = (kernel_len - 1) // 2
        sep_padding = (sep_kernel_len - 1) // 2
        dropout_conv = dropout_p["conv"]
        dropout_classifier = dropout_p["classifier"]

        self.num_channels = num_channels

        self.temporal_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, kernel_len),
                      padding=(0, padding), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Conv2d(16, 16, kernel_size=(1, sep_kernel_len),
                      padding=(0, sep_padding), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout2d(dropout_conv),
        )

        self.spatial_block = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(num_channels, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout2d(dropout_conv),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.num_channels, num_timepoints)
            dummy_out = self.temporal_block(dummy)
            dummy_out = self.spatial_block(dummy_out)
            flattened_size = dummy_out.shape[1] * dummy_out.shape[2] * dummy_out.shape[3]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 32),
            nn.ELU(),
            nn.Dropout(dropout_classifier),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (batch, 1, channels, T), got {x.shape}")
        if x.shape[2] != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} EEG channels, got {x.shape[2]}")
        x = self.temporal_block(x)
        x = self.spatial_block(x)
        logits = self.classifier(x)
        return logits