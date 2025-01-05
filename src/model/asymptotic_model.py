import torch
import torch.nn as nn
from .asymptotic import AsymptoticFunctions

class AsymptoticModel(nn.Module):
    def __init__(self):
        """
        Initialize the asymptotic regression model.
        The model assumes a maximum accuracy of 1.0 (100%).
        """
        super().__init__()
        
        # Initialize asymptotic function
        self.asymptotic = AsymptoticFunctions()
        
        # Initial performance level (between 0 and 1)
        self.b = nn.Parameter(torch.zeros(1))
        
        # Learning rate (positive)
        self.k = nn.Parameter(torch.ones(1))
        
    def forward(self, days):
        """
        Forward pass computing predicted accuracy over time.
        A perfect score (1.0) is achievable.
        """
        k = torch.exp(self.k)  # Ensure positive learning rate
        b = torch.sigmoid(self.b)  # Initial performance between 0 and 1
        A = 1.0 - b  # Room for improvement
        
        return self.asymptotic.exponential(days, A, k, b) 