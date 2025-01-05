import torch
import torch.nn as nn
from .asymptotic import AsymptoticFunctions

class AsymptoticModel(nn.Module):
    def __init__(self):
        """
        Initialize the asymptotic regression model.
        The model predicts accuracy based on protocol number,
        ensuring predictions never exceed 1.0.
        """
        super().__init__()
        
        # Initialize asymptotic function
        self.asymptotic = AsymptoticFunctions()
        
        # Learnable parameters
        self.k = nn.Parameter(torch.ones(1))  # Learning rate parameter
        self.b = nn.Parameter(torch.zeros(1))  # Initial performance level
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Protocol numbers (as tensor)
            
        Returns:
            Predicted accuracy values (between 0 and 1)
        """
        # Ensure parameters meet constraints
        k = torch.exp(self.k)  # Ensure positive learning rate
        b = torch.sigmoid(self.b)  # Ensure baseline is between 0 and 1
        
        # Maximum improvement possible from baseline
        A = 1.0 - b  # This ensures predictions never exceed 1.0
        
        # Predict accuracy using protocol numbers
        return self.asymptotic.exponential(x, A=A, k=k, b=b) 