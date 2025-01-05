import torch
import torch.nn as nn
import os
from .asymptotic import AsymptoticFunctions

class AsymptoticModel(nn.Module):
    def __init__(self):
        """
        Initialize the asymptotic regression model.
        The model uses both protocol number and days since start
        as predictors in a GLM framework for the learning rate.
        """
        super().__init__()
        
        # Initialize asymptotic function
        self.asymptotic = AsymptoticFunctions()
        
        # Learnable parameters
        self.beta_protocol = nn.Parameter(torch.ones(1))  # Protocol coefficient
        self.beta_time = nn.Parameter(torch.zeros(1))     # Time coefficient
        self.b = nn.Parameter(torch.zeros(1))            # Initial performance level
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Tensor of shape (N, 2) containing:
               - x[:,0]: Protocol numbers
               - x[:,1]: Days since program start
            
        Returns:
            Predicted accuracy values (between 0 and 1)
        """
        # Ensure baseline parameter is between 0 and 1
        b = torch.sigmoid(self.b)
        
        # Maximum improvement possible from baseline
        A = 1.0 - b  # This ensures predictions never exceed 1.0
        
        # Split input into protocol and time components
        protocols = x[:, 0]
        days = x[:, 1]
        
        # Compute learning rate using GLM structure
        # Scale time to be comparable with protocols and ensure positive rate
        k = torch.exp(
            self.beta_protocol * protocols + 
            self.beta_time * days / 100  # Scale time to be comparable with protocols
        )
        
        # Use unit time steps since rate already incorporates both effects
        return self.asymptotic.exponential(
            torch.ones_like(protocols),  # Unit time steps
            A=A,
            k=k,  # Now a vector of rates
            b=b
        )
    
    @classmethod
    def from_file(cls, param_file):
        """Load model from parameter file"""
        model = cls()
        with open(param_file, 'r') as f:
            params = {}
            for line in f:
                key, value = line.strip().split(': ')
                params[key] = float(value)
        
        with torch.no_grad():
            model.beta_protocol.data = torch.tensor([params['beta_protocol']])
            model.beta_time.data = torch.tensor([params['beta_time']])
            model.b.data = torch.tensor([params['b']])
        return model
    
    @classmethod
    def from_directory(cls, model_dir='model_params'):
        """Load most recent model from directory"""
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        param_files = [f for f in os.listdir(model_dir) 
                      if f.startswith('model_params_')]
        if not param_files:
            raise FileNotFoundError(f"No parameter files found in {model_dir}")
        
        latest_file = max(param_files)
        return cls.from_file(os.path.join(model_dir, latest_file)) 