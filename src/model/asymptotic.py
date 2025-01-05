import torch

class AsymptoticFunctions:
    def exponential(self, x, A, k, b):
        """
        Simple exponential approach to a limit: y = A * (1 - e^(-k*x)) + b
        
        Args:
            x: Input tensor of time points
            A: Amplitude parameter (must be positive)
            k: Rate parameter (must be positive)
            b: Baseline parameter (initial performance)
        """
        if A <= 0:
            raise ValueError("Amplitude parameter A must be positive")
        if k <= 0:
            raise ValueError("Rate parameter k must be positive")
            
        return A * (1 - torch.exp(-k * x)) + b