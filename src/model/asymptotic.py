import torch

class AsymptoticFunctions:
    def exponential(self, x, A=1.0, k=1.0, b=0.0):
        """
        Exponential asymptotic function.
        
        Args:
            x: Input tensor
            A: Maximum improvement (can be scalar or tensor)
            k: Learning rate (can be scalar or tensor)
            b: Initial level (can be scalar or tensor)
        """
        return A * (1 - torch.exp(-k * x)) + b