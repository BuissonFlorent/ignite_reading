import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from datetime import datetime

class ProtocolEffectPlot:
    """Class to visualize the relationship between protocols and accuracy."""
    
    def __init__(self, figsize=(12, 6)):
        """Initialize plot configuration."""
        self.figsize = figsize
    
    def create_figure(self):
        """Create and configure figure."""
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        return fig, ax
    
    def plot_protocol_effect(self, model, protocols=range(2, 12), time_point=0.0):
        """Plot model predictions across different protocols at a fixed time.
        
        Args:
            model: Trained AsymptoticModel
            protocols: Range of protocols to visualize
            time_point: Fixed time point for predictions
            
        Returns:
            matplotlib Figure
        """
        fig, ax = self.create_figure()
        
        # Create input tensor
        X = torch.tensor([[p, time_point] for p in protocols], dtype=torch.float32)
        
        # Generate predictions
        with torch.no_grad():
            y_pred = model(X)
        
        # Plot predictions
        ax.plot(protocols, y_pred.numpy(), 'r-', linewidth=2, 
                label='Model Predictions')
        
        # Configure axes
        ax.set_xticks(list(protocols))
        ax.set_xlabel('Protocol Number')
        ax.set_ylabel('Predicted Accuracy')
        ax.set_title('Estimated Protocol Effect on Reading Accuracy')
        ax.legend()
        
        return fig
    
    def save_plot(self, fig, save_dir='results'):
        """Save plot to file."""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(save_dir, f'protocol_effect_{timestamp}.png')
        fig.savefig(filename)
        plt.close(fig)
        return filename 