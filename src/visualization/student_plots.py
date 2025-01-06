import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import torch

class StudentTrajectoryPlot:
    """Class to handle student trajectory visualization."""
    
    def __init__(self, figsize=(12, 6)):
        """Initialize plot configuration.
        
        Args:
            figsize (tuple): Figure size in inches (width, height)
        """
        self.figsize = figsize
    
    def create_figure(self):
        """Create and configure figure."""
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        return fig, ax
    
    def plot_trajectory(self, student_data, model, student_id):
        """Create trajectory plot for a student."""
        fig, ax = self.create_figure()
        
        # Plot actual data points
        ax.scatter(student_data['protocol'], student_data['accuracy'],
                  color='blue', alpha=0.7, s=100,
                  label='Actual Scores')
        
        # Connect points chronologically
        ax.plot(student_data['protocol'], student_data['accuracy'],
                'b:', alpha=0.5, linewidth=1.5,
                label='Test Sequence')
        
        # Generate and plot predictions
        X = torch.tensor([
            [p, d] for p, d in zip(
                student_data['protocol'],
                student_data['days_since_start']
            )
        ], dtype=torch.float32)
        
        with torch.no_grad():
            y_pred = model(X)
        
        ax.plot(student_data['protocol'], y_pred.numpy(),
                'r-', linewidth=2, label='Model Predictions')
        
        # Configure axes
        min_protocol = int(student_data['protocol'].min())
        max_protocol = int(student_data['protocol'].max())
        ax.set_xticks(np.arange(min_protocol, max_protocol + 1))
        
        ax.set_xlabel('Protocol Number')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Student {student_id} Reading Progress')
        ax.legend()
        
        return fig
    
    def save_plot(self, fig, student_id, save_dir='results'):
        """Save plot to file."""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(save_dir, 
                              f'student_{student_id}_trajectory_{timestamp}.png')
        fig.savefig(filename)
        plt.close(fig)
        return filename 