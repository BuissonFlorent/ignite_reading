import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from src.data.dataset import ReadingScoreDataset
from src.model.asymptotic_model import AsymptoticModel
from src.examples.train_simple import load_parameters

def plot_learning_curves(dataset, model, save_dir='plots', selected_indices=None):
    """
    Plot actual vs predicted values for selected sequences.
    
    Args:
        dataset: ReadingScoreDataset instance
        model: Trained AsymptoticModel
        save_dir: Directory to save plots
        selected_indices: List of indices to plot (if None, plots all)
    
    Returns:
        str: Path to saved plot
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Use selected indices or all if none provided
    indices_to_plot = selected_indices if selected_indices is not None else range(len(dataset))
    n_sequences = len(indices_to_plot)
    
    # Create single figure with good size
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot selected sequences
    for i in indices_to_plot:
        X, y = dataset[i]
        with torch.no_grad():
            predictions = model(X)
        
        # Get days and protocols for plotting
        days = X[:, 1].numpy()
        protocols = X[:, 0].numpy()
        
        # Create label with protocol information
        unique_protocols = np.unique(protocols)
        protocol_str = f"Student {i} (Protocols: {', '.join(map(str, unique_protocols.astype(int)))})"
        
        # Plot with different markers for clarity
        ax.scatter(days, y.numpy(), label=f'Actual {protocol_str}', alpha=0.5)
        ax.plot(days, predictions.numpy(), '--', alpha=0.7)
    
    ax.set_xlabel('Days since start')
    ax.set_ylabel('Accuracy')
    ax.set_title('Learning Curves: Actual vs Predicted (Sample)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Add legend with smaller font and outside plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
             fontsize='small', borderaxespad=0.)
    
    # Save plot with lower DPI
    plot_path = os.path.join(save_dir, f'learning_curves_{timestamp}.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=100)
    plt.close()
    
    print(f"Saved plot to {plot_path}")
    return plot_path 