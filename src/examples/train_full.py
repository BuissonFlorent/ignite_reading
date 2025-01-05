import sys
import os
import json
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from src.data.dataset import ReadingScoreDataset
from src.model.asymptotic_model import AsymptoticModel
from src.examples.train_simple import save_parameters, load_parameters
from src.visualization.learning_curves import plot_learning_curves

def plot_results(dataset, model, save_dir='plots', max_curves_per_plot=6):
    """
    Plot actual vs predicted values for each sequence in multiple subplots.
    
    Args:
        dataset: ReadingScoreDataset instance
        model: Trained AsymptoticModel
        save_dir: Directory to save plots
        max_curves_per_plot: Maximum number of learning curves per subplot
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Calculate number of subplots needed
    n_sequences = len(dataset)
    n_plots = int(np.ceil(n_sequences / max_curves_per_plot))
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 5 * n_plots))
    plt.subplots_adjust(hspace=0.4)
    
    for plot_idx in range(n_plots):
        # Calculate which sequences go in this subplot
        start_idx = plot_idx * max_curves_per_plot
        end_idx = min((plot_idx + 1) * max_curves_per_plot, n_sequences)
        
        ax = fig.add_subplot(n_plots, 1, plot_idx + 1)
        
        # Plot sequences for this subplot
        for i in range(start_idx, end_idx):
            X, y = dataset[i]
            with torch.no_grad():
                predictions = model(X)
            
            # Get days and protocols for plotting
            days = X[:, 1].numpy()
            protocols = X[:, 0].numpy()
            
            # Create label with protocol information
            unique_protocols = np.unique(protocols)
            protocol_str = f"Student {i} (Protocols: {', '.join(map(str, unique_protocols.astype(int)))})"
            
            # Plot actual values
            ax.scatter(days, y.numpy(), label=f'Actual {protocol_str}', alpha=0.5)
            # Plot predicted values
            ax.plot(days, predictions.numpy(), '--', label=f'Predicted', alpha=0.7)
        
        ax.set_xlabel('Days since start')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Learning Curves (Students {start_idx}-{end_idx-1})')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)  # Set y-axis limits for accuracy
        
        # Add legend with smaller font and outside plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                 fontsize='small', borderaxespad=0.)
    
    # Save plot
    plot_path = os.path.join(save_dir, f'learning_curves_{timestamp}.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved plot to {plot_path}")
    return plot_path

def train_full_dataset(data_path, student_data_path, num_epochs=20, init_params=None,
                      patience=5, min_delta=1e-4):
    """
    Train model on full dataset with early stopping.
    
    Args:
        data_path: Path to raw_test_data.csv
        student_data_path: Path to raw_student_data.csv
        num_epochs: Maximum number of training epochs
        init_params: Optional path to initial parameter values
        patience: Number of epochs to wait for improvement before stopping
        min_delta: Minimum change in loss to qualify as an improvement
    """
    try:
        dataset = ReadingScoreDataset(data_path, student_data_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    # Split dataset indices for validation
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    split = max(1, int(np.floor(0.2 * len(dataset))))
    train_indices = indices[split:]
    val_indices = indices[:split]
    
    print(f"Training samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    
    model = AsymptoticModel()
    
    # Load initial parameters if provided
    if init_params and os.path.exists(init_params):
        model = load_parameters(model, init_params)
        print("Using initial parameters from:", init_params)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\nTraining for up to {num_epochs} epochs...")
    print("Initial model parameters:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.data}")
    
    # Early stopping setup
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        num_train_sequences = 0
        
        for i in train_indices:
            X, y = dataset[i]
            predictions = model(X)
            loss = criterion(predictions, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_train_sequences += 1
        
        avg_train_loss = train_loss / num_train_sequences
        
        # Validation phase
        model.eval()
        val_loss = 0
        num_val_sequences = len(val_indices)
        
        with torch.no_grad():
            for i in val_indices:
                X, y = dataset[i]
                predictions = model(X)
                loss = criterion(predictions, y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / num_val_sequences
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f'\nEpoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {avg_train_loss:.4f}')
            print(f'Validation Loss: {avg_val_loss:.4f}')
            print("Current model parameters:")
            for name, param in model.named_parameters():
                print(f"{name}: {param.data}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f'\nEarly stopping triggered after {epoch + 1} epochs')
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_val_loss

if __name__ == "__main__":
    data_path = 'raw_test_data.csv'
    student_data_path = 'raw_student_data.csv'
    
    # Look for most recent parameter file
    param_dir = 'model_params'
    if os.path.exists(param_dir):
        param_files = sorted([f for f in os.listdir(param_dir) if f.startswith('model_params_')])
        init_params = os.path.join(param_dir, param_files[-1]) if param_files else None
    else:
        init_params = None
    
    print("Starting training process...")
    model, final_loss = train_full_dataset(data_path, student_data_path, init_params=init_params)
    
    if model is not None:
        print("\nTraining completed.")
        print(f"Best validation loss: {final_loss:.4f}")
        print("Final model parameters:")
        for name, param in model.named_parameters():
            print(f"{name}: {param.data}")
        
        # Save final parameters with loss
        save_parameters(model, final_loss=final_loss)
        
        # Create and save plot
        dataset = ReadingScoreDataset(data_path, student_data_path)
        plot_learning_curves(dataset, model) 