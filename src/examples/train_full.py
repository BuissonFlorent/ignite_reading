import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from src.data.dataset import ReadingScoreDataset
from src.model.asymptotic_model import AsymptoticModel

def train_model(dataset_path, num_epochs=20, learning_rate=0.01, patience=5, min_delta=1e-4):
    """
    Train the model on the full dataset with early stopping.
    Args:
        dataset_path: Path to raw_test_data.csv
        num_epochs: Maximum number of training epochs
        learning_rate: Learning rate for optimizer
        patience: Number of epochs to wait for improvement before stopping
        min_delta: Minimum change in loss to qualify as an improvement
    """
    try:
        dataset = ReadingScoreDataset(dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None
    
    model = AsymptoticModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    best_loss = float('inf')
    patience_counter = 0
    
    print(f"Starting training on {len(dataset)} student sequences")
    print("=" * 50)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_sequences = 0
        
        # Iterate over all students
        for i in range(len(dataset)):
            protocols, accuracies = dataset[i]
            predictions = model(protocols)
            loss = criterion(predictions, accuracies)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_sequences += 1
        
        avg_loss = epoch_loss / num_sequences
        train_losses.append(avg_loss)
        
        # Print progress every epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
        
        # Early stopping check
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    return model, train_losses

def save_results(model, train_losses, save_dir='results'):
    """Save model parameters and training plot"""
    # Create results directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save training loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, f'training_loss_{timestamp}.png'))
    plt.close()
    
    # Save model parameters
    with torch.no_grad():
        k = torch.exp(model.k)
        b = torch.sigmoid(model.b)
        params = {
            'learning_rate (k)': k.item(),
            'initial_level (b)': b.item(),
            'final_loss': train_losses[-1]
        }
        
        # Save parameters to file
        with open(os.path.join(save_dir, f'model_params_{timestamp}.txt'), 'w') as f:
            for key, value in params.items():
                f.write(f'{key}: {value:.4f}\n')

def plot_predictions(model, dataset, save_dir='results'):
    """Plot model predictions against actual data"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    plt.figure(figsize=(10, 6))
    
    # Collect all data points
    all_protocols = []
    all_accuracies = []
    for i in range(len(dataset)):
        protocols, accuracies = dataset[i]
        all_protocols.extend(protocols.numpy())
        all_accuracies.extend(accuracies.numpy())
    
    # Plot actual data
    plt.scatter(all_protocols, all_accuracies, 
               alpha=0.5, label='Actual Data', 
               color='blue')
    
    # Plot model predictions
    with torch.no_grad():
        x = torch.arange(1, int(max(all_protocols)) + 1, dtype=torch.float)
        y = model(x)
        plt.plot(x.numpy(), y.numpy(), 
                'r-', label='Model Predictions', 
                linewidth=2)
    
    plt.xlabel('Protocol Number')
    plt.ylabel('Accuracy')
    plt.title('Reading Accuracy vs Protocol Number')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(np.arange(1, int(max(all_protocols)) + 1))
    
    plt.savefig(os.path.join(save_dir, f'predictions_{timestamp}.png'))
    plt.close()

if __name__ == "__main__":
    dataset_path = 'raw_test_data.csv'
    
    print(f"Loading dataset from: {dataset_path}")
    model, train_losses = train_model(
        dataset_path,
        num_epochs=20,
        patience=5,      # Stop if no improvement for 5 epochs
        min_delta=1e-4   # Minimum improvement threshold
    )
    
    if model is not None:
        print("\nTraining completed. Saving results...")
        save_results(model, train_losses)
        
        dataset = ReadingScoreDataset(dataset_path)
        plot_predictions(model, dataset)
        
        print("\nResults saved in 'results' directory") 