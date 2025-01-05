import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # Add project root to path

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from src.data.dataset import ReadingScoreDataset
from src.model.asymptotic_model import AsymptoticModel

def create_sample_data(filepath):
    """Create a sample dataset for testing"""
    # Ensure we're not overwriting the main data file
    if os.path.basename(filepath) == 'raw_test_data.csv':
        raise ValueError("Cannot overwrite main data file 'raw_test_data.csv'")
    
    data = {
        'student_id': [1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
        'test_time': [
            '2023-01-01', '2023-01-02', '2023-01-03',
            '2023-01-01', '2023-01-02', '2023-01-03',
            '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'
        ],
        'protocol': [1, 2, 3, 1, 2, 3, 1, 2, 3, 4],
        'accuracy': [0.5, 0.7, 0.8, 0.4, 0.6, 0.7, 0.3, 0.5, 0.7, 0.8]
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save to CSV
    pd.DataFrame(data).to_csv(filepath, index=False)

def train_model(csv_path, num_epochs=100):
    # Ensure we're not modifying the main data file
    if os.path.basename(csv_path) == 'raw_test_data.csv':
        raise ValueError("Cannot use main data file 'raw_test_data.csv' for training examples")
    
    # Create dataset
    dataset = ReadingScoreDataset(csv_path)
    
    # Initialize model
    model = AsymptoticModel()
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_sequences = 0
        
        # Iterate over all students in the dataset
        for i in range(len(dataset)):
            # Get student data (protocol numbers and accuracy values)
            protocols, accuracies = dataset[i]
            
            # Forward pass
            predictions = model(protocols)
            loss = criterion(predictions, accuracies)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            epoch_loss += loss.item()
            num_sequences += 1
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / num_sequences
            print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
    
    return model

def plot_results(model, dataset):
    """Plot the trained model's predictions against actual data"""
    plt.figure(figsize=(10, 6))
    
    # Get all protocol numbers and accuracies from dataset
    all_protocols = []
    all_accuracies = []
    for i in range(len(dataset)):
        protocols, accuracies = dataset[i]
        all_protocols.extend(protocols.numpy())
        all_accuracies.extend(accuracies.numpy())
    
    # Plot actual data points
    plt.scatter(all_protocols, all_accuracies, 
               alpha=0.5, label='Actual Data', 
               color='blue')
    
    # Generate predictions for integer protocol numbers
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
    
    # Set x-axis to show integer ticks only
    plt.xticks(np.arange(1, int(max(all_protocols)) + 1))
    
    # Save plot
    plt.savefig('data/training_results.png')
    plt.close()

if __name__ == "__main__":
    # Create and use sample data
    data_path = 'data/sample_training_data.csv'
    create_sample_data(data_path)
    
    try:
        # Train model
        model = train_model(data_path)
        
        # Print final parameters
        with torch.no_grad():
            k = torch.exp(model.k)
            b = torch.sigmoid(model.b)
            print(f"\nFinal Parameters:")
            print(f"Learning rate (k): {k.item():.4f}")
            print(f"Initial level (b): {b.item():.4f}")
        
        # Create visualization
        dataset = ReadingScoreDataset(data_path)
        plot_results(model, dataset)
        print(f"\nPlot saved as 'data/training_results.png'")
        
    finally:
        # Clean up sample data file
        if os.path.exists(data_path):
            os.remove(data_path) 