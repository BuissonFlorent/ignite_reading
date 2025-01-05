import sys
import os
import json
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import numpy as np
from src.data.dataset import ReadingScoreDataset
from src.model.asymptotic_model import AsymptoticModel

def save_parameters(model, save_dir='model_params', final_loss=None):
    """Save model parameters to a JSON file"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Get parameters as dictionary
    with torch.no_grad():
        params = {
            'beta_protocol': model.beta_protocol.item(),
            'beta_time': model.beta_time.item(),
            'b': model.b.item()
        }
        # Only add final_loss if it's not None
        if final_loss is not None:
            params['final_loss'] = final_loss
        params['timestamp'] = timestamp
    
    # Save to file
    filename = os.path.join(save_dir, f'model_params_{timestamp}.json')
    with open(filename, 'w') as f:
        json.dump(params, f, indent=4)
    
    print(f"Saved parameters to {filename}")
    return filename

def load_parameters(model, param_file):
    """Load parameters from a JSON file with validation"""
    with open(param_file, 'r') as f:
        params = json.load(f)
    
    # Check for required parameters
    required_params = ['beta_protocol', 'beta_time', 'b']
    for param in required_params:
        if param not in params:
            raise KeyError(f"Missing required parameter: {param}")
    
    # Validate parameter values
    for param in required_params:  # Only validate required parameters
        value = params[param]
        try:
            float_value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid value for {param}: {value}")
    
    # Load validated parameters
    with torch.no_grad():
        model.beta_protocol.data = torch.tensor([float(params['beta_protocol'])])
        model.beta_time.data = torch.tensor([float(params['beta_time'])])
        model.b.data = torch.tensor([float(params['b'])])
    
    print(f"Loaded parameters from {param_file}")
    return model

def train_model(data_path, student_data_path, num_epochs=20, init_params=None, 
                patience=5, min_delta=1e-4):
    """
    Train model with early stopping.
    
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

    # Split dataset indices into train and validation
    # Ensure at least one validation sample
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
        num_train = 0
        
        for i in train_indices:
            X, y = dataset[i]
            predictions = model(X)
            loss = criterion(predictions, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_train += 1
        
        avg_train_loss = train_loss / num_train
        
        # Validation phase
        model.eval()
        val_loss = 0
        num_val = len(val_indices)  # We know this is at least 1
        
        with torch.no_grad():
            for i in val_indices:
                X, y = dataset[i]
                predictions = model(X)
                loss = criterion(predictions, y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / num_val
        
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
    model, final_loss = train_model(data_path, student_data_path, init_params=init_params)
    
    if model is not None:
        print("\nTraining completed.")
        print(f"Best validation loss: {final_loss:.4f}")
        print("Final model parameters:")
        for name, param in model.named_parameters():
            print(f"{name}: {param.data}")
        
        # Save final parameters with loss
        save_parameters(model, final_loss=final_loss) 