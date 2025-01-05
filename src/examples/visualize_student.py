import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from src.data.dataset import ReadingScoreDataset
from src.model.asymptotic_model import AsymptoticModel

def load_last_model():
    """Load parameters from the most recent model"""
    results_dir = 'results'
    
    # Find the most recent parameter file
    param_files = [f for f in os.listdir(results_dir) if f.startswith('model_params_')]
    if not param_files:
        raise FileNotFoundError("No model parameter files found")
    
    latest_file = max(param_files)
    
    # Read parameters
    model = AsymptoticModel()
    with open(os.path.join(results_dir, latest_file), 'r') as f:
        params = {}
        for line in f:
            key, value = line.strip().split(': ')
            params[key] = float(value)
    
    # Set model parameters
    with torch.no_grad():
        model.k.data = torch.log(torch.tensor([params['learning_rate (k)']]))
        model.b.data = torch.logit(torch.tensor([params['initial_level (b)']]))
    
    return model

def plot_student_trajectory(model, dataset, min_tests=10, save_dir='results', jitter_amount=0.2):
    """Plot a random student's trajectory with at least min_tests reading tests."""
    # Find eligible students
    eligible_students = []
    for i in range(len(dataset)):
        X, y = dataset[i]
        if len(y) >= min_tests:
            eligible_students.append(i)
    
    if not eligible_students:
        print(f"No students found with at least {min_tests} tests.")
        return
    
    # Select random student
    student_idx = np.random.choice(eligible_students)
    X, y = dataset[student_idx]
    student_id = dataset.student_ids[student_idx]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Add small random jitter to protocol numbers for visualization
    protocols = X[:, 0]
    days = X[:, 1]
    jittered_protocols = protocols.numpy() + np.random.uniform(-jitter_amount, jitter_amount, size=len(protocols))
    
    # Plot actual data points with jitter
    plt.scatter(jittered_protocols, y.numpy(), 
               color='blue', alpha=0.7, s=100,
               label='Actual Scores')
    
    # Connect actual scores with dotted line in chronological order
    plt.plot(jittered_protocols, y.numpy(), 
            'b:', alpha=0.5, linewidth=1.5,
            label='Test Sequence')
    
    # Create smooth prediction curve
    # Generate a grid of points for smooth curve
    min_protocol = float(protocols.min())
    max_protocol = float(protocols.max())
    min_days = float(days.min())
    max_days = float(days.max())
    
    # Create smooth curve using protocol numbers
    x_smooth = torch.linspace(min_protocol, max_protocol, 100)
    days_interp = torch.tensor(np.interp(x_smooth, protocols, days))
    X_smooth = torch.stack([x_smooth, days_interp], dim=1)
    
    with torch.no_grad():
        y_smooth = model(X_smooth)
    
    # Plot smooth prediction curve
    plt.plot(x_smooth.numpy(), y_smooth.numpy(), 
            'r-', linewidth=2, label='Model Predictions')
    
    plt.xlabel('Protocol Number')
    plt.ylabel('Accuracy')
    plt.title(f'Student {student_id} Reading Progress\n({len(protocols)} tests)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set integer ticks for protocols
    plt.xticks(np.arange(int(min_protocol), int(max_protocol) + 1))
    
    # Format y-axis to show 2 decimal places
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(os.path.join(save_dir, f'student_{student_id}_trajectory_{timestamp}.png'))
    plt.close()
    
    print(f"Plotted trajectory for student {student_id}")
    return student_id

if __name__ == "__main__":
    try:
        # Load the most recent model
        model = load_last_model()
        
        # Load dataset with both data files
        dataset = ReadingScoreDataset('raw_test_data.csv', 'raw_student_data.csv')
        
        # Plot random student trajectory
        student_id = plot_student_trajectory(model, dataset, min_tests=10)
        
        print("\nPlot saved in 'results' directory")
        
    except Exception as e:
        print(f"Error: {e}") 