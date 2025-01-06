import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from src.data.dataset import ReadingScoreDataset
from src.model.asymptotic_model import AsymptoticModel
from src.examples.train_simple import load_parameters

def plot_learning_curves(dataset, model, selected_students=None, n_points=100):
    """Plot learning curves for selected students."""
    if selected_students is None:
        selected_students = dataset.data['student_id'].unique()
    
    plt.figure(figsize=(10, 6))
    
    for student_id in selected_students:
        # Get all sequences for this student
        student_data = dataset.get_student_data(student_id)
        
        # Plot actual data points
        plt.scatter(student_data['days_since_start'], 
                   student_data['accuracy'])
        
        # Generate prediction curve
        max_days = max(student_data['days_since_start'])
        time_points = np.linspace(0, max_days, n_points)
        protocols = np.ones_like(time_points) * student_data['protocol'].iloc[0]
        
        X_pred = torch.tensor([[p, t] for p, t in zip(protocols, time_points)], 
                            dtype=torch.float32)
        
        with torch.no_grad():
            y_pred = model(X_pred)
        
        # Plot predicted curve
        plt.plot(time_points, y_pred, '--')
    
    plt.xlabel('Days since start')
    plt.ylabel('Accuracy')
    plt.title('Learning Curves: Actual vs Predicted')
    plt.grid(True)
    
    # Save plot
    os.makedirs('plots', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = f'plots/learning_curves_{timestamp}.png'
    plt.savefig(plot_path) 