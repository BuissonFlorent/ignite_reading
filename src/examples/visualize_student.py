import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from src.data.dataset import ReadingScoreDataset
from src.model.asymptotic_model import AsymptoticModel
from src.visualization.student_plots import StudentTrajectoryPlot

def load_parameters_from_file(param_file):
    """
    Load model parameters from a specific file
    
    Args:
        param_file: Path to parameter file
    
    Returns:
        dict: Parameter values
    """
    with open(param_file, 'r') as f:
        params = {}
        for line in f:
            key, value = line.strip().split(': ')
            params[key] = float(value)
    return params

def create_model_from_parameters(params):
    """Create model from parameter dictionary.
    
    Args:
        params: Dictionary containing model parameters
        
    Returns:
        Initialized AsymptoticModel
        
    Raises:
        KeyError: If required parameters are missing
        ValueError: If parameters are invalid
    """
    # Validate parameters exist
    required_params = {'learning_rate (k)', 'initial_level (b)'}
    if not required_params.issubset(params.keys()):
        raise KeyError(f"Missing required parameters. Need: {required_params}")
    
    # Validate parameter values
    k = params['learning_rate (k)']
    b = params['initial_level (b)']
    
    if k <= 0:
        raise ValueError(f"Learning rate must be positive, got {k}")
    if not 0 <= b <= 1:
        raise ValueError(f"Initial level must be between 0 and 1, got {b}")
    
    # Create and initialize model
    model = AsymptoticModel()
    model.beta_protocol.data = torch.tensor([k])
    model.beta_time.data = torch.zeros(1)
    model.b.data = torch.logit(torch.tensor([b]))
    
    return model

def load_last_model(results_dir='results'):
    """
    Load parameters from the most recent model file
    
    Args:
        results_dir: Directory containing model parameter files
    
    Returns:
        AsymptoticModel: Model with loaded parameters
    """
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    param_files = [f for f in os.listdir(results_dir) if f.startswith('model_params_')]
    if not param_files:
        raise FileNotFoundError(f"No model parameter files found in {results_dir}")
    
    latest_file = max(param_files)
    param_file = os.path.join(results_dir, latest_file)
    
    params = load_parameters_from_file(param_file)
    return create_model_from_parameters(params)

def plot_student_trajectory(model, dataset, min_tests=10, save_dir='results'):
    """Plot a student's trajectory with at least min_tests reading tests."""
    # Find eligible students
    eligible_students = dataset.get_students_with_min_tests(min_tests)
    if not eligible_students:
        raise ValueError(f"No students found with at least {min_tests} tests")
    
    # Select random student
    student_id = np.random.choice(eligible_students)
    student_data = dataset.get_student_data(student_id)
    
    # Create and save plot using plotter
    plotter = StudentTrajectoryPlot(figsize=(12, 6))  # Explicit figsize
    fig = plotter.plot_trajectory(student_data, model, student_id)
    plotter.save_plot(fig, student_id, save_dir)
    
    return student_id

def find_student_indices(dataset, student_id):
    """
    Find all sequence indices for a given student ID.
    
    Args:
        dataset: ReadingScoreDataset instance
        student_id: ID of the student to find
    
    Returns:
        list: Indices of sequences belonging to the student
    """
    indices = []
    for i in range(len(dataset)):
        if dataset.data.iloc[i]['student_id'] == student_id:
            indices.append(i)
    return indices

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