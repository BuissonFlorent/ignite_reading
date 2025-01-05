import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.dataset import ReadingScoreDataset
from src.model.asymptotic_model import AsymptoticModel
from src.examples.train_simple import load_parameters
from src.visualization.learning_curves import plot_learning_curves

def visualize_results(data_path, student_data_path, param_file, n_students=6, random_seed=None):
    """
    Create visualization for model results using a random sample of students.
    
    Args:
        data_path: Path to raw_test_data.csv
        student_data_path: Path to raw_student_data.csv
        param_file: Path to saved model parameters
        n_students: Number of students to plot (default: 6)
        random_seed: Optional random seed for reproducibility
    """
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
    
    # Load dataset
    dataset = ReadingScoreDataset(data_path, student_data_path)
    
    # Randomly sample students
    all_indices = list(range(len(dataset)))
    sample_size = min(n_students, len(dataset))
    sampled_indices = random.sample(all_indices, sample_size)
    
    print(f"Plotting learning curves for {sample_size} randomly selected students")
    
    # Load model with parameters
    model = AsymptoticModel()
    model = load_parameters(model, param_file)
    
    # Create plot with sampled students
    plot_learning_curves(dataset, model, selected_indices=sampled_indices)

if __name__ == "__main__":
    # Default paths
    data_path = 'raw_test_data.csv'
    student_data_path = 'raw_student_data.csv'
    
    # Find most recent parameter file
    param_dir = 'model_params'
    if os.path.exists(param_dir):
        param_files = sorted([f for f in os.listdir(param_dir) if f.startswith('model_params_')])
        if param_files:
            param_file = os.path.join(param_dir, param_files[-1])
            print(f"Using parameters from: {param_file}")
            
            # Create visualization with 6 random students
            visualize_results(data_path, student_data_path, param_file, n_students=6)
        else:
            print("No parameter files found in model_params directory")
    else:
        print("model_params directory not found") 