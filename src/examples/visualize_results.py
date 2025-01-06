import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.dataset import ReadingScoreDataset
from src.model.asymptotic_model import AsymptoticModel
from src.examples.train_simple import load_parameters
from src.visualization.learning_curves import plot_learning_curves

def visualize_results(data_path, student_data_path, param_file=None, n_students=6, min_sequences=5, random_seed=None):
    """
    Create visualization for model results using a random sample of students.
    
    Args:
        data_path: Path to raw_test_data.csv
        student_data_path: Path to raw_student_data.csv
        param_file: Path to saved model parameters (if None, uses most recent)
        n_students: Number of students to plot (default: 6)
        min_sequences: Minimum number of sequences per student (default: 5)
        random_seed: Optional random seed for reproducibility
    """
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
    
    # Find most recent parameter file if none provided
    if param_file is None:
        param_dir = 'model_params'
        if os.path.exists(param_dir):
            param_files = sorted([f for f in os.listdir(param_dir) if f.startswith('model_params_')])
            if param_files:
                param_file = os.path.join(param_dir, param_files[-1])
                print(f"Using most recent parameters from: {param_file}")
            else:
                raise ValueError("No parameter files found in model_params directory")
        else:
            raise ValueError("model_params directory not found")
    
    # Load dataset
    dataset = ReadingScoreDataset(data_path, student_data_path)
    
    # Find students with sufficient data
    student_counts = dataset.data.groupby('student_id').size()
    eligible_students = student_counts[student_counts >= min_sequences].index
    
    if len(eligible_students) == 0:
        raise ValueError(f"No students found with at least {min_sequences} sequences")
    
    # Randomly sample from eligible students
    sample_size = min(n_students, len(eligible_students))
    sampled_students = random.sample(list(eligible_students), sample_size)
    
    print(f"Plotting learning curves for {sample_size} randomly selected students "
          f"(with >= {min_sequences} sequences each)")
    
    # After sampling
    print("\nDebugging sample selection:")
    for student_id in sampled_students:
        total_sequences = len(dataset.data[dataset.data['student_id'] == student_id])
        print(f"Selected student {student_id} has {total_sequences} total sequences")
    
    # Load model with parameters
    model = AsymptoticModel()
    model = load_parameters(model, param_file)
    
    # Create plot with sampled students
    plot_learning_curves(dataset, model, selected_students=sampled_students)

if __name__ == "__main__":
    # Default paths
    data_path = 'raw_test_data.csv'
    student_data_path = 'raw_student_data.csv'
    
    # Create visualization with 6 random students using most recent model
    visualize_results(data_path, student_data_path) 