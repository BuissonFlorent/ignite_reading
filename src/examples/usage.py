import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.dataset import ReadingScoreDataset
from src.data.student import Student
from torch.utils.data import DataLoader

# Create dataset
dataset = ReadingScoreDataset('raw_test_data.csv')

# Print feature dimensionality
print(f"Input feature dimension: {dataset.input_size}")

# Get first student_id from the dataset
first_student_id = dataset.student_ids[0]  # This should be "1464179777" based on your sample

# Access student-specific data
student = Student(student_id=first_student_id, dataset=dataset)
print(f"Student observations: {student.n_observations}")
print(f"Student mean accuracy: {student.mean_accuracy:.2%}")
print(f"Titles read: {len(student.unique_titles)}")

# Use with PyTorch DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True) 