import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.dataset import ReadingScoreDataset
from src.data.student import Student
from torch.utils.data import DataLoader

# Create dataset with both data sources
dataset = ReadingScoreDataset(
    test_data_path='raw_test_data.csv',
    student_data_path='raw_student_data.csv'
)

# Training loop example
for X_packed, protocols_packed, y_packed in loader:
    # X_packed: days since first observation (PackedSequence)
    # protocols_packed: protocol levels (PackedSequence)
    # y_packed: accuracy scores (PackedSequence)
    pass 