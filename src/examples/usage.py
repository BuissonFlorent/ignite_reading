import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.dataset import ReadingScoreDataset
from src.data.student import Student
from torch.utils.data import DataLoader

# Create dataset and loader
dataset = ReadingScoreDataset('raw_test_data.csv')
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=dataset.collate_fn
)

# Training loop example
for X_packed, protocols_packed, y_packed in loader:
    # X_packed: days since first observation (PackedSequence)
    # protocols_packed: protocol levels (PackedSequence)
    # y_packed: accuracy scores (PackedSequence)
    pass 