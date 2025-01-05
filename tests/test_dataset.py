import unittest
import torch
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import PackedSequence
from src.data.dataset import ReadingScoreDataset

class TestReadingScoreDataset(unittest.TestCase):
    def setUp(self):
        # Create a small test dataset
        data = {
            'student_id': [1, 1, 1, 2, 2],
            'test_time': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-01', '2023-01-02'],
            'protocol': [1, 1, 2, 1, 2],
            'accuracy': [0.5, 0.6, 0.8, 0.4, 0.7]
        }
        self.test_csv = 'test_data.csv'
        pd.DataFrame(data).to_csv(self.test_csv, index=False)
        self.dataset = ReadingScoreDataset(self.test_csv)
    
    def test_dataset_loading(self):
        """Test basic dataset loading"""
        self.assertIsInstance(self.dataset.data, pd.DataFrame)
        self.assertEqual(len(self.dataset.student_ids), 2)  # We have 2 students
    
    def test_dataset_getitem(self):
        """Test that __getitem__ returns correct structure"""
        X, protocols, y = self.dataset[0]  # Get first student
        self.assertIsInstance(X, torch.Tensor)
        self.assertIsInstance(protocols, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(len(X), 3)  # First student has 3 tests
    
    def test_dataloader_batch(self):
        """Test that DataLoader returns correct batch structure"""
        loader = DataLoader(
            self.dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=self.dataset.collate_fn
        )
        
        X_packed, protocols_packed, y_packed = next(iter(loader))
        
        # Check that we get PackedSequence objects
        self.assertIsInstance(X_packed, PackedSequence)
        self.assertIsInstance(protocols_packed, PackedSequence)
        self.assertIsInstance(y_packed, PackedSequence)
    
    def tearDown(self):
        if os.path.exists(self.test_csv):
            os.remove(self.test_csv)

if __name__ == '__main__':
    unittest.main(verbosity=2) 