import unittest
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from torch.utils.data import DataLoader
from src.data.dataset import ReadingScoreDataset

class TestUsage(unittest.TestCase):
    def setUp(self):
        # Create a small test dataset
        data = {
            'student_id': [1, 1, 1, 2, 2],
            'test_time': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-01', '2023-01-02'],
            'protocol': [1, 1, 2, 1, 2],
            'accuracy': [0.5, 0.6, 0.8, 0.4, 0.7]
        }
        self.test_csv = 'test_data.csv'
        import pandas as pd
        pd.DataFrame(data).to_csv(self.test_csv, index=False)
        self.dataset = ReadingScoreDataset(self.test_csv)
    
    def test_basic_usage(self):
        """Test basic dataset usage with single items"""
        X, y = self.dataset[0]
        self.assertIsInstance(X, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        
        # Check that protocols and accuracy values match
        self.assertEqual(X.shape, y.shape)
    
    def test_iteration(self):
        """Test that we can iterate over the dataset"""
        for i in range(len(self.dataset)):
            X, y = self.dataset[i]
            self.assertIsInstance(X, torch.Tensor)
            self.assertIsInstance(y, torch.Tensor)
    
    def tearDown(self):
        if os.path.exists(self.test_csv):
            os.remove(self.test_csv)

if __name__ == '__main__':
    unittest.main(verbosity=2) 