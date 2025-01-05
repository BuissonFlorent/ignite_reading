import unittest
import torch
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from torch.utils.data import DataLoader
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
        X, y = self.dataset[0]  # Get first student
        self.assertIsInstance(X, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(len(X), 3)  # First student has 3 tests
        
        # Check that X contains protocol numbers
        self.assertTrue(torch.all(X >= 1))  # All protocols should be >= 1
        self.assertTrue(X.dtype == torch.float)  # Should be float tensor
    
    def test_single_student_data(self):
        """Test getting individual student data"""
        X, y = self.dataset[0]
        
        # Check shapes match
        self.assertEqual(X.shape, y.shape)
        
        # Check data types
        self.assertEqual(X.dtype, torch.float)
        self.assertEqual(y.dtype, torch.float)
        
        # Check value ranges
        self.assertTrue(torch.all(X >= 1))  # Protocols start at 1
        self.assertTrue(torch.all(y >= 0) and torch.all(y <= 1))  # Accuracy between 0 and 1
    
    def tearDown(self):
        if os.path.exists(self.test_csv):
            os.remove(self.test_csv)

if __name__ == '__main__':
    unittest.main(verbosity=2) 