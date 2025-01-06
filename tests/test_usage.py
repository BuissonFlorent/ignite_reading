import unittest
import torch
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data.dataset import ReadingScoreDataset

class TestUsage(unittest.TestCase):
    def setUp(self):
        # Create a small test dataset for reading scores
        test_data = {
            'student_id': [1, 1, 1, 2, 2],
            'test_time': ['2023-01-01', '2023-01-02', '2023-01-03', 
                         '2023-01-01', '2023-01-02'],
            'protocol': [1, 1, 2, 1, 2],
            'accuracy': [0.5, 0.6, 0.8, 0.4, 0.7]
        }
        
        # Create matching student data
        student_data = {
            'student_id': [1, 2],
            'program_start_date': ['2023-01-01', '2023-01-01'],
            'grade_level': ['3rd', '4th'],
            'school_name': ['Test School', 'Test School']
        }
        
        self.test_csv = 'test_data.csv'
        self.student_csv = 'student_data.csv'
        pd.DataFrame(test_data).to_csv(self.test_csv, index=False)
        pd.DataFrame(student_data).to_csv(self.student_csv, index=False)
        
        self.dataset = ReadingScoreDataset(self.test_csv, self.student_csv)
    
    def test_basic_usage(self):
        """Test basic dataset usage with single items"""
        X, y = self.dataset[0]
        self.assertIsInstance(X, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        
        # Check shapes and features
        self.assertEqual(X.shape[1], 2)  # Two features: protocol and days
        self.assertEqual(X.shape[0], y.shape[0])  # Same number of samples
        
        # Verify feature properties
        protocols = X[:, 0]
        days = X[:, 1]
        self.assertTrue(torch.all(protocols >= 1))  # Protocols start at 1
        self.assertTrue(torch.all(days >= 0))  # Days start at 0
        self.assertFalse(torch.isnan(X).any())  # No NaN values
    
    def test_iteration(self):
        """Test that we can iterate over the dataset sequences"""
        num_sequences = len(self.dataset)
        self.assertEqual(num_sequences, 5)  # Should have 5 sequences total
        
        for i in range(num_sequences):
            X, y = self.dataset[i]
            self.assertIsInstance(X, torch.Tensor)
            self.assertIsInstance(y, torch.Tensor)
            self.assertEqual(X.shape[1], 2)  # Two features
            self.assertFalse(torch.isnan(X).any())  # No NaN values
            self.assertTrue(torch.all(y >= 0) and torch.all(y <= 1))  # Valid accuracies
    
    def test_feature_scaling(self):
        """Test that features are in appropriate ranges"""
        for i in range(len(self.dataset)):
            X, y = self.dataset[i]
            protocols = X[:, 0]
            days = X[:, 1]
            
            # Check protocol range
            self.assertTrue(torch.all(protocols >= 1))
            self.assertTrue(torch.all(protocols <= 100))  # Reasonable upper bound
            
            # Check days range
            self.assertTrue(torch.all(days >= 0))
            self.assertTrue(torch.all(days.diff()[1:] >= 0))  # Days should increase
    
    def tearDown(self):
        if os.path.exists(self.test_csv):
            os.remove(self.test_csv)
        if os.path.exists(self.student_csv):
            os.remove(self.student_csv)

if __name__ == '__main__':
    unittest.main(verbosity=2) 