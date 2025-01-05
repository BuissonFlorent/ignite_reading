import unittest
import torch
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data.dataset import ReadingScoreDataset

class TestReadingScoreDataset(unittest.TestCase):
    def setUp(self):
        # Create a small test dataset for reading scores
        test_data = {
            'student_id': [1, 1, 1, 2, 2],
            'test_time': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-01', '2023-01-02'],
            'protocol': [1, 1, 2, 1, 2],
            'accuracy': [0.5, 0.6, 0.8, 0.4, 0.7]
        }
        
        # Create a small test dataset for student info
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
    
    def test_dataset_loading(self):
        """Test basic dataset loading"""
        self.assertIsInstance(self.dataset.data, pd.DataFrame)
        self.assertEqual(len(self.dataset.student_ids), 2)  # We have 2 students
        
        # Check that program_start_date and days_since_start were added
        self.assertIn('program_start_date', self.dataset.data.columns)
        self.assertIn('days_since_start', self.dataset.data.columns)
    
    def test_dataset_getitem(self):
        """Test that __getitem__ returns correct structure"""
        X, y = self.dataset[0]  # Get first student
        self.assertIsInstance(X, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        
        # Check X shape (should be 2D with protocol and days)
        self.assertEqual(X.shape[1], 2)  # Two features
        self.assertEqual(X.shape[0], 3)  # First student has 3 tests
        
        # Check that X contains valid values
        self.assertTrue(torch.all(X[:, 0] >= 1))  # All protocols should be >= 1
        self.assertTrue(torch.all(X[:, 1] >= 0))  # All days should be >= 0
        self.assertTrue(X.dtype == torch.float)  # Should be float tensor
    
    def test_single_student_data(self):
        """Test getting individual student data"""
        X, y = self.dataset[0]
        
        # Check shapes
        self.assertEqual(X.shape[0], y.shape[0])  # Same number of samples
        self.assertEqual(X.shape[1], 2)  # Two features
        
        # Check data types
        self.assertEqual(X.dtype, torch.float)
        self.assertEqual(y.dtype, torch.float)
        
        # Check value ranges
        self.assertTrue(torch.all(X[:, 0] >= 1))  # Protocols start at 1
        self.assertTrue(torch.all(X[:, 1] >= 0))  # Days since start >= 0
        self.assertTrue(torch.all(y >= 0) and torch.all(y <= 1))  # Accuracy between 0 and 1
    
    def test_days_since_start(self):
        """Test that days_since_start is calculated correctly"""
        student_data = self.dataset.get_student_data(1)  # Get first student
        self.assertTrue(all(student_data['days_since_start'] >= 0))
        # Check that days increase with test_time
        days = student_data['days_since_start'].values
        self.assertTrue(all(days[i] <= days[i+1] for i in range(len(days)-1)))
    
    def tearDown(self):
        if os.path.exists(self.test_csv):
            os.remove(self.test_csv)
        if os.path.exists(self.student_csv):
            os.remove(self.student_csv)

if __name__ == '__main__':
    unittest.main(verbosity=2) 