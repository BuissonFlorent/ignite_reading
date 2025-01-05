import unittest
import torch
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data.dataset import ReadingScoreDataset

class TestReadingScoreDataset(unittest.TestCase):
    def setUp(self):
        self.temp_files = []
        # Create test files and track them
        self.test_csv = self._create_temp_file('test_data.csv')
        self.student_csv = self._create_temp_file('student_data.csv')
        # Create a small test dataset for reading scores
        test_data = {
            'student_id': [1, 1, 1, 2, 2, 3],  # Added student 3 with no matching data
            'test_time': ['2023-01-01', '2023-01-02', '2023-01-03', 
                         '2023-01-01', '2023-01-02', '2023-01-01'],
            'protocol': [1, 1, 2, 1, 2, 1],
            'accuracy': [0.5, 0.6, 0.8, 0.4, 0.7, 0.5]
        }
        
        # Create a small test dataset for student info
        student_data = {
            'student_id': [1, 2],  # Deliberately exclude student 3
            'program_start_date': ['2023-01-01', '2023-01-01'],
            'grade_level': ['3rd', '4th'],
            'school_name': ['Test School', 'Test School']
        }
        
        pd.DataFrame(test_data).to_csv(self.test_csv, index=False)
        pd.DataFrame(student_data).to_csv(self.student_csv, index=False)
        
        self.dataset = ReadingScoreDataset(self.test_csv, self.student_csv)
    
    def _create_temp_file(self, filename):
        """Create a temporary file with a unique name"""
        import tempfile
        fd, path = tempfile.mkstemp(prefix=filename)
        os.close(fd)
        self.temp_files.append(path)
        return path
    
    def test_dataset_loading(self):
        """Test basic dataset loading and inner join behavior"""
        self.assertIsInstance(self.dataset.data, pd.DataFrame)
        self.assertEqual(len(self.dataset.student_ids), 2)  # Only 2 students should remain
        
        # Verify student 3 was excluded
        self.assertNotIn(3, self.dataset.student_ids)
        
        # Check that all necessary columns exist
        self.assertIn('program_start_date', self.dataset.data.columns)
        self.assertIn('days_since_start', self.dataset.data.columns)
        
        # Verify no missing values in key columns
        self.assertFalse(self.dataset.data['days_since_start'].isna().any())
    
    def test_dataset_getitem(self):
        """Test that __getitem__ returns correct structure"""
        X, y = self.dataset[0]  # Get first student
        self.assertIsInstance(X, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        
        # Check X shape (should be 2D with protocol and days)
        self.assertEqual(X.shape[1], 2)  # Two features
        self.assertEqual(X.shape[0], 3)  # First student has 3 tests
        
        # Verify no NaN values in features
        self.assertFalse(torch.isnan(X).any())
        
        # Check that X contains valid values
        self.assertTrue(torch.all(X[:, 0] >= 1))  # All protocols should be >= 1
        self.assertTrue(torch.all(X[:, 1] >= 0))  # All days should be >= 0
        self.assertTrue(X.dtype == torch.float)  # Should be float tensor
    
    def test_days_calculation(self):
        """Test that days_since_start is calculated correctly"""
        student_data = self.dataset.get_student_data(1)  # Get first student
        
        # Days should start at 0 and increase
        days = student_data['days_since_start'].values
        self.assertEqual(days[0], 0)  # First day should be 0
        self.assertTrue(all(days[i] <= days[i+1] for i in range(len(days)-1)))
        
        # Verify specific day differences
        self.assertEqual(days[1], 1.0)  # One day difference
        self.assertEqual(days[2], 2.0)  # Two days difference
    
    def test_incomplete_data(self):
        """Test that students with missing data are properly excluded"""
        # Create datasets with missing information
        test_data_missing = {
            'student_id': [4, 4],
            'test_time': ['2023-01-01', '2023-01-02'],
            'protocol': [1, 2],
            'accuracy': [0.5, 0.6]
        }
        student_data_missing = {
            'student_id': [5],  # Different student
            'program_start_date': ['2023-01-01']
        }
        
        # Save to temporary files
        temp_test = 'temp_test.csv'
        temp_student = 'temp_student.csv'
        pd.DataFrame(test_data_missing).to_csv(temp_test, index=False)
        pd.DataFrame(student_data_missing).to_csv(temp_student, index=False)
        
        # Load dataset
        dataset = ReadingScoreDataset(temp_test, temp_student)
        
        # Verify that no students remain after inner join
        self.assertEqual(len(dataset), 0)
        
        # Clean up
        os.remove(temp_test)
        os.remove(temp_student)
    
    def tearDown(self):
        """Clean up all temporary files"""
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except OSError:
                pass

if __name__ == '__main__':
    unittest.main(verbosity=2) 