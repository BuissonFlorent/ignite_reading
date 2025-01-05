import unittest
import sys
import os
import pandas as pd
import torch
import tempfile
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data.dataset import ReadingScoreDataset
from src.model.asymptotic_model import AsymptoticModel
from src.examples.visualize_student import find_student_indices

class TestVisualization(unittest.TestCase):
    def setUp(self):
        """Create a small synthetic dataset for testing"""
        self.temp_files = []
        
        # Create test data with known student IDs
        test_data = {
            'student_id': [1, 1, 1, 2, 2, 2],
            'test_time': ['2023-01-01', '2023-01-02', '2023-01-03',
                         '2023-01-01', '2023-01-02', '2023-01-03'],
            'protocol': [1, 2, 3, 1, 2, 3],
            'accuracy': [0.5, 0.6, 0.7, 0.4, 0.5, 0.6]
        }
        
        student_data = {
            'student_id': [1, 2],
            'program_start_date': ['2023-01-01', '2023-01-01'],
            'grade_level': ['3rd', '4th'],
            'school_name': ['Test School', 'Test School']
        }
        
        # Create temporary files
        self.test_csv = self._create_temp_file('test_data.csv')
        self.student_csv = self._create_temp_file('student_data.csv')
        
        pd.DataFrame(test_data).to_csv(self.test_csv, index=False)
        pd.DataFrame(student_data).to_csv(self.student_csv, index=False)
        
        # Create dataset
        self.dataset = ReadingScoreDataset(self.test_csv, self.student_csv)
    
    def _create_temp_file(self, prefix):
        """Create a temporary file and track it for cleanup"""
        fd, path = tempfile.mkstemp(prefix=prefix)
        os.close(fd)
        self.temp_files.append(path)
        return path
    
    def test_dataset_structure(self):
        """Test that we can access student data correctly"""
        # Get first sequence
        X, y = self.dataset[0]
        
        # Check data structure
        self.assertEqual(X.shape[1], 2)  # protocol and days
        self.assertTrue(isinstance(y, torch.Tensor))
        
        # Get student data for first sequence
        student_data = self.dataset.test_data.iloc[0]
        self.assertEqual(student_data['student_id'], 1)
    
    def test_find_student_indices(self):
        """Test finding indices for a specific student"""
        # Test existing student
        indices = find_student_indices(self.dataset, 1)
        self.assertEqual(len(indices), 2)  # Should find 2 sequences for student 1
        
        # Verify all indices point to the same student
        for idx in indices:
            student_id = self.dataset.data.iloc[idx]['student_id']
            self.assertEqual(student_id, 1)
        
        # Test non-existent student
        indices = find_student_indices(self.dataset, 999)
        self.assertEqual(len(indices), 0)  # Should find no sequences
        
        # Test edge case - student 2
        indices = find_student_indices(self.dataset, 2)
        self.assertEqual(len(indices), 2)  # Should find 2 sequences for student 2
        self.assertTrue(all(self.dataset.data.iloc[idx]['student_id'] == 2 
                           for idx in indices))
    
    def tearDown(self):
        """Clean up temporary files"""
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except OSError:
                pass

if __name__ == '__main__':
    unittest.main(verbosity=2)