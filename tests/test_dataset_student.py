import unittest
import sys
import os
import pandas as pd
import tempfile
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data.dataset import ReadingScoreDataset

class TestDatasetStudent(unittest.TestCase):
    def setUp(self):
        """Create a minimal test dataset"""
        self.temp_files = []
        
        # Create test data
        test_data = {
            'student_id': [1, 1, 2, 2],
            'test_time': ['2023-01-01', '2023-01-02', '2023-01-01', '2023-01-02'],
            'protocol': [1, 2, 1, 2],
            'accuracy': [0.5, 0.6, 0.4, 0.5]
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
    
    def test_student_data_access(self):
        """Test basic student data access"""
        # Check number of students
        self.assertEqual(len(self.dataset.student_ids), 2)
        
        # Check student IDs
        self.assertTrue(1 in self.dataset.student_ids)
        self.assertTrue(2 in self.dataset.student_ids)
        
        # Check sequence counts
        student1_sequences = [i for i in range(len(self.dataset)) 
                            if self.dataset.data.iloc[i]['student_id'] == 1]
        self.assertEqual(len(student1_sequences), 2)
    
    def test_student_data_integrity(self):
        """Test that student data is correctly linked and accessible"""
        # Get data for first sequence
        student_data = self.dataset.get_student_data(0)
        
        # Check all expected fields are present
        self.assertIn('grade_level', student_data.columns)
        self.assertIn('school_name', student_data.columns)
        self.assertIn('program_start_date', student_data.columns)
        
        # Check data types
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(student_data['program_start_date']))
        
        # Check values match our test data
        self.assertEqual(student_data['grade_level'].iloc[0], '3rd')
        self.assertEqual(student_data['school_name'].iloc[0], 'Test School')
    
    def test_sequence_ordering(self):
        """Test that sequences are properly ordered by time"""
        # Get sequences for student 1
        student1_sequences = [i for i in range(len(self.dataset)) 
                            if self.dataset.data.iloc[i]['student_id'] == 1]
        
        # Check timestamps are in order
        times = [self.dataset.data.iloc[i]['test_time'] for i in student1_sequences]
        self.assertEqual(times, sorted(times))
        
        # Check protocols are sequential
        protocols = [self.dataset.data.iloc[i]['protocol'] for i in student1_sequences]
        self.assertEqual(protocols, [1, 2])
    
    def test_data_consistency(self):
        """Test that data is consistent across different access methods"""
        # Get first sequence data
        X, y = self.dataset[0]
        
        # Check tensor shapes
        self.assertEqual(X.shape[1], 2)  # protocol and days
        self.assertEqual(y.shape[0], 1)  # single accuracy value
        
        # Check values match original data
        self.assertEqual(float(y), 0.5)  # first accuracy value
        self.assertEqual(int(X[0][0]), 1)  # first protocol number
    
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
