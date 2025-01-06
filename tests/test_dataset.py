import unittest
import torch
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data.dataset import ReadingScoreDataset
from tests.test_data import TestData

class TestReadingScoreDataset(unittest.TestCase):
    def setUp(self):
        """Set up test data using TestData utility"""
        self.temp_files = []
        self.test_csv, self.student_csv = TestData.create_test_csvs(self.temp_files)
        self.dataset = ReadingScoreDataset(self.test_csv, self.student_csv)
    
    def test_single_sequence(self):
        """Test that __getitem__ returns single sequence correctly"""
        X, y = self.dataset[0]
        self.assertIsInstance(X, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(X.shape, (1, 2))
        self.assertFalse(torch.isnan(X).any())
        self.assertTrue(torch.all(X[:, 0] >= 1))  # Protocol >= 1
        self.assertTrue(torch.all(X[:, 1] >= 0))  # Days >= 0
    
    def test_student_tensor(self):
        """Test that get_student_tensor returns all sequences for a student"""
        X, y = self.dataset.get_student_tensor(1)
        self.assertEqual(X.shape, (3, 2))  # 3 sequences, 2 features
        self.assertEqual(y.shape, (3,))    # 3 accuracy values
        self.assertTrue(torch.all(X[:, 1].diff()[1:] > 0))  # Days increase
        self.assertEqual(X[0, 1].item(), 0)  # First day should be 0
    
    def test_student_data(self):
        """Test that get_student_data returns correct DataFrame"""
        print(f"\nIn test_student_data:")
        print(f"Dataset contents:\n{self.dataset.data.head()}")
        student_df = self.dataset.get_student_data(1)
        print(f"Retrieved student data:\n{student_df}")
        self.assertEqual(len(student_df), 3)  # 3 sequences per student
        self.assertTrue(all(student_df['student_id'] == 1))
        self.assertTrue(all(student_df['days_since_start'].diff().dropna() > 0))
        
        # Test invalid student_id
        with self.assertRaises(ValueError):
            self.dataset.get_student_data(999)
    
    def test_get_sequence(self):
        """Test that get_sequence returns correct single sequence"""
        sequence = self.dataset.get_sequence(0)
        self.assertEqual(len(sequence), 1)  # Single sequence
        self.assertIsInstance(sequence, pd.DataFrame)
        
        # Test out of bounds
        with self.assertRaises(IndexError):
            self.dataset.get_sequence(-1)
        with self.assertRaises(IndexError):
            self.dataset.get_sequence(len(self.dataset.data))
    
    def test_dataset_loading(self):
        """Test basic dataset loading and validation"""
        self.assertIsInstance(self.dataset.data, pd.DataFrame)
        self.assertEqual(len(self.dataset.student_ids), 2)  # Default 2 students
        
        # Check required columns
        required_columns = {'student_id', 'test_time', 'protocol', 'accuracy',
                          'program_start_date', 'grade_level', 'school_name'}
        self.assertTrue(all(col in self.dataset.data.columns for col in required_columns))
    
    def test_data_validation(self):
        """Test data validation checks"""
        # Create invalid test data
        test_data, _ = TestData.create_base_test_data(num_students=1, sequences_per_student=1)
        test_data.loc[0, 'accuracy'] = 1.5  # Invalid accuracy > 1
        
        invalid_test_csv = TestData.create_temp_file('invalid_test.csv', self.temp_files)
        test_data.to_csv(invalid_test_csv, index=False)
        
        with self.assertRaises(ValueError):
            ReadingScoreDataset(invalid_test_csv, self.student_csv)
    
    def test_get_sequence_validates_index(self):
        """Test that get_sequence properly validates index input"""
        with self.assertRaises(IndexError):
            self.dataset.get_sequence(-1)
        
        with self.assertRaises(IndexError):
            self.dataset.get_sequence(len(self.dataset.data))
    
    def test_get_student_data_validates_id(self):
        """Test that get_student_data properly validates student_id input"""
        with self.assertRaises(ValueError):
            self.dataset.get_student_data(999999)  # Non-existent student
    
    def test_get_student_data_returns_ordered(self):
        """Test that get_student_data returns time-ordered sequences"""
        student_id = 1  # Using TestData's first student
        student_data = self.dataset.get_student_data(student_id)
        
        # Check if dates are ordered
        dates = student_data['test_time'].tolist()
        self.assertEqual(dates, sorted(dates))
    
    def test_get_sequence_returns_single(self):
        """Test that get_sequence always returns exactly one row"""
        sequence = self.dataset.get_sequence(0)
        self.assertEqual(len(sequence), 1)
        self.assertTrue(isinstance(sequence, pd.DataFrame))
    
    def test_method_definitions(self):
        """Test for duplicate method definitions and show their locations"""
        import inspect
        import src.data.dataset as dataset_module
        
        print("\nChecking method definitions in dataset.py:")
        
        # Get the source code
        source_lines = inspect.getsourcelines(dataset_module)[0]
        
        # Find all method definitions
        method_locations = {}
        for i, line in enumerate(source_lines, 1):
            if line.strip().startswith('def '):
                method_name = line.split('def ')[1].split('(')[0].strip()
                if method_name in method_locations:
                    method_locations[method_name].append(i)
                else:
                    method_locations[method_name] = [i]
        
        # Print all methods and their locations
        for method, lines in method_locations.items():
            print(f"Method '{method}' defined at line(s): {lines}")
            if len(lines) > 1:
                print(f"WARNING: '{method}' is defined multiple times!")
        
        # Assert no duplicates
        duplicates = {m: l for m, l in method_locations.items() if len(l) > 1}
        self.assertEqual(len(duplicates), 0, 
                        f"Found duplicate method definitions: {duplicates}")
    
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