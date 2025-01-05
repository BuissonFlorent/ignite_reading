import unittest
import sys
import os
import pandas as pd
import torch
import tempfile
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data.dataset import ReadingScoreDataset
from src.model.asymptotic_model import AsymptoticModel
from src.examples.visualize_student import find_student_indices, load_last_model, load_parameters_from_file, create_model_from_parameters

class TestVisualizeStudent(unittest.TestCase):
    def setUp(self):
        """Create a minimal test dataset and model parameters"""
        self.temp_files = []
        
        # Create test data
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
        
        # Create temporary model parameters file
        self.results_dir = tempfile.mkdtemp()
        self.temp_files.append(self.results_dir)
        self.param_file = os.path.join(self.results_dir, 'model_params_20230101.txt')
        with open(self.param_file, 'w') as f:
            f.write('learning_rate (k): 0.1\n')
            f.write('initial_level (b): 0.5\n')
    
    def _create_temp_file(self, prefix):
        """Create a temporary file and track it for cleanup"""
        fd, path = tempfile.mkstemp(prefix=prefix)
        os.close(fd)
        self.temp_files.append(path)
        return path
    
    def test_find_student_indices(self):
        """Test finding indices for a specific student"""
        # Test existing student
        indices = find_student_indices(self.dataset, 1)
        self.assertEqual(len(indices), 3)
        
        # Verify all indices point to the same student
        for idx in indices:
            student_id = self.dataset.data.iloc[idx]['student_id']
            self.assertEqual(student_id, 1)
        
        # Test non-existent student
        indices = find_student_indices(self.dataset, 999)
        self.assertEqual(len(indices), 0)
    
    def test_parameter_loading(self):
        """Test loading parameters from file"""
        params = load_parameters_from_file(self.param_file)
        self.assertEqual(params['learning_rate (k)'], 0.1)
        self.assertEqual(params['initial_level (b)'], 0.5)
    
    def test_model_creation(self):
        """Test creating model from parameters"""
        params = {'learning_rate (k)': 0.1, 'initial_level (b)': 0.5}
        model = create_model_from_parameters(params)
        
        with torch.no_grad():
            self.assertAlmostEqual(float(torch.exp(model.k)), 0.1)
            self.assertAlmostEqual(float(torch.sigmoid(model.b)), 0.5)
    
    def test_load_last_model(self):
        """Test full model loading pipeline"""
        model = load_last_model(self.results_dir)
        self.assertIsInstance(model, AsymptoticModel)
    
    def tearDown(self):
        """Clean up temporary files and directories"""
        for path in self.temp_files:
            try:
                if os.path.isdir(path):
                    os.rmdir(path)
                elif os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass

if __name__ == '__main__':
    unittest.main(verbosity=2) 