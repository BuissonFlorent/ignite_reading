import unittest
import torch
import numpy as np
import os
import sys
import pandas as pd
import tempfile
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.examples.train_full import train_full_dataset
from src.examples.train_simple import save_parameters
from src.model.asymptotic_model import AsymptoticModel

class TestTrainFull(unittest.TestCase):
    def setUp(self):
        """Create a small synthetic dataset for testing"""
        self.temp_files = []
        
        # Create test data
        test_data = {
            'student_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'test_time': ['2023-01-01', '2023-01-02', '2023-01-03',
                         '2023-01-01', '2023-01-02', '2023-01-03',
                         '2023-01-01', '2023-01-02', '2023-01-03'],
            'protocol': [1, 2, 3, 1, 2, 3, 1, 2, 3],
            'accuracy': [0.5, 0.6, 0.7, 0.4, 0.5, 0.6, 0.3, 0.4, 0.5]
        }
        
        student_data = {
            'student_id': [1, 2, 3],
            'program_start_date': ['2023-01-01', '2023-01-01', '2023-01-01'],
            'grade_level': ['3rd', '4th', '3rd'],
            'school_name': ['Test School', 'Test School', 'Test School']
        }
        
        # Create temporary files
        self.test_csv = self._create_temp_file('test_data.csv')
        self.student_csv = self._create_temp_file('student_data.csv')
        
        pd.DataFrame(test_data).to_csv(self.test_csv, index=False)
        pd.DataFrame(student_data).to_csv(self.student_csv, index=False)
    
    def _create_temp_file(self, prefix):
        """Create a temporary file and track it for cleanup"""
        fd, path = tempfile.mkstemp(prefix=prefix)
        os.close(fd)
        self.temp_files.append(path)
        return path
    
    def test_parameter_reuse(self):
        """Test that parameters can be reused between simple and full training"""
        # First train with simple training
        model = AsymptoticModel()
        with torch.no_grad():
            model.beta_protocol.data = torch.tensor([1.0])
            model.beta_time.data = torch.tensor([0.1])
            model.b.data = torch.tensor([0.0])
        
        # Save parameters
        param_file = save_parameters(model, save_dir='test_params')
        self.temp_files.append(param_file)
        
        # Train full dataset with these parameters
        model_full, loss = train_full_dataset(
            self.test_csv,
            self.student_csv,
            num_epochs=5,
            init_params=param_file
        )
        
        self.assertIsNotNone(model_full)
        self.assertIsNotNone(loss)
        self.assertLess(loss, float('inf'))
    
    def test_early_stopping(self):
        """Test early stopping in full training"""
        # Train with strict early stopping
        model, loss = train_full_dataset(
            self.test_csv,
            self.student_csv,
            num_epochs=20,
            patience=1,
            min_delta=1.0
        )
        
        self.assertIsNotNone(model)
        self.assertIsNotNone(loss)
        self.assertLess(loss, float('inf'))
        self.assertGreater(loss, 0)
    
    def test_validation_split(self):
        """Test that validation split works correctly"""
        # Train with default settings
        model, loss = train_full_dataset(
            self.test_csv,
            self.student_csv,
            num_epochs=5
        )
        
        self.assertIsNotNone(model)
        self.assertIsNotNone(loss)
    
    def test_plot_results(self):
        """Test that plotting functionality works"""
        # Train model
        model, loss = train_full_dataset(
            self.test_csv,
            self.student_csv,
            num_epochs=5
        )
        
        # Create plot
        from src.examples.train_full import plot_results
        from src.data.dataset import ReadingScoreDataset
        
        dataset = ReadingScoreDataset(self.test_csv, self.student_csv)
        plot_path = plot_results(dataset, model, save_dir='test_plots')
        
        # Verify plot was created
        self.assertTrue(os.path.exists(plot_path))
        self.temp_files.append(plot_path)
        
        # Clean up plot directory
        try:
            os.rmdir('test_plots')
        except OSError:
            pass
    
    def tearDown(self):
        """Clean up temporary files"""
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except OSError:
                pass
        
        # Clean up test_params directory if empty
        try:
            os.rmdir('test_params')
        except OSError:
            pass

if __name__ == '__main__':
    unittest.main(verbosity=2) 