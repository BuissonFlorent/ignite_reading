import unittest
import sys
import os
import pandas as pd
import torch
import tempfile
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data.dataset import ReadingScoreDataset
from src.model.asymptotic_model import AsymptoticModel

class TestLearningCurves(unittest.TestCase):
    def setUp(self):
        """Create a minimal test dataset and model"""
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
        
        # Create dataset and model
        self.dataset = ReadingScoreDataset(self.test_csv, self.student_csv)
        self.model = AsymptoticModel()
    
    def _create_temp_file(self, prefix):
        """Create a temporary file and track it for cleanup"""
        fd, path = tempfile.mkstemp(prefix=prefix)
        os.close(fd)
        self.temp_files.append(path)
        return path
    
    def test_plot_creation(self):
        """Test that we can create a basic plot"""
        # Create temporary directory for plots
        plot_dir = tempfile.mkdtemp()
        self.temp_files.append(plot_dir)
        
        # Create plot
        fig, ax = plt.subplots()
        X, y = self.dataset[0]
        
        # Plot actual values
        ax.scatter([0], y.numpy(), label='Actual')
        
        # Get prediction
        with torch.no_grad():
            pred = self.model(X)
        ax.plot([0], pred.numpy(), '--', label='Predicted')
        
        # Save plot
        plot_path = os.path.join(plot_dir, 'test_plot.png')
        plt.savefig(plot_path)
        plt.close()
        
        # Verify plot was created
        self.assertTrue(os.path.exists(plot_path))
    
    def test_data_preparation(self):
        """Test that data is properly prepared for plotting"""
        # Get sequence data
        X, y = self.dataset[0]
        
        # Check data structure
        self.assertEqual(X.shape[1], 2)  # protocol and days
        self.assertEqual(y.shape[0], 1)  # single accuracy value
        
        # Check value ranges
        self.assertTrue(torch.all(y >= 0) and torch.all(y <= 1))
        self.assertTrue(torch.all(X[:, 0] >= 0))  # protocols
        self.assertTrue(torch.all(X[:, 1] >= 0))  # days
    
    def test_multiple_sequences(self):
        """Test that we can plot multiple sequences"""
        plot_dir = tempfile.mkdtemp()
        self.temp_files.append(plot_dir)
        
        fig, ax = plt.subplots()
        
        # Plot sequences for first student
        student_sequences = [i for i in range(len(self.dataset)) 
                           if self.dataset.data.iloc[i]['student_id'] == 1]
        
        for idx in student_sequences:
            X, y = self.dataset[idx]
            ax.scatter(X[:, 1].numpy(), y.numpy(), label=f'Sequence {idx}')
            
            with torch.no_grad():
                pred = self.model(X)
            ax.plot(X[:, 1].numpy(), pred.numpy(), '--')
        
        plot_path = os.path.join(plot_dir, 'test_multiple.png')
        plt.savefig(plot_path)
        plt.close()
        
        self.assertTrue(os.path.exists(plot_path))
    
    def test_plot_formatting(self):
        """Test plot formatting and labels"""
        fig, ax = plt.subplots()
        X, y = self.dataset[0]
        
        # Plot data
        ax.scatter([0], y.numpy(), label='Actual')
        
        # Set labels and title
        ax.set_xlabel('Days since start')
        ax.set_ylabel('Accuracy')
        ax.set_title('Learning Curve')
        
        # Check plot properties
        self.assertEqual(ax.get_xlabel(), 'Days since start')
        self.assertEqual(ax.get_ylabel(), 'Accuracy')
        self.assertEqual(ax.get_title(), 'Learning Curve')
        
        plt.close()
    
    def test_prediction_range(self):
        """Test that model predictions are in valid range"""
        X, _ = self.dataset[0]
        
        with torch.no_grad():
            pred = self.model(X)
        
        # Check prediction range
        self.assertTrue(torch.all(pred >= 0) and torch.all(pred <= 1))
    
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
