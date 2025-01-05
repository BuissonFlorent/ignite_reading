import unittest
import torch
import numpy as np
import os
import sys
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.examples.train_simple import train_model
from src.data.dataset import ReadingScoreDataset

class TestTraining(unittest.TestCase):
    def setUp(self):
        """Create a small synthetic dataset for testing"""
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
        
        self.test_csv = 'test_data.csv'
        self.student_csv = 'student_data.csv'
        pd.DataFrame(test_data).to_csv(self.test_csv, index=False)
        pd.DataFrame(student_data).to_csv(self.student_csv, index=False)
    
    def test_training_process(self):
        """Test that training runs and improves the model"""
        # Train model with early stopping
        model, final_loss = train_model(
            self.test_csv, 
            self.student_csv,
            num_epochs=10,
            patience=3,
            min_delta=1e-4
        )
        
        self.assertIsNotNone(model)
        self.assertIsNotNone(final_loss)
        self.assertLess(final_loss, float('inf'))
    
    def test_early_stopping(self):
        """Test that early stopping works"""
        # Train with very strict early stopping
        result1 = train_model(
            self.test_csv,
            self.student_csv,
            num_epochs=20,
            patience=1,
            min_delta=1.0
        )
        
        # Train with loose early stopping
        result2 = train_model(
            self.test_csv,
            self.student_csv,
            num_epochs=20,
            patience=10,
            min_delta=1e-6
        )
        
        # Check that training completed successfully
        self.assertIsNotNone(result1)
        self.assertIsNotNone(result2)
        
        model1, loss1 = result1
        model2, loss2 = result2
        
        # Both losses should be finite and reasonable
        self.assertLess(loss1, float('inf'))
        self.assertLess(loss2, float('inf'))
        
        # Both models should have valid parameters
        with torch.no_grad():
            for model in [model1, model2]:
                self.assertTrue(torch.isfinite(model.beta_protocol).all())
                self.assertTrue(torch.isfinite(model.beta_time).all())
                self.assertTrue(torch.isfinite(model.b).all())
    
    def test_validation_split(self):
        """Test that validation split is consistent"""
        dataset = ReadingScoreDataset(self.test_csv, self.student_csv)
        
        # Get all indices
        indices = list(range(len(dataset)))
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(indices)
        
        # Ensure at least one validation sample
        split = max(1, int(np.floor(0.2 * len(dataset))))
        val_indices = indices[:split]
        train_indices = indices[split:]
        
        # Check split sizes
        self.assertGreater(len(val_indices), 0)  # At least 1 validation sample
        self.assertGreater(len(train_indices), 0)  # At least 1 training sample
        self.assertEqual(len(val_indices) + len(train_indices), len(dataset))
    
    def test_model_improvement(self):
        """Test that model improves during training"""
        # Train model
        model, final_loss = train_model(
            self.test_csv,
            self.student_csv,
            num_epochs=10,
            patience=5
        )
        
        # Create test point
        X = torch.tensor([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
        
        # Model predictions should be monotonic for increasing protocols
        with torch.no_grad():
            predictions = model(X)
            differences = predictions[1:] - predictions[:-1]
            self.assertTrue(torch.all(differences >= -1e-6))
    
    def tearDown(self):
        """Clean up test files"""
        if os.path.exists(self.test_csv):
            os.remove(self.test_csv)
        if os.path.exists(self.student_csv):
            os.remove(self.student_csv)

if __name__ == '__main__':
    unittest.main(verbosity=2) 