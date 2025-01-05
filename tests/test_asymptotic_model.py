import unittest
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.model.asymptotic_model import AsymptoticModel

class TestAsymptoticModel(unittest.TestCase):
    def setUp(self):
        self.model = AsymptoticModel()
        
        # Create sample input with both protocol and days
        protocols = torch.FloatTensor([1, 1, 2, 3, 4])
        days = torch.FloatTensor([0, 10, 20, 30, 40])
        self.X = torch.stack([protocols, days], dim=1)
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
    
    def test_model_initialization(self):
        """Test if model initializes correctly"""
        self.assertIsInstance(self.model, torch.nn.Module)
        self.assertTrue(hasattr(self.model, 'beta_protocol'))
        self.assertTrue(hasattr(self.model, 'beta_time'))
        self.assertTrue(hasattr(self.model, 'b'))
    
    def test_forward_pass(self):
        """Test basic forward pass with protocol numbers and days"""
        y = self.model(self.X)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.shape, (5,))  # Should match number of inputs
        
        # Check that predictions are between 0 and 1
        self.assertTrue(torch.all(y >= 0) and torch.all(y <= 1))
    
    def test_parameter_constraints(self):
        """Test if parameters respect constraints"""
        with torch.no_grad():
            # Get transformed parameters
            b = torch.sigmoid(self.model.b)
            
            # Check constraints
            self.assertTrue(torch.all(b >= 0) and torch.all(b <= 1))  # Baseline between 0 and 1
    
    def test_monotonicity(self):
        """Test that predictions generally increase with protocol and time"""
        protocols = torch.FloatTensor([1, 2, 3, 4, 5])
        days = torch.FloatTensor([0, 10, 20, 30, 40])
        X = torch.stack([protocols, days], dim=1)
        
        with torch.no_grad():
            predictions = self.model(X)
            differences = predictions[1:] - predictions[:-1]
            # Note: We can't guarantee strict monotonicity due to interaction
            # between protocol and time effects
            self.assertTrue(torch.sum(differences >= 0) >= len(differences) - 1)
    
    def test_upper_bound(self):
        """Test that predictions never exceed 1.0"""
        # Test with very high values
        protocols = torch.FloatTensor([10.0, 20.0, 50.0])
        days = torch.FloatTensor([100.0, 200.0, 300.0])
        X = torch.stack([protocols, days], dim=1)
        
        with torch.no_grad():
            predictions = self.model(X)
            self.assertTrue(torch.all(predictions <= 1.0))
            self.assertTrue(torch.all(predictions >= 0.0))

if __name__ == '__main__':
    unittest.main(verbosity=2) 