import unittest
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.model.asymptotic_model import AsymptoticModel

class TestAsymptoticModel(unittest.TestCase):
    def setUp(self):
        self.model = AsymptoticModel()
        
        # Create sample protocol sequence
        self.protocols = torch.FloatTensor([1, 1, 2, 3, 4])  # Example protocol sequence
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
    
    def test_model_initialization(self):
        """Test if model initializes correctly"""
        self.assertIsInstance(self.model, torch.nn.Module)
        self.assertTrue(hasattr(self.model, 'k'))
        self.assertTrue(hasattr(self.model, 'b'))
    
    def test_forward_pass(self):
        """Test basic forward pass with protocol numbers"""
        y = self.model(self.protocols)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.shape, self.protocols.shape)
        
        # Check that predictions are between 0 and 1
        self.assertTrue(torch.all(y >= 0) and torch.all(y <= 1))
    
    def test_parameter_constraints(self):
        """Test if parameters respect constraints"""
        with torch.no_grad():
            # Get transformed parameters
            k = torch.exp(self.model.k)
            b = torch.sigmoid(self.model.b)
            
            # Check constraints
            self.assertTrue(torch.all(k > 0))  # Learning rate should be positive
            self.assertTrue(torch.all(b >= 0) and torch.all(b <= 1))  # Baseline between 0 and 1
    
    def test_monotonicity(self):
        """Test that predictions increase with protocol number"""
        protocols = torch.FloatTensor([1, 2, 3, 4, 5])
        with torch.no_grad():
            predictions = self.model(protocols)
            differences = predictions[1:] - predictions[:-1]
            self.assertTrue(torch.all(differences >= 0))
    
    def test_upper_bound(self):
        """Test that predictions never exceed 1.0"""
        # Test with very high protocol numbers
        high_protocols = torch.FloatTensor([10.0, 20.0, 50.0])
        with torch.no_grad():
            predictions = self.model(high_protocols)
            self.assertTrue(torch.all(predictions <= 1.0))
            self.assertTrue(torch.all(predictions >= 0.0))

if __name__ == '__main__':
    unittest.main(verbosity=2) 