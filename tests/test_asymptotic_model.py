import unittest
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.model.asymptotic_model import AsymptoticModel

class TestAsymptoticModel(unittest.TestCase):
    def setUp(self):
        self.model = AsymptoticModel()
        
        # Create sample data similar to what our DataLoader provides
        self.days = torch.linspace(0, 100, 50)  # 100 days of data
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
    
    def test_model_initialization(self):
        """Test if model initializes correctly"""
        self.assertIsInstance(self.model, torch.nn.Module)
        self.assertTrue(hasattr(self.model, 'k'))
        self.assertTrue(hasattr(self.model, 'b'))
    
    def test_forward_pass(self):
        """Test basic forward pass"""
        y = self.model(self.days)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.shape, self.days.shape)
    
    def test_parameter_constraints(self):
        """Test if parameters respect constraints"""
        with torch.no_grad():
            # Get transformed parameters
            k = torch.exp(self.model.k)
            b = torch.sigmoid(self.model.b)
            
            # Initial performance should be between 0 and 1
            self.assertTrue(torch.all(b >= 0) and torch.all(b <= 1))
            
            # Learning rate should be positive
            self.assertTrue(torch.all(k > 0))
    
    def test_learning_curve_properties(self):
        """Test properties of the learning curve"""
        with torch.no_grad():
            # Test at different time points
            t0 = torch.tensor([0.0])  # Start
            t1 = torch.tensor([1000.0])  # Long time
            
            y0 = self.model(t0)
            y1 = self.model(t1)
            
            # Initial value should equal transformed b parameter
            b = torch.sigmoid(self.model.b)
            self.assertAlmostEqual(y0.item(), b.item(), places=5)
            
            # Final value can reach 1.0 (perfect score achievable)
            self.assertLessEqual(y1.item(), 1.0)
            self.assertGreater(y1.item(), y0.item())  # Should show improvement
    
    def test_monotonicity(self):
        """Test that predictions are monotonically increasing"""
        times = torch.linspace(0, 100, 10)
        with torch.no_grad():
            predictions = self.model(times)
            differences = predictions[1:] - predictions[:-1]
            self.assertTrue(torch.all(differences >= 0))

if __name__ == '__main__':
    unittest.main(verbosity=2) 