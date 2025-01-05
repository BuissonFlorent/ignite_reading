import unittest
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.model.asymptotic import AsymptoticFunctions

class TestAsymptoticFunctions(unittest.TestCase):
    def setUp(self):
        # Create standard input range for testing
        self.x = torch.linspace(0, 10, 100)
        self.functions = AsymptoticFunctions()
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
    
    def test_exponential_basic(self):
        """Test basic exponential function behavior"""
        A, k, b = 1.0, 0.5, 0.0
        y = self.functions.exponential(self.x, A, k, b)
        
        # Check output is correct type and shape
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.shape, self.x.shape)
        
        # Test asymptotic behavior
        self.assertLess(torch.abs(y[-1] - (A + b)), 0.01)  # Should approach A + b
        self.assertAlmostEqual(y[0].item(), b, places=5)   # Should start at b
    
    def test_parameter_constraints(self):
        """Test parameter constraints"""
        with self.assertRaises(ValueError):
            self.functions.exponential(self.x, -1.0, 0.5, 0.0)  # Negative amplitude
        
        with self.assertRaises(ValueError):
            self.functions.exponential(self.x, 1.0, -0.5, 0.0)  # Negative rate

if __name__ == '__main__':
    unittest.main(verbosity=2)