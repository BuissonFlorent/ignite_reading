import unittest
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.model.asymptotic import AsymptoticFunctions

class TestAsymptoticFunctions(unittest.TestCase):
    def setUp(self):
        self.asymptotic = AsymptoticFunctions()
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
    
    def test_exponential_basic(self):
        """Test basic exponential function behavior"""
        x = torch.tensor([1.0, 2.0, 3.0])
        y = self.asymptotic.exponential(x)
        
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.shape, x.shape)
    
    def test_exponential_parameters(self):
        """Test exponential function with different parameters"""
        x = torch.tensor([1.0, 2.0, 3.0])
        
        # Test with different A values
        y1 = self.asymptotic.exponential(x, A=2.0)
        y2 = self.asymptotic.exponential(x, A=0.5)
        self.assertTrue(torch.all(y1 > y2))
        
        # Test with different k values
        y1 = self.asymptotic.exponential(x, k=2.0)
        y2 = self.asymptotic.exponential(x, k=0.5)
        self.assertTrue(torch.all(y1 > y2))
        
        # Test with different b values
        y1 = self.asymptotic.exponential(x, b=0.5)
        y2 = self.asymptotic.exponential(x, b=0.1)
        self.assertTrue(torch.all(y1 > y2))
    
    def test_exponential_vectorized(self):
        """Test exponential function with vectorized parameters"""
        x = torch.tensor([1.0, 2.0, 3.0])
        k = torch.tensor([0.5, 1.0, 2.0])
        
        # Test with vectorized k
        y = self.asymptotic.exponential(x, k=k)
        self.assertEqual(y.shape, x.shape)
        
        # Test with vectorized A
        A = torch.tensor([0.5, 1.0, 1.5])
        y = self.asymptotic.exponential(x, A=A)
        self.assertEqual(y.shape, x.shape)
    
    def test_exponential_monotonicity(self):
        """Test that exponential function is monotonically increasing"""
        x = torch.linspace(0, 10, 100)
        y = self.asymptotic.exponential(x)
        
        # Check that differences are non-negative
        differences = y[1:] - y[:-1]
        self.assertTrue(torch.all(differences >= 0))
    
    def test_exponential_bounds(self):
        """Test that exponential function respects bounds"""
        x = torch.linspace(0, 10, 100)
        A = 0.5
        b = 0.2
        
        y = self.asymptotic.exponential(x, A=A, b=b)
        
        # Check lower bound
        self.assertTrue(torch.all(y >= b))
        
        # Check upper bound
        self.assertTrue(torch.all(y <= A + b))

if __name__ == '__main__':
    unittest.main(verbosity=2)