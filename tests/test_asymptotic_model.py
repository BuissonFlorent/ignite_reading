import unittest
import torch
import sys
import os
import tempfile
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.model.asymptotic_model import AsymptoticModel

class TestAsymptoticModel(unittest.TestCase):
    def setUp(self):
        self.model = AsymptoticModel()
        
        # Create sample input with both protocol and days
        self.protocols = torch.FloatTensor([1, 2, 3, 4, 5])
        self.days = torch.FloatTensor([0, 10, 20, 30, 40])
        self.X = torch.stack([self.protocols, self.days], dim=1)
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        """Create test parameter file"""
        self.temp_files = []
        self.model_dir = tempfile.mkdtemp()
        self.temp_files.append(self.model_dir)
        
        # Create test parameter file with correct parameter names
        self.param_file = os.path.join(self.model_dir, 'model_params_20230101.txt')
        with open(self.param_file, 'w') as f:
            f.write('beta_protocol: 1.0\n')
            f.write('beta_time: 0.1\n')
            f.write('b: 0.0\n')
    
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
    
    def test_parameter_effects(self):
        """Test how different parameter values affect predictions"""
        # Test with different beta_protocol values
        with torch.no_grad():
            self.model.beta_protocol.data = torch.tensor([1.0])
            y1 = self.model(self.X)
            self.model.beta_protocol.data = torch.tensor([0.1])
            y2 = self.model(self.X)
        
        # Higher beta_protocol should lead to faster learning
        self.assertTrue(torch.all(y1 >= y2))
        
        # Test with different beta_time values
        with torch.no_grad():
            self.model.beta_time.data = torch.tensor([0.1])
            y1 = self.model(self.X)
            self.model.beta_time.data = torch.tensor([0.0])
            y2 = self.model(self.X)
        
        # Positive beta_time should lead to faster learning over time
        self.assertTrue(torch.all(y1[1:] >= y2[1:]))  # Compare all but first day
    
    def test_monotonicity(self):
        """Test that predictions generally increase with protocol and time"""
        y = self.model(self.X)
        differences = y[1:] - y[:-1]
        
        # Learning should generally improve over time
        # Note: We allow small negative differences due to interaction effects
        self.assertTrue(torch.sum(differences >= -1e-6) >= len(differences) - 1)
    
    def test_learning_bounds(self):
        """Test that learning stays within reasonable bounds"""
        # Test with very high values
        high_protocols = torch.FloatTensor([10.0, 20.0, 50.0])
        high_days = torch.FloatTensor([100.0, 200.0, 300.0])
        X_high = torch.stack([high_protocols, high_days], dim=1)
        
        y_high = self.model(X_high)
        
        # Predictions should stay between 0 and 1
        self.assertTrue(torch.all(y_high >= 0.0))
        self.assertTrue(torch.all(y_high <= 1.0))
        
        # Test with very low values
        low_protocols = torch.FloatTensor([1.0, 1.0, 1.0])
        low_days = torch.FloatTensor([0.0, 0.0, 0.0])
        X_low = torch.stack([low_protocols, low_days], dim=1)
        
        y_low = self.model(X_low)
        
        # Predictions should stay between 0 and 1
        self.assertTrue(torch.all(y_low >= 0.0))
        self.assertTrue(torch.all(y_low <= 1.0))
    
    def test_from_file(self):
        """Test loading model from file"""
        model = AsymptoticModel.from_file(self.param_file)
        
        with torch.no_grad():
            # Check actual parameters that exist in the model
            self.assertAlmostEqual(float(model.beta_protocol), 1.0)
            self.assertAlmostEqual(float(model.beta_time), 0.1)
            self.assertAlmostEqual(float(model.b), 0.0)
    
    def test_from_directory(self):
        """Test loading most recent model from directory"""
        model = AsymptoticModel.from_directory(self.model_dir)
        self.assertIsInstance(model, AsymptoticModel)
    
    def tearDown(self):
        """Clean up temporary files"""
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