import unittest
import torch
import sys
import os
import json
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.model.asymptotic_model import AsymptoticModel
from src.examples.train_simple import save_parameters, load_parameters

class TestParameterHandling(unittest.TestCase):
    def setUp(self):
        self.model = AsymptoticModel()
        self.test_dir = 'test_params'
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Set specific parameter values for testing
        with torch.no_grad():
            self.model.beta_protocol.data = torch.tensor([1.5])
            self.model.beta_time.data = torch.tensor([0.3])
            self.model.b.data = torch.tensor([-0.5])  # Will be transformed by sigmoid
    
    def test_parameter_saving(self):
        """Test that parameters are correctly saved"""
        # Save parameters
        param_file = save_parameters(self.model, save_dir=self.test_dir)
        
        # Check file exists
        self.assertTrue(os.path.exists(param_file))
        
        # Load and check content
        with open(param_file, 'r') as f:
            params = json.load(f)
        
        # Check all parameters are present
        self.assertIn('beta_protocol', params)
        self.assertIn('beta_time', params)
        self.assertIn('b', params)
        
        # Check values match (within floating point precision)
        self.assertAlmostEqual(params['beta_protocol'], 1.5, places=6)
        self.assertAlmostEqual(params['beta_time'], 0.3, places=6)
        self.assertAlmostEqual(params['b'], -0.5, places=6)
    
    def test_parameter_loading(self):
        """Test that parameters are correctly loaded"""
        # Save parameters
        param_file = save_parameters(self.model, save_dir=self.test_dir)
        
        # Create new model with different parameters
        new_model = AsymptoticModel()
        
        # Load parameters
        new_model = load_parameters(new_model, param_file)
        
        # Check values match
        with torch.no_grad():
            self.assertAlmostEqual(
                new_model.beta_protocol.item(),
                self.model.beta_protocol.item(),
                places=6
            )
            self.assertAlmostEqual(
                new_model.beta_time.item(),
                self.model.beta_time.item(),
                places=6
            )
            self.assertAlmostEqual(
                new_model.b.item(),
                self.model.b.item(),
                places=6
            )
    
    def test_parameter_validation(self):
        """Test parameter validation"""
        # Create invalid parameter file
        invalid_params = {
            'beta_protocol': 'not_a_number',
            'beta_time': 0.3,
            'b': -0.5
        }
        invalid_file = os.path.join(self.test_dir, 'invalid_params.json')
        with open(invalid_file, 'w') as f:
            json.dump(invalid_params, f)
        
        # Check that loading invalid parameters raises error
        with self.assertRaises(ValueError):
            load_parameters(self.model, invalid_file)
        
        # Create incomplete parameter file
        incomplete_params = {
            'beta_protocol': 1.5,
            'beta_time': 0.3
            # missing 'b'
        }
        incomplete_file = os.path.join(self.test_dir, 'incomplete_params.json')
        with open(incomplete_file, 'w') as f:
            json.dump(incomplete_params, f)
        
        # Check that loading incomplete parameters raises error
        with self.assertRaises(KeyError):
            load_parameters(self.model, incomplete_file)
    
    def tearDown(self):
        # Clean up test files
        if os.path.exists(self.test_dir):
            for file in os.listdir(self.test_dir):
                os.remove(os.path.join(self.test_dir, file))
            os.rmdir(self.test_dir)

if __name__ == '__main__':
    unittest.main(verbosity=2) 