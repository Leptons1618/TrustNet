"""
Test cases for TrustLens models
"""

import unittest
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models import SimpleCNN, TrustModel
from src.trust_methods import compute_entropy, TemperatureScaling


class TestModels(unittest.TestCase):
    """Test cases for model components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cpu')
        self.model = SimpleCNN(num_classes=10)
        self.batch_size = 4
        self.input_tensor = torch.randn(self.batch_size, 3, 32, 32)
    
    def test_simple_cnn_forward(self):
        """Test SimpleCNN forward pass"""
        self.model.eval()
        
        with torch.no_grad():
            output = self.model(self.input_tensor)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 10))
        
        # Check that outputs are not NaN
        self.assertFalse(torch.isnan(output).any())
    
    def test_entropy_computation(self):
        """Test entropy computation"""
        logits = torch.randn(self.batch_size, 10)
        entropy = compute_entropy(logits)
        
        # Check output shape
        self.assertEqual(entropy.shape, (self.batch_size,))
        
        # Check that entropy values are non-negative
        self.assertTrue((entropy >= 0).all())
        
        # Check that entropy is bounded by log(num_classes)
        max_entropy = torch.log(torch.tensor(10.0))
        self.assertTrue((entropy <= max_entropy + 1e-6).all())  # Small tolerance for numerical errors
    
    def test_trust_model_initialization(self):
        """Test TrustModel initialization"""
        trust_model = TrustModel(device=self.device)
        
        # Check that model is properly initialized
        self.assertIsNotNone(trust_model.model)
        self.assertEqual(trust_model.device, self.device)
    
    def test_temperature_scaling(self):
        """Test temperature scaling"""
        temp_scaler = TemperatureScaling()
        logits = torch.randn(self.batch_size, 10)
        labels = torch.randint(0, 10, (self.batch_size,))
        
        # Test forward pass
        calibrated_probs = temp_scaler(logits)
        self.assertEqual(calibrated_probs.shape, (self.batch_size, 10))
        
        # Check probabilities sum to 1
        prob_sums = torch.sum(calibrated_probs, dim=1)
        self.assertTrue(torch.allclose(prob_sums, torch.ones(self.batch_size), atol=1e-6))


class TestTrustMethods(unittest.TestCase):
    """Test cases for trust methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.batch_size = 4
        self.num_classes = 10
        self.logits = torch.randn(self.batch_size, self.num_classes)
    
    def test_entropy_range(self):
        """Test that entropy values are in expected range"""
        entropy = compute_entropy(self.logits)
        
        # Entropy should be between 0 and log(num_classes)
        max_entropy = torch.log(torch.tensor(float(self.num_classes)))
        self.assertTrue((entropy >= 0).all())
        self.assertTrue((entropy <= max_entropy + 1e-6).all())
    
    def test_uniform_distribution_entropy(self):
        """Test entropy for uniform distribution"""
        # Create uniform logits
        uniform_logits = torch.zeros(1, self.num_classes)
        entropy = compute_entropy(uniform_logits)
        
        # Entropy should be close to log(num_classes) for uniform distribution
        expected_entropy = torch.log(torch.tensor(float(self.num_classes)))
        self.assertAlmostEqual(entropy.item(), expected_entropy.item(), places=5)
    
    def test_deterministic_distribution_entropy(self):
        """Test entropy for deterministic distribution"""
        # Create deterministic logits (one very large value)
        det_logits = torch.tensor([[-10, -10, -10, -10, -10, 10, -10, -10, -10, -10]], dtype=torch.float32)
        entropy = compute_entropy(det_logits)
        
        # Entropy should be close to 0 for deterministic distribution
        self.assertLess(entropy.item(), 0.1)


if __name__ == '__main__':
    unittest.main()
