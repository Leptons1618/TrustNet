"""
Trust-aware model wrapper that integrates various uncertainty and OOD detection methods
"""

import torch
import torch.nn.functional as F
import numpy as np
from .cnn import SimpleCNN
from ..trust_methods.entropy import compute_entropy
from ..trust_methods.odin import odin_predict
from ..trust_methods.mahalanobis import MahalanobisDetector
from ..trust_methods.temperature import TemperatureScaling


class TrustModel:
    """
    Wrapper class that combines CNN with trust and uncertainty methods
    """
    
    def __init__(self, model_path=None, device='cpu'):
        self.device = device
        self.model = SimpleCNN(num_classes=10).to(device)
        self.temperature_scaler = None
        self.mahalanobis_detector = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):        """Load pre-trained model weights"""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    def predict_with_trust(self, inputs, return_features=False):
        """
        Make predictions with comprehensive trust analysis
        
        Args:
            inputs: Input tensor [batch_size, 3, 32, 32]
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing predictions and trust metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass
            logits = self.model(inputs)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            # Compute trust metrics
            entropy_scores = compute_entropy(logits)
            
            # Temperature scaling if available
            calibrated_probs = probabilities
            if self.temperature_scaler:
                calibrated_probs = self.temperature_scaler.calibrate(logits)
            
            max_prob = torch.max(probabilities, dim=1)[0]
            calibrated_confidence = torch.max(calibrated_probs, dim=1)[0]
            
            results = {
                'predictions': predictions.cpu().numpy(),
                'probabilities': probabilities.cpu().numpy(),
                'logits': logits.cpu().numpy(),
                'entropy': entropy_scores.cpu().numpy(),
                'max_probability': max_prob.cpu().numpy(),
                'calibrated_confidence': calibrated_confidence.cpu().numpy(),
            }
            
            # ODIN scores - improved error handling
            try:
                print("Computing ODIN scores...")
                odin_probs = odin_predict(self.model, inputs)
                if odin_probs is not None:
                    odin_entropy = -torch.sum(odin_probs * torch.log(odin_probs + 1e-12), dim=1)
                    results['odin_entropy'] = odin_entropy.cpu().numpy()
                    print(f"ODIN computation successful. Entropy: {odin_entropy.cpu().numpy()}")
                else:
                    print("ODIN returned None, using fallback")
                    results['odin_entropy'] = None
            except Exception as e:
                print(f"ODIN computation failed: {e}")
                results['odin_entropy'] = None
            
            # Mahalanobis distance - improved implementation
            print("Computing Mahalanobis distances...")
            if self.mahalanobis_detector:
                try:
                    mahal_distances = self.mahalanobis_detector.compute_distances(inputs, self.model)
                    results['mahalanobis_distance'] = mahal_distances
                    print(f"Mahalanobis computation successful. Distances: {mahal_distances}")
                except Exception as e:
                    print(f"Mahalanobis computation failed: {e}")
                    results['mahalanobis_distance'] = None
            else:
                print("Mahalanobis detector not initialized")
                results['mahalanobis_distance'] = None
            
            return results
    
    def set_temperature_scaler(self, temperature_scaler):
        """Set the temperature scaling module"""
        self.temperature_scaler = temperature_scaler
    
    def set_mahalanobis_detector(self, mahalanobis_detector):
        """Set the Mahalanobis distance detector"""
        self.mahalanobis_detector = mahalanobis_detector
    
    def get_trust_score(self, entropy, mahal_dist=None, odin_entropy=None):
        """
        Compute overall trust score based on multiple metrics
        
        Args:
            entropy: Prediction entropy
            mahal_dist: Mahalanobis distance (optional)
            odin_entropy: ODIN entropy (optional)
            
        Returns:
            Trust score between 0 and 1 (higher = more trustworthy)
        """
        # Normalize entropy (lower entropy = higher trust)
        max_entropy = np.log(10)  # Maximum possible entropy for 10 classes
        entropy_trust = 1 - (entropy / max_entropy)
        
        trust_score = entropy_trust
        
        # Incorporate Mahalanobis distance if available
        if mahal_dist is not None:
            # Normalize Mahalanobis distance (lower distance = higher trust)
            mahal_trust = 1 / (1 + mahal_dist / 10.0)  # Sigmoid-like scaling
            trust_score = 0.7 * trust_score + 0.3 * mahal_trust
        
        # Incorporate ODIN entropy if available
        if odin_entropy is not None:
            odin_trust = 1 - (odin_entropy / max_entropy)
            trust_score = 0.6 * trust_score + 0.4 * odin_trust
        
        return np.clip(trust_score, 0, 1)
