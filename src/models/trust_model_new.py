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
    
    def load_model(self, model_path):
        """Load pre-trained model weights"""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    def predict_with_trust(self, inputs, return_features=False, use_odin=True, use_mahalanobis=True):
        """
        Make predictions with comprehensive trust analysis
        
        Args:
            inputs: Input tensor [batch_size, 3, 32, 32]
            return_features: Whether to return intermediate features
            use_odin: Whether to compute ODIN scores
            use_mahalanobis: Whether to compute Mahalanobis distances
            
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
            
            # ODIN scores - only compute if enabled
            if use_odin:
                try:
                    odin_probs = odin_predict(self.model, inputs)
                    odin_entropy = -torch.sum(odin_probs * torch.log(odin_probs + 1e-12), dim=1)
                    results['odin_entropy'] = odin_entropy.cpu().numpy()
                except Exception as e:
                    print(f"ODIN computation failed: {e}")
                    results['odin_entropy'] = np.array([None] * len(inputs))
            
            # Mahalanobis distance - only compute if enabled and detector is available
            if use_mahalanobis and self.mahalanobis_detector:
                try:
                    mahal_distances = self.mahalanobis_detector.compute_distances(inputs, self.model)
                    results['mahalanobis_distance'] = mahal_distances
                except Exception as e:
                    print(f"Mahalanobis computation failed: {e}")
                    results['mahalanobis_distance'] = np.array([None] * len(inputs))
            
            return results
    
    def set_temperature_scaler(self, temperature_scaler):
        """Set the temperature scaling module"""
        self.temperature_scaler = temperature_scaler
    
    def set_mahalanobis_detector(self, mahalanobis_detector):
        """Set the Mahalanobis distance detector"""
        self.mahalanobis_detector = mahalanobis_detector
    
    def get_trust_score(self, entropy, mahalanobis_distance=None, odin_entropy=None):
        """
        Compute overall trust score based on multiple metrics
        
        Args:
            entropy: Prediction entropy
            mahalanobis_distance: Mahalanobis distance score (optional)
            odin_entropy: ODIN entropy score (optional)
            
        Returns:
            Float trust score between 0 and 1
        """
        # Normalize entropy (lower is better, so we invert it)
        max_entropy = np.log(10)  # For 10 classes
        entropy_score = max(0, 1 - (entropy / max_entropy))
        
        # Start with entropy as base score
        trust_score = entropy_score
        scores = [entropy_score]
        
        # Add Mahalanobis if available
        if mahalanobis_distance is not None:
            # Normalize Mahalanobis (lower is better)
            mahal_score = max(0, 1 - (mahalanobis_distance / 100))  # Rough normalization
            scores.append(mahal_score)
        
        # Add ODIN if available
        if odin_entropy is not None:
            # Normalize ODIN entropy (lower is better)
            odin_score = max(0, 1 - (odin_entropy / max_entropy))
            scores.append(odin_score)
        
        # Average all available scores
        trust_score = np.mean(scores)
        
        return trust_score
    
    def calibrate_confidence(self, validation_loader):
        """
        Calibrate confidence using temperature scaling
        
        Args:
            validation_loader: DataLoader for validation data
        """
        self.temperature_scaler = TemperatureScaling()
        self.temperature_scaler.fit(self.model, validation_loader)
    
    def setup_mahalanobis(self, training_loader):
        """
        Setup Mahalanobis distance detector using training data
        
        Args:
            training_loader: DataLoader for training data
        """
        self.mahalanobis_detector = MahalanobisDetector()
        self.mahalanobis_detector.fit(self.model, training_loader)
