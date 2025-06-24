"""
Temperature scaling for model calibration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS


class TemperatureScaling(nn.Module):
    """
    Temperature scaling module for model calibration
    """
    
    def __init__(self):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits):
        """
        Apply temperature scaling to logits
        
        Args:
            logits: Model outputs before softmax
            
        Returns:
            Calibrated probabilities
        """
        return F.softmax(logits / self.temperature, dim=1)
    
    def calibrate(self, logits):
        """
        Apply calibration to logits
        
        Args:
            logits: Model outputs
            
        Returns:
            Calibrated probabilities
        """
        return self.forward(logits)
    
    def fit(self, logits, labels, lr=0.01, max_iter=50):
        """
        Fit temperature parameter using validation data
        
        Args:
            logits: Validation logits
            labels: Validation labels
            lr: Learning rate
            max_iter: Maximum iterations
        """
        self.train()
        
        # Convert to tensors if needed
        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits, dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
        
        # Use NLL loss for calibration
        nll_criterion = nn.CrossEntropyLoss()
        
        # Optimize temperature
        optimizer = LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(logits / self.temperature, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        self.eval()
        print(f"Optimal temperature: {self.temperature.item():.4f}")
        
        return self.temperature.item()


def apply_temperature_scaling(logits, temperature):
    """
    Apply temperature scaling to logits
    
    Args:
        logits: Model outputs
        temperature: Temperature parameter
        
    Returns:
        Calibrated probabilities
    """
    return F.softmax(logits / temperature, dim=1)
