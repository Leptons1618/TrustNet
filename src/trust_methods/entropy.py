"""
Entropy-based uncertainty quantification
"""

import torch
import torch.nn.functional as F


def compute_entropy(logits):
    """
    Computes entropy for each prediction in a batch.
    
    Args:
        logits: Model outputs before softmax, shape [batch_size, num_classes]
        
    Returns:
        entropy: Entropy scores, shape [batch_size]
    """
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    entropy = -torch.sum(probs * log_probs, dim=1)
    return entropy


def entropy_based_confidence(logits):
    """
    Convert entropy to confidence score
    
    Args:
        logits: Model outputs before softmax
        
    Returns:
        confidence: Confidence scores (higher = more confident)
    """
    entropy = compute_entropy(logits)
    max_entropy = torch.log(torch.tensor(logits.shape[1], dtype=torch.float32))
    confidence = 1 - (entropy / max_entropy)
    return confidence
