"""
TrustLens: Explainable & Trust-Aware AI
Main package initialization
"""

__version__ = "1.0.0"
__author__ = "TrustLens Team"
__description__ = "Explainable and trust-aware AI for image classification"

# Import main components
try:
    from .src.models import SimpleCNN, TrustModel
    from .src.trust_methods import (
        compute_entropy,
        odin_predict,
        MahalanobisDetector,
        TemperatureScaling
    )
    from .src.utils import (
        CIFAR10DataLoader,
        preprocess_uploaded_image,
        plot_trust_metrics,
        create_trust_gauge
    )
    
    __all__ = [
        'SimpleCNN',
        'TrustModel',
        'compute_entropy',
        'odin_predict',
        'MahalanobisDetector',
        'TemperatureScaling',
        'CIFAR10DataLoader',
        'preprocess_uploaded_image',
        'plot_trust_metrics',
        'create_trust_gauge'
    ]
    
except ImportError as e:
    # Graceful degradation if some dependencies are missing
    print(f"Warning: Some TrustLens components could not be imported: {e}")
    __all__ = []
