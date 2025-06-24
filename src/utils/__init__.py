"""
Utils package initialization
"""

from .data_loader import CIFAR10DataLoader, preprocess_uploaded_image, tensor_to_image, get_sample_images
from .visualization import (
    plot_trust_metrics, 
    plot_confidence_calibration, 
    plot_roc_curve, 
    create_trust_gauge,
    plot_prediction_confidence
)

try:
    from .gradcam import GradCAM, show_gradcam_on_image, create_gradcam_for_model
    gradcam_available = True
except ImportError:
    gradcam_available = False
    print("Warning: GradCAM functionality not available. Install opencv-python to enable.")

__all__ = [
    'CIFAR10DataLoader', 
    'preprocess_uploaded_image', 
    'tensor_to_image', 
    'get_sample_images',
    'plot_trust_metrics', 
    'plot_confidence_calibration', 
    'plot_roc_curve', 
    'create_trust_gauge',
    'plot_prediction_confidence'
]

if gradcam_available:
    __all__.extend(['GradCAM', 'show_gradcam_on_image', 'create_gradcam_for_model'])
