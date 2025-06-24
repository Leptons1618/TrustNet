"""
Configuration settings for TrustLens
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Model settings
MODEL_CONFIG = {
    'num_classes': 10,
    'input_size': (32, 32),
    'device': 'auto',  # 'auto', 'cpu', or 'cuda'
    'model_paths': [
        'models/trustnet_cnn.pth',
        'cookbook/trustnet_cnn.pth',
        'og_Model/trustnet_cnn.pth'
    ]
}

# Data settings
DATA_CONFIG = {
    'data_dir': 'cookbook/data',
    'batch_size': 32,
    'num_workers': 2,
    'classes': [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
}

# Trust method settings
TRUST_CONFIG = {
    'entropy': {
        'enabled': True,
        'threshold': 1.5  # High entropy threshold
    },
    'odin': {
        'enabled': True,
        'epsilon': 0.0014,
        'temperature': 1000
    },
    'mahalanobis': {
        'enabled': False,  # Requires training data preprocessing
        'regularization': 1e-5
    },
    'temperature_scaling': {
        'enabled': False,  # Requires calibration data
        'max_iter': 50,
        'lr': 0.01
    }
}

# Visualization settings
VIZ_CONFIG = {
    'gradcam': {
        'enabled': True,
        'alpha': 0.5,
        'colormap': 'jet'
    },
    'plots': {
        'width': 500,
        'height': 400,
        'theme': 'plotly_white'
    }
}

# Streamlit settings
STREAMLIT_CONFIG = {
    'page_title': 'TrustLens AI',
    'page_icon': '🔍',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
    'theme': {
        'base': 'light',
        'primaryColor': '#667eea',
        'backgroundColor': '#ffffff',
        'secondaryBackgroundColor': '#f0f2f6'
    }
}

# Logging settings
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_dir': 'logs',
    'max_log_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5
}

# Cache settings
CACHE_CONFIG = {
    'model_cache': True,
    'data_cache': True,
    'gradcam_cache': False,  # Disable for dynamic visualizations
    'ttl': 3600  # 1 hour
}

# Security settings
SECURITY_CONFIG = {
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'allowed_extensions': ['.png', '.jpg', '.jpeg', '.bmp', '.tiff'],
    'sanitize_filenames': True
}

def get_model_path():
    """Get the first available model path"""
    for path in MODEL_CONFIG['model_paths']:
        full_path = PROJECT_ROOT / path
        if full_path.exists():
            return str(full_path)
    return None

def ensure_directories():
    """Ensure required directories exist"""
    directories = [
        'models',
        'data',
        'logs',
        'temp',
        'assets'
    ]
    
    for directory in directories:
        dir_path = PROJECT_ROOT / directory
        dir_path.mkdir(exist_ok=True)

# Initialize directories on import
ensure_directories()
