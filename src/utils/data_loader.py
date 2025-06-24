"""
Data loading and preprocessing utilities
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image


class CIFAR10DataLoader:
    """
    CIFAR-10 data loader with various transformations
    """
    
    def __init__(self, data_dir='./data', batch_size=128, num_workers=2):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Define transformations
        self.transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        self.transform_test_clean = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        self.transform_test_shifted = transforms.Compose([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
        ])
        
        # CIFAR-10 class names
        self.classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    
    def get_loaders(self):
        """
        Get train and test data loaders
        
        Returns:
            Dictionary with data loaders
        """
        # Load datasets
        trainset = datasets.CIFAR10(
            root=self.data_dir, 
            train=True, 
            download=True, 
            transform=self.transform_train
        )
        
        testset_clean = datasets.CIFAR10(
            root=self.data_dir, 
            train=False, 
            download=True, 
            transform=self.transform_test_clean
        )
        
        testset_shifted = datasets.CIFAR10(
            root=self.data_dir, 
            train=False, 
            download=True, 
            transform=self.transform_test_shifted
        )
        
        # Create data loaders
        trainloader = DataLoader(
            trainset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers
        )
        
        testloader_clean = DataLoader(
            testset_clean, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )
        
        testloader_shifted = DataLoader(
            testset_shifted, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )
        
        return {
            'train': trainloader,
            'test_clean': testloader_clean,
            'test_shifted': testloader_shifted,
            'datasets': {
                'train': trainset,
                'test_clean': testset_clean,
                'test_shifted': testset_shifted
            }
        }


def preprocess_uploaded_image(image, target_size=(32, 32)):
    """
    Preprocess uploaded image for model inference
    
    Args:
        image: PIL Image or numpy array
        target_size: Target image size (height, width)
        
    Returns:
        Preprocessed tensor ready for model
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Resize image
    image = image.resize(target_size, Image.LANCZOS)
    
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor


def tensor_to_image(tensor):
    """
    Convert tensor to displayable image
    
    Args:
        tensor: Image tensor [C, H, W] or [1, C, H, W]
        
    Returns:
        PIL Image
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # Remove batch dimension
    
    # Denormalize if needed (assuming values are in [0, 1])
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2  # Convert from [-1, 1] to [0, 1]
    
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and transpose
    img_array = tensor.permute(1, 2, 0).cpu().numpy()
    img_array = (img_array * 255).astype(np.uint8)
    
    return Image.fromarray(img_array)


def get_sample_images(dataset, num_samples=6):
    """
    Get sample images from dataset
    
    Args:
        dataset: PyTorch dataset
        num_samples: Number of samples to get
        
    Returns:
        List of (image, label) tuples
    """
    samples = []
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for idx in indices:
        image, label = dataset[idx]
        samples.append((image, label))
    
    return samples
