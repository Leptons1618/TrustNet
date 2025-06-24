"""
Grad-CAM implementation for model explainability
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:
    """
    Grad-CAM implementation for CNN models
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Save forward activations"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Save backward gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Input image tensor [1, C, H, W]
            class_idx: Target class index (if None, uses predicted class)
            
        Returns:
            heatmap: Normalized heatmap as numpy array
        """
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        
        # Backward pass
        class_score = output[:, class_idx]
        class_score.backward()
        
        # Generate Grad-CAM
        if self.gradients is not None and self.activations is not None:
            # Global average pooling of gradients
            pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])  # [C]
            activations = self.activations.squeeze(0)  # [C, H, W]
            
            # Weight activations by gradients
            for i in range(activations.shape[0]):
                activations[i] *= pooled_gradients[i]
            
            # Sum across channels and apply ReLU
            heatmap = torch.sum(activations, dim=0)
            heatmap = torch.relu(heatmap)
            
            # Normalize
            if torch.max(heatmap) > 0:
                heatmap /= torch.max(heatmap)
            
            return heatmap.cpu().numpy()
        else:
            # Return empty heatmap if hooks failed
            return np.zeros((8, 8))  # Default size for our CNN
    
    def __del__(self):
        """Clean up hooks"""
        if hasattr(self, 'forward_hook'):
            self.forward_hook.remove()
        if hasattr(self, 'backward_hook'):
            self.backward_hook.remove()


def show_gradcam_on_image(img_tensor, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlay Grad-CAM heatmap on original image
    
    Args:
        img_tensor: Original image tensor [C, H, W]
        heatmap: Grad-CAM heatmap
        alpha: Overlay alpha
        colormap: OpenCV colormap
        
    Returns:
        overlayed_img: Image with heatmap overlay
    """
    # Convert tensor to numpy
    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.permute(1, 2, 0).cpu().numpy()
    else:
        img = img_tensor
    
    img = np.clip(img, 0, 1)
    
    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
    heatmap_colored = heatmap_colored[..., ::-1] / 255.0  # BGR to RGB and normalize
    
    # Overlay
    overlayed_img = alpha * heatmap_colored + (1 - alpha) * img
    overlayed_img = np.clip(overlayed_img, 0, 1)
    
    return overlayed_img


def create_gradcam_for_model(model, input_tensor, target_class=None):
    """
    Create Grad-CAM visualization for a model
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor
        target_class: Target class for visualization
        
    Returns:
        Dictionary with heatmap and overlayed image
    """
    # Get the target layer (last conv layer)
    if hasattr(model, 'get_feature_layer'):
        target_layer = model.get_feature_layer()
    else:
        # Fallback for SimpleCNN
        target_layer = model.features[-3]
    
    # Create GradCAM instance
    gradcam = GradCAM(model, target_layer)
    
    # Generate heatmap
    heatmap = gradcam.generate(input_tensor, target_class)
    
    # Create overlay
    img_for_overlay = input_tensor.squeeze(0)  # Remove batch dimension
    overlayed_img = show_gradcam_on_image(img_for_overlay, heatmap)
    
    return {
        'heatmap': heatmap,
        'overlayed_image': overlayed_img,
        'original_image': img_for_overlay.permute(1, 2, 0).cpu().numpy()
    }
