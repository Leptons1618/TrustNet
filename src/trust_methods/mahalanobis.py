"""
Mahalanobis distance-based out-of-distribution detection
"""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


class MahalanobisDetector:
    """
    Mahalanobis distance detector for OOD detection
    """
    
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.class_means = {}
        self.inv_cov = None
        self.feature_list = []
        self.hook_handle = None
    
    def _feature_hook(self, module, input, output):
        """Hook function to extract features"""
        self.feature_list.append(output.flatten(start_dim=1))
    
    def fit(self, model, dataloader):
        """
        Fit the Mahalanobis detector using training data
        
        Args:
            model: PyTorch model
            dataloader: Training data loader
        """
        model.eval()
        
        # Register hook on the feature layer
        target_layer = model.get_feature_layer() if hasattr(model, 'get_feature_layer') else model.features[-3]
        self.hook_handle = target_layer.register_forward_hook(self._feature_hook)
        
        # Extract features for each class
        features_by_class = {i: [] for i in range(self.num_classes)}
        
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Extracting features for Mahalanobis"):
                self.feature_list.clear()
                inputs = inputs.to(next(model.parameters()).device)
                _ = model(inputs)
                
                if self.feature_list:
                    feats = self.feature_list[0].cpu()
                    self.feature_list.clear()

                    for feat, label in zip(feats, labels):
                        features_by_class[label.item()].append(feat.numpy())
        
        # Compute class means and covariance
        all_features = []
        for c in range(self.num_classes):
            if features_by_class[c]:  # Check if class has samples
                feats_c = np.stack(features_by_class[c])
                self.class_means[c] = np.mean(feats_c, axis=0)
                all_features.append(feats_c)
        
        if all_features:
            all_features = np.vstack(all_features)
            cov = np.cov(all_features, rowvar=False)
            self.inv_cov = np.linalg.inv(cov + 1e-5 * np.eye(cov.shape[0]))  # Regularization
        
        # Remove hook
        if self.hook_handle:
            self.hook_handle.remove()
    
    def mahalanobis_distance(self, x, mean):
        """
        Compute Mahalanobis distance
        
        Args:
            x: Feature vector
            mean: Class mean
            
        Returns:
            Mahalanobis distance
        """
        if self.inv_cov is None:
            return np.linalg.norm(x - mean)  # Fallback to Euclidean distance
        
        diff = x - mean
        dist = np.sqrt(np.dot(np.dot(diff, self.inv_cov), diff.T))
        return dist
    
    def compute_distances(self, inputs, model):
        """
        Compute Mahalanobis distances for input batch
        
        Args:
            inputs: Input tensor
            model: PyTorch model
            
        Returns:
            Array of minimum distances to class centroids
        """
        model.eval()
        
        # Register hook temporarily
        target_layer = model.get_feature_layer() if hasattr(model, 'get_feature_layer') else model.features[-3]
        hook_handle = target_layer.register_forward_hook(self._feature_hook)
        
        distances = []
        
        with torch.no_grad():
            self.feature_list.clear()
            _ = model(inputs)
            
            if self.feature_list:
                feats = self.feature_list[0].cpu().numpy()
                self.feature_list.clear()
                
                for feat in feats:
                    dists = []
                    for c in range(self.num_classes):
                        if c in self.class_means:
                            dist = self.mahalanobis_distance(feat, self.class_means[c])
                            dists.append(dist)
                    
                    if dists:
                        distances.append(min(dists))  # Use min distance across classes
                    else:
                        distances.append(10.0)  # Default high distance
            else:
                distances = [10.0] * len(inputs)  # Default values
        
        # Remove hook
        hook_handle.remove()
        
        return np.array(distances)
