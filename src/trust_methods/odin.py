"""
ODIN (Out-of-Distribution Detection) method
"""

import torch
import torch.nn.functional as F


def odin_predict(model, inputs, epsilon=0.0014, temperature=1000):
    """
    ODIN prediction with input preprocessing and temperature scaling
    
    Args:
        model: PyTorch model
        inputs: Input tensor
        epsilon: Perturbation magnitude
        temperature: Temperature for scaling
        
    Returns:
        softmax_outputs: Calibrated softmax predictions
    """
    # Store original training mode
    model_training = model.training
    device = inputs.device
    
    try:
        # Ensure inputs require grad and are on correct device
        inputs = inputs.clone().detach().to(device)
        inputs.requires_grad_(True)
        
        # Set model to eval mode to prevent batch norm issues, but enable gradients
        model.eval()
        
        # Enable gradient computation for this context
        with torch.enable_grad():
            # Forward pass
            outputs = model(inputs)
            outputs = outputs / temperature
            
            # Compute gradient w.r.t max logit
            max_logit, pred_label = torch.max(outputs, dim=1)
            loss = -torch.mean(max_logit)
            
            # Clear any existing gradients
            if inputs.grad is not None:
                inputs.grad.zero_()
            
            # Compute gradients
            loss.backward(retain_graph=False)
            
            # Check if gradients were computed
            if inputs.grad is not None and inputs.grad.numel() > 0:
                gradient = torch.sign(inputs.grad.data)
                perturbed_inputs = inputs - epsilon * gradient
                perturbed_inputs = torch.clamp(perturbed_inputs, 0, 1)
            else:
                # Fallback: use small random perturbation
                perturbation = epsilon * torch.sign(torch.randn_like(inputs))
                perturbed_inputs = inputs + perturbation
                perturbed_inputs = torch.clamp(perturbed_inputs, 0, 1)
        
        # Forward pass with perturbed inputs (no gradients needed)
        with torch.no_grad():
            outputs = model(perturbed_inputs.detach())
            outputs = outputs / temperature
            softmax_outputs = F.softmax(outputs, dim=1)
            
    except Exception as e:
        # Complete fallback: return standard forward pass
        with torch.no_grad():
            outputs = model(inputs.detach())
            outputs = outputs / temperature
            softmax_outputs = F.softmax(outputs, dim=1)
    
    finally:
        # Restore original training mode
        model.train(model_training)
    
    return softmax_outputs


def evaluate_with_odin(model, dataloader, epsilon=0.0014, temperature=1000):
    """
    Evaluate model with ODIN method
    
    Args:
        model: PyTorch model
        dataloader: Data loader
        epsilon: Perturbation magnitude
        temperature: Temperature for scaling
        
    Returns:
        Dictionary with predictions and ODIN scores
    """
    model.eval()
    all_entropy = []
    all_preds = []
    all_labels = []
    all_correct = []

    for inputs, labels in dataloader:
        softmax_outputs = odin_predict(model, inputs, epsilon, temperature)

        with torch.no_grad():
            entropy = -torch.sum(softmax_outputs * torch.log(softmax_outputs + 1e-12), dim=1)
            preds = torch.argmax(softmax_outputs, dim=1)

            all_entropy.extend(entropy.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_correct.extend((preds.cpu() == labels).numpy())

    return {
        "entropy": all_entropy,
        "preds": all_preds,
        "labels": all_labels,
        "correct": all_correct
    }
