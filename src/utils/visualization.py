"""
Visualization utilities for TrustLens
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def plot_trust_metrics(entropy, mahal_dist=None, odin_entropy=None, title="Trust Metrics"):
    """
    Plot trust metrics distribution
    
    Args:
        entropy: Entropy scores
        mahal_dist: Mahalanobis distances (optional)
        odin_entropy: ODIN entropy scores (optional)
        title: Plot title
        
    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=1, 
        cols=2 if mahal_dist is not None else 1,
        subplot_titles=["Entropy Distribution", "Mahalanobis Distance"] if mahal_dist is not None else ["Entropy Distribution"]
    )
    
    # Entropy histogram
    fig.add_trace(
        go.Histogram(
            x=entropy,
            name="Entropy",
            nbinsx=30,
            opacity=0.7,
            marker_color="blue"
        ),
        row=1, col=1
    )
    
    # Mahalanobis distance histogram
    if mahal_dist is not None:
        fig.add_trace(
            go.Histogram(
                x=mahal_dist,
                name="Mahalanobis Distance",
                nbinsx=30,
                opacity=0.7,
                marker_color="red"
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        title=title,
        showlegend=False,
        height=400
    )
    
    return fig


def plot_confidence_calibration(y_true, y_prob, n_bins=10):
    """
    Plot confidence calibration curve
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration
        
    Returns:
        Plotly figure
    """
    # Compute calibration curve
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies = []
    confidences = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            accuracies.append(accuracy_in_bin)
            confidences.append(avg_confidence_in_bin)
        else:
            accuracies.append(0)
            confidences.append(0)
    
    # Create plot
    fig = go.Figure()
    
    # Perfect calibration line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(dash='dash', color='gray')
        )
    )
    
    # Calibration curve
    fig.add_trace(
        go.Scatter(
            x=confidences,
            y=accuracies,
            mode='lines+markers',
            name='Model Calibration',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        )
    )
    
    fig.update_layout(
        title="Confidence Calibration Curve",
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction of Positives",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        width=500,
        height=400
    )
    
    return fig


def plot_roc_curve(y_true, y_score, title="ROC Curve"):
    """
    Plot ROC curve
    
    Args:
        y_true: True binary labels
        y_score: Predicted scores
        title: Plot title
        
    Returns:
        Plotly figure
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    
    # ROC curve
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color='darkorange', width=2)
        )
    )
    
    # Random classifier line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='navy')
        )
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        width=500,
        height=400
    )
    
    return fig


def create_trust_gauge(trust_score, title="Trust Score"):
    """
    Create a gauge chart for trust score
    
    Args:
        trust_score: Trust score between 0 and 1
        title: Gauge title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=trust_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        delta={'reference': 80},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig


def plot_prediction_confidence(predictions, confidences, class_names):
    """
    Plot prediction confidence for multiple classes
    
    Args:
        predictions: Predicted class indices
        confidences: Confidence scores for each prediction
        class_names: List of class names
        
    Returns:
        Plotly figure
    """
    # Create confidence by class
    class_confidences = {}
    for pred, conf in zip(predictions, confidences):
        class_name = class_names[pred]
        if class_name not in class_confidences:
            class_confidences[class_name] = []
        class_confidences[class_name].append(conf)
    
    fig = go.Figure()
    
    for class_name, conf_list in class_confidences.items():
        fig.add_trace(
            go.Box(
                y=conf_list,
                name=class_name,
                boxpoints='outliers'
            )
        )
    
    fig.update_layout(
        title="Prediction Confidence by Class",
        yaxis_title="Confidence Score",
        xaxis_title="Predicted Class",
        height=400
    )
    
    return fig
