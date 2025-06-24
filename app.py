"""
TrustLens: Explainable & Trust-Aware AI
Main Streamlit Application
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
from PIL import Image
import os
import sys
import traceback

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models import TrustModel
from src.utils import (
    CIFAR10DataLoader, 
    preprocess_uploaded_image, 
    tensor_to_image,
    plot_trust_metrics,
    create_trust_gauge,
    plot_confidence_calibration
)
from src.trust_methods import MahalanobisDetector, TemperatureScaling

# Try to import GradCAM
try:
    from src.utils.gradcam import create_gradcam_for_model
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False


# Page configuration
st.set_page_config(
    page_title="TrustLens AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .trust-high { border-left-color: #10b981 !important; }
    .trust-medium { border-left-color: #f59e0b !important; }
    .trust-low { border-left-color: #ef4444 !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the pre-trained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check for model file
    model_paths = [
        "cookbook/trustnet_cnn.pth",
        "og_Model/trustnet_cnn.pth",
        "models/trustnet_cnn.pth"
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        st.error("Model file not found! Please ensure trustnet_cnn.pth exists in one of: " + ", ".join(model_paths))
        return None
    
    try:
        trust_model = TrustModel(model_path=model_path, device=device)
        st.success(f"✅ Model loaded successfully from {model_path}")
        return trust_model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None


@st.cache_resource
def load_data_loader():
    """Load CIFAR-10 data loader"""
    try:
        data_loader = CIFAR10DataLoader(data_dir='./cookbook/data', batch_size=32)
        return data_loader
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return None


def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 TrustLens AI</h1>', unsafe_allow_html=True)
    st.markdown("### Explainable & Trust-Aware Image Classification")
    st.markdown("---")
    
    # Load model
    trust_model = load_model()
    if trust_model is None:
        st.stop()
    
    # Load data loader
    data_loader = load_data_loader()
    
    # Sidebar
    st.sidebar.header("🎛️ Controls")
    
    # Image input options
    input_method = st.sidebar.selectbox(
        "Choose input method:",
        ["Upload Image", "Sample Images", "CIFAR-10 Test"]
    )
    
    # Trust method selection
    st.sidebar.subheader("Trust Analysis Methods")
    use_entropy = st.sidebar.checkbox("Entropy-based Uncertainty", value=True)
    use_odin = st.sidebar.checkbox("ODIN Detection", value=True)
    use_mahalanobis = st.sidebar.checkbox("Mahalanobis Distance", value=False)
    use_gradcam = st.sidebar.checkbox("Grad-CAM Explanation", value=GRADCAM_AVAILABLE)
    
    if use_gradcam and not GRADCAM_AVAILABLE:
        st.sidebar.warning("Grad-CAM requires opencv-python. Install it to enable this feature.")
        use_gradcam = False
    
    # Advanced settings
    with st.sidebar.expander("⚙️ Advanced Settings"):
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.8)
        temperature = st.slider("Temperature Scaling", 0.1, 5.0, 1.0)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Input Image")
        
        input_tensor = None
        original_image = None
        
        if input_method == "Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an image...", 
                type=['png', 'jpg', 'jpeg'],
                help="Upload an image for classification and trust analysis"            )
            
            if uploaded_file is not None:
                original_image = Image.open(uploaded_file)
                st.image(original_image, caption="Uploaded Image", use_container_width=True)
                input_tensor = preprocess_uploaded_image(original_image)
        
        elif input_method == "Sample Images":
            # Create sample images if data loader is available
            if data_loader:
                sample_images = [
                    "Sample 1", "Sample 2", "Sample 3", "Sample 4", "Sample 5"
                ]
                selected_sample = st.selectbox("Choose a sample:", sample_images)
                
                # For demo, create a random sample
                if st.button("Generate Random Sample"):
                    try:
                        data_loaders = data_loader.get_loaders()
                        test_dataset = data_loaders['datasets']['test_clean']
                        
                        # Get a random sample
                        idx = np.random.randint(0, len(test_dataset))
                        sample_tensor, sample_label = test_dataset[idx]
                        
                        original_image = tensor_to_image(sample_tensor)
                        st.image(original_image, caption=f"Sample Image (True: {data_loader.classes[sample_label]})", use_container_width=True)
                        input_tensor = sample_tensor.unsqueeze(0)
                        
                    except Exception as e:
                        st.error(f"Failed to load sample: {str(e)}")
        
        elif input_method == "CIFAR-10 Test":
            if data_loader:
                test_type = st.selectbox("Test Set Type:", ["Clean", "Domain-Shifted"])
                
                if st.button("Get Random Test Image"):
                    try:
                        data_loaders = data_loader.get_loaders()
                        test_key = 'test_clean' if test_type == "Clean" else 'test_shifted'
                        test_dataset = data_loaders['datasets'][test_key]
                        
                        idx = np.random.randint(0, len(test_dataset))
                        sample_tensor, sample_label = test_dataset[idx]
                        
                        original_image = tensor_to_image(sample_tensor)
                        st.image(original_image, caption=f"Test Image ({test_type}) - True: {data_loader.classes[sample_label]}", use_container_width=True)
                        input_tensor = sample_tensor.unsqueeze(0)
                        
                    except Exception as e:
                        st.error(f"Failed to load test image: {str(e)}")
    
    with col2:
        st.subheader("🤖 AI Analysis")
        
        if input_tensor is not None:
            with st.spinner("Analyzing image..."):
                try:
                    # Move tensor to device
                    device = next(trust_model.model.parameters()).device
                    input_tensor = input_tensor.to(device)
                    
                    # Get predictions with trust analysis
                    results = trust_model.predict_with_trust(input_tensor)
                    
                    # Extract results
                    prediction = results['predictions'][0]
                    probabilities = results['probabilities'][0]
                    entropy = results['entropy'][0]
                    max_prob = results['max_probability'][0]
                    
                    # Get class names
                    if data_loader:
                        class_names = data_loader.classes
                        predicted_class = class_names[prediction]
                    else:
                        class_names = [f"Class {i}" for i in range(10)]
                        predicted_class = class_names[prediction]
                    
                    # Display prediction
                    st.success(f"**Predicted Class:** {predicted_class}")
                    st.info(f"**Confidence:** {max_prob:.3f}")
                    
                    # Trust score calculation
                    trust_score = trust_model.get_trust_score(
                        entropy,
                        results.get('mahalanobis_distance', [None])[0] if 'mahalanobis_distance' in results else None,
                        results.get('odin_entropy', [None])[0] if 'odin_entropy' in results else None
                    )
                    
                    # Trust level
                    if trust_score > 0.8:
                        trust_level = "HIGH"
                        trust_color = "trust-high"
                    elif trust_score > 0.5:
                        trust_level = "MEDIUM"
                        trust_color = "trust-medium"
                    else:
                        trust_level = "LOW"
                        trust_color = "trust-low"
                    
                    st.markdown(f'<div class="metric-card {trust_color}"><h4>Trust Level: {trust_level}</h4><p>Score: {trust_score:.3f}</p></div>', unsafe_allow_html=True)
                    
                    # Detailed metrics
                    st.subheader("📊 Trust Metrics")
                    
                    metric_cols = st.columns(3)
                    
                    with metric_cols[0]:
                        st.metric("Entropy", f"{entropy:.3f}", help="Lower = more confident")
                    
                    with metric_cols[1]:
                        if 'odin_entropy' in results:
                            st.metric("ODIN Score", f"{results['odin_entropy'][0]:.3f}", help="OOD detection score")
                        else:
                            st.metric("ODIN Score", "N/A")
                    
                    with metric_cols[2]:
                        if 'mahalanobis_distance' in results:
                            st.metric("Mahal. Dist.", f"{results['mahalanobis_distance'][0]:.3f}", help="Distance to training distribution")
                        else:
                            st.metric("Mahal. Dist.", "N/A")
                    
                    # Probability distribution
                    st.subheader("🎯 Class Probabilities")
                    prob_df = pd.DataFrame({
                        'Class': class_names,
                        'Probability': probabilities
                    }).sort_values('Probability', ascending=False)
                    
                    st.bar_chart(prob_df.set_index('Class')['Probability'])
                    
                    # Top 3 predictions
                    st.subheader("🏆 Top 3 Predictions")
                    for i, (idx, row) in enumerate(prob_df.head(3).iterrows()):
                        rank_icon = ["🥇", "🥈", "🥉"][i]
                        st.write(f"{rank_icon} **{row['Class']}**: {row['Probability']:.3f}")
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.error(traceback.format_exc())
    
    # Full-width sections
    if input_tensor is not None:
          # Grad-CAM Explanation
        if use_gradcam and GRADCAM_AVAILABLE:
            st.subheader("🔍 Explainability: Grad-CAM")
            
            try:
                with st.spinner("Generating explanation..."):
                    gradcam_results = create_gradcam_for_model(trust_model.model, input_tensor)
                    
                    explanation_cols = st.columns(3)
                    
                    with explanation_cols[0]:
                        st.image(gradcam_results['original_image'], caption="Original", use_container_width=True)
                    
                    with explanation_cols[1]:
                        st.image(gradcam_results['heatmap'], caption="Attention Heatmap", use_container_width=True, clamp=True)
                    
                    with explanation_cols[2]:
                        st.image(gradcam_results['overlayed_image'], caption="Grad-CAM Overlay", use_container_width=True)
                    
                    st.info("🔥 **Red areas** show regions that most influenced the model's decision.")
                    
            except Exception as e:
                st.error(f"Grad-CAM generation failed: {str(e)}")
        
        # Trust Dashboard
        st.subheader("📈 Trust Dashboard")
        
        dashboard_cols = st.columns(2)
        
        with dashboard_cols[0]:
            # Trust gauge
            if 'trust_score' in locals():
                gauge_fig = create_trust_gauge(trust_score, "Overall Trust Score")
                st.plotly_chart(gauge_fig, use_container_width=True)
        
        with dashboard_cols[1]:
            # Trust metrics plot
            if 'results' in locals():
                trust_data = {
                    'Entropy': [entropy],
                }
                
                if 'odin_entropy' in results:
                    trust_data['ODIN Entropy'] = [results['odin_entropy'][0]]
                
                if 'mahalanobis_distance' in results:
                    trust_data['Mahalanobis Distance'] = [results['mahalanobis_distance'][0]]
                
                trust_df = pd.DataFrame(trust_data)
                st.bar_chart(trust_df.T)
    
    # Footer
    st.markdown("---")
    st.markdown("### About TrustLens")
    st.markdown("""
    TrustLens combines state-of-the-art uncertainty quantification and explainability techniques to provide trustworthy AI predictions:
    
    - 🎯 **Entropy-based Uncertainty**: Measures prediction confidence using information theory
    - 🚨 **ODIN Detection**: Identifies out-of-distribution inputs using input preprocessing
    - 📏 **Mahalanobis Distance**: Detects anomalies in the feature space
    - 🔍 **Grad-CAM**: Visualizes which parts of the image influenced the decision
    - 🎚️ **Temperature Scaling**: Calibrates confidence scores for better reliability
    
    Built with ❤️ for transparent and trustworthy AI.
    """)


if __name__ == "__main__":
    main()
