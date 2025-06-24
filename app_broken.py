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
from src.trust_methods im                    # Detailed metrics
                    st.subheader("📊 Trust Metrics")
                    
                    metric_cols = st.columns(3)t MahalanobisDetector, TemperatureScaling

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
    .feature-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
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
    
    # Sidebar navigation and settings
    st.sidebar.title("🔍 TrustLens AI")
    
    # About button in sidebar
    if st.sidebar.button("📖 About TrustLens", use_container_width=True):
        st.session_state.show_about = True
    
    # Check if about page should be shown
    if st.session_state.get('show_about', False):
        show_about_page()
        return
    
    # Load model
    trust_model = load_model()
    if trust_model is None:
        st.stop()
    
    # Load data loader
    data_loader = load_data_loader()
    
    # Sidebar settings
    st.sidebar.header("🎛️ Trust Analysis Settings")
    
    # Trust method selection in sidebar
    use_entropy = st.sidebar.checkbox("Entropy-based Uncertainty", value=True)
    use_odin = st.sidebar.checkbox("ODIN Detection", value=True)
    use_mahalanobis = st.sidebar.checkbox("Mahalanobis Distance", value=False)
    use_gradcam = st.sidebar.checkbox("Grad-CAM Explanation", value=GRADCAM_AVAILABLE)
    
    if use_gradcam and not GRADCAM_AVAILABLE:
        st.sidebar.warning("Grad-CAM requires opencv-python. Install it to enable this feature.")
        use_gradcam = False
    
    # Advanced settings in sidebar
    with st.sidebar.expander("⚙️ Advanced Settings"):
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.8)
        temperature = st.slider("Temperature Scaling", 0.1, 5.0, 1.0)
        
        # ODIN settings
        st.markdown("**ODIN Parameters:**")
        odin_epsilon = st.slider("Perturbation (ε)", 0.0001, 0.01, 0.0014, format="%.4f")
        odin_temperature = st.slider("ODIN Temperature", 100, 2000, 1000)
        
        # Trust thresholds
        st.markdown("**Trust Thresholds:**")
        high_trust_thresh = st.slider("High Trust", 0.7, 0.95, 0.8)
        low_trust_thresh = st.slider("Low Trust", 0.3, 0.6, 0.5)
        
        # Display options
        st.markdown("**Display Options:**")
        show_raw_scores = st.checkbox("Show Raw Trust Scores", value=False)
        show_technical_details = st.checkbox("Show Technical Details", value=False)
    
    # Main home page content
    show_main_page(trust_model, data_loader, use_entropy, use_odin, use_mahalanobis, use_gradcam, 
                   confidence_threshold, temperature)


def show_about_page():
    """Display the About page"""
    st.markdown('<h1 class="main-header">📖 About TrustLens</h1>', unsafe_allow_html=True)
    
    # Close button
    if st.button("← Back to Main App", key="back_to_main"):
        st.session_state.show_about = False
        st.rerun()
    
    st.markdown("---")
    
    # Introduction
    st.markdown("""
    ## 🔍 What is TrustLens?
    
    TrustLens is an **explainable and trust-aware AI prototype** that combines state-of-the-art uncertainty quantification 
    and explainability techniques to provide transparent and trustworthy image classification predictions.
    """)
    
    # Features in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎲 Entropy", "🚨 ODIN", "📏 Mahalanobis", "🔍 Grad-CAM"])
    
    with tab1:
        st.markdown("""
        ### 🎲 Entropy-based Uncertainty
        
        **What it does:** Measures how confident the model is in its predictions using information theory.
        
        **How it works:** Calculates the entropy of the prediction probabilities. Lower entropy = higher confidence.
        
        **Example:** 
        - If the model predicts [0.9, 0.05, 0.05], the entropy is low (confident)
        - If it predicts [0.4, 0.3, 0.3], the entropy is high (uncertain)
        
        **Parameter Effects:**
        - **Threshold:** Lower values flag more predictions as uncertain
        - **Temperature:** Higher values make predictions more uniform (higher entropy)
        
        **Formula:** `H(p) = -Σ p_i * log(p_i)`
        """)
    
    with tab2:
        st.markdown("""
        ### 🚨 ODIN Detection
        
        **What it does:** Identifies out-of-distribution (OOD) samples using input preprocessing and temperature scaling.
        
        **How it works:** Applies small perturbations to inputs and uses temperature scaling to amplify differences.
        
        **Example:** Helps distinguish between CIFAR-10 images and completely different datasets.
        
        **Parameter Effects:**
        - **Epsilon (ε):** Controls perturbation magnitude (typically 0.0014)
        - **Temperature:** Higher values separate in-distribution from OOD better
        
        **Key Insight:** OOD samples become more distinguishable after perturbation and temperature scaling.
        """)
    
    with tab3:
        st.markdown("""
        ### 📏 Mahalanobis Distance
        
        **What it does:** Detects anomalies by measuring how far an input is from the training data distribution.
        
        **How it works:** Computes distance in the feature space, accounting for correlations between features.
        
        **Example:** An image of a car when trained on animals would have high Mahalanobis distance.
        
        **Parameter Effects:**
        - **Threshold:** Higher values allow more deviation from training data
        - **Layer:** Deeper layers capture more semantic features
        
        **Formula:** `d = sqrt((x-μ)ᵀ Σ⁻¹ (x-μ))`
        """)
    
    with tab4:
        st.markdown("""
        ### 🔍 Grad-CAM Explanation
        
        **What it does:** Visualizes which parts of the image the model focuses on for its decision.
        
        **How it works:** Uses gradients to highlight important regions in the input image.
        
        **Example:** For a cat image, it might highlight the ears, eyes, and whiskers.
        
        **Interpretation:**
        - 🔴 **Red areas:** Most important for the decision
        - 🟡 **Yellow areas:** Moderately important  
        - 🔵 **Blue areas:** Less important
        
        **Benefit:** Provides visual explanations for model decisions, increasing transparency.
        """)
    
    # Trust Score Calculation
    st.markdown("---")
    st.markdown("""
    ### 🎚️ Trust Score Calculation
    
    TrustLens combines multiple uncertainty measures into a single trust score:
    
    ```
    Trust Score = weighted_average(
        normalized_entropy,
        normalized_odin_score, 
        normalized_mahalanobis_score
    )
    ```
    
    **Trust Levels:**
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("🟢 **HIGH (>0.8)**")
        st.write("Model is very confident and the input seems familiar")
    
    with col2:
        st.warning("🟡 **MEDIUM (0.5-0.8)**")
        st.write("Some uncertainty or potential anomaly detected")
    
    with col3:
        st.error("🔴 **LOW (<0.5)**")
        st.write("High uncertainty or likely out-of-distribution input")
    
    # Use Cases
    st.markdown("---")
    st.markdown("### 🔬 Use Cases and Examples")
    
    use_case_cols = st.columns(3)
    
    with use_case_cols[0]:
        st.markdown("""
        #### 🏥 Medical Imaging
        - **High Trust:** Clear X-ray with obvious fracture
        - **Medium Trust:** Ambiguous scan requiring expert review  
        - **Low Trust:** Corrupted or unusual image format
        """)
    
    with use_case_cols[1]:
        st.markdown("""
        #### 🚗 Autonomous Driving
        - **High Trust:** Clear road with standard traffic signs
        - **Medium Trust:** Weather conditions affecting visibility
        - **Low Trust:** Construction zone with unfamiliar setup
        """)
    
    with use_case_cols[2]:
        st.markdown("""
        #### 🏭 Quality Control
        - **High Trust:** Standard product with clear defect
        - **Medium Trust:** Borderline quality issues
        - **Low Trust:** Completely new product type
        """)
    
    # Technical Details
    st.markdown("---")
    st.markdown("""
    ### 🛠️ Technical Implementation
    
    **Model Architecture:**
    - Convolutional Neural Network (CNN) trained on CIFAR-10
    - 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    
    **Framework:** Built with PyTorch, Streamlit, and modern ML libraries
    
    ### 📚 References
    
    - **ODIN:** Liang et al. "Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks" (2018)
    - **Mahalanobis:** Lee et al. "A Simple Unified Framework for Detecting Out-of-Distribution Samples" (2018)
    - **Grad-CAM:** Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks" (2017)
    - **Temperature Scaling:** Guo et al. "On Calibration of Modern Neural Networks" (2017)
    
    ### 🚀 Getting Started
    
    1. **Choose Input Method:** Select from Upload, Samples, or Test data
    2. **Configure Trust Analysis:** Enable desired trust methods in sidebar
    3. **Analyze Results:** Review predictions, trust scores, and explanations
    4. **Adjust Settings:** Fine-tune parameters in Advanced Settings
      Built with ❤️ for transparent and trustworthy AI.
    """)


def show_main_page(trust_model, data_loader, use_entropy, use_odin, use_mahalanobis, use_gradcam, 
                   confidence_threshold, temperature):
    """Display the main application page"""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 TrustLens AI</h1>', unsafe_allow_html=True)
    st.markdown("### Explainable & Trust-Aware Image Classification")
    st.markdown("---")
    
    # Input method tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload Image", "🖼️ Sample Images", "🧪 CIFAR-10 Test"])
    
    input_tensor = None
    original_image = None
    current_method = None
    
    with tab1:
        st.subheader("� Upload Your Own Image")
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image for classification and trust analysis"
        )
        
        if uploaded_file is not None:
            original_image = Image.open(uploaded_file)
            st.image(original_image, caption="Uploaded Image", use_container_width=True)
            input_tensor = preprocess_uploaded_image(original_image)
            current_method = "upload"
            
            # Dynamic instructions for upload
            st.info("✅ **Image uploaded successfully!** The AI analysis will appear automatically below.")
    
    with tab2:
        st.subheader("🖼️ Pre-loaded Sample Images")
        if data_loader:
            st.write("**Select a sample to immediately display and analyze:**")
            
            # Create columns for sample selection
            sample_cols = st.columns(5)
            
            for i in range(5):
                with sample_cols[i]:
                    if st.button(f"Sample {i+1}", key=f"sample_{i+1}", use_container_width=True):
                        try:
                            data_loaders = data_loader.get_loaders()
                            test_dataset = data_loaders['datasets']['test_clean']
                            
                            # Use fixed seed for consistent samples
                            np.random.seed(i * 123)  # Different seed for each sample
                            idx = np.random.randint(0, len(test_dataset))
                            sample_tensor, sample_label = test_dataset[idx]
                            
                            # Store in session state for persistence
                            st.session_state[f'sample_{i+1}_tensor'] = sample_tensor
                            st.session_state[f'sample_{i+1}_label'] = sample_label
                            st.session_state['selected_sample'] = i+1
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Failed to load sample: {str(e)}")
              # Display selected sample
            if 'selected_sample' in st.session_state:
                sample_id = st.session_state['selected_sample']
                if f'sample_{sample_id}_tensor' in st.session_state:
                    sample_tensor = st.session_state[f'sample_{sample_id}_tensor']
                    sample_label = st.session_state[f'sample_{sample_id}_label']
                    
                    original_image = tensor_to_image(sample_tensor)
                    
                    # Create smaller columns for the image display
                    img_col1, img_col2, img_col3 = st.columns([1, 2, 1])
                    with img_col2:
                        st.image(original_image, 
                                caption=f"Sample {sample_id} (True: {data_loader.classes[sample_label]})", 
                                width=200)  # Fixed width for smaller display
                    
                    input_tensor = sample_tensor.unsqueeze(0)
                    current_method = "sample"
                    
                    # Dynamic instructions for sample
                    st.success(f"✅ **Sample {sample_id} loaded!** This is a {data_loader.classes[sample_label]} from the test dataset.")
            
            # Alternative: Random sample generator
            st.markdown("---")
            st.write("**Or generate a completely random sample:**")
            if st.button("🎲 Generate Random Sample", key="random_sample", use_container_width=True):
                try:
                    data_loaders = data_loader.get_loaders()
                    test_dataset = data_loaders['datasets']['test_clean']
                      # Get a truly random sample
                    idx = np.random.randint(0, len(test_dataset))
                    sample_tensor, sample_label = test_dataset[idx]
                    
                    original_image = tensor_to_image(sample_tensor)
                    
                    # Create smaller columns for the image display
                    img_col1, img_col2, img_col3 = st.columns([1, 2, 1])
                    with img_col2:
                        st.image(original_image, caption=f"Random Sample (True: {data_loader.classes[sample_label]})", width=200)
                    
                    input_tensor = sample_tensor.unsqueeze(0)
                    current_method = "random"
                    
                    # Dynamic instructions for random
                    st.success(f"🎲 **Random sample generated!** This is a {data_loader.classes[sample_label]}.")
                    
                except Exception as e:
                    st.error(f"Failed to load sample: {str(e)}")
        else:
            st.error("Data loader not available. Please check your data directory.")
    
    with tab3:
        st.subheader("🧪 CIFAR-10 Test Dataset")
        if data_loader:
            test_type = st.selectbox("Test Set Type:", ["Clean", "Domain-Shifted"])
            
            if st.button("Get Random Test Image", use_container_width=True):
                try:
                    data_loaders = data_loader.get_loaders()
                    test_key = 'test_clean' if test_type == "Clean" else 'test_shifted'
                    test_dataset = data_loaders['datasets'][test_key]                    
                    idx = np.random.randint(0, len(test_dataset))
                    sample_tensor, sample_label = test_dataset[idx]
                    
                    original_image = tensor_to_image(sample_tensor)
                    
                    # Create smaller columns for the image display  
                    img_col1, img_col2, img_col3 = st.columns([1, 2, 1])
                    with img_col2:
                        st.image(original_image, caption=f"Test Image ({test_type}) - True: {data_loader.classes[sample_label]}", width=200)
                    
                    input_tensor = sample_tensor.unsqueeze(0)
                    current_method = "test"
                    
                    # Dynamic instructions for test
                    st.success(f"🧪 **Test image loaded!** This is a {test_type.lower()} {data_loader.classes[sample_label]} sample.")
                    
                except Exception as e:
                    st.error(f"Failed to load test image: {str(e)}")
        else:
            st.error("Data loader not available. Please check your data directory.")
    
    # Analysis section - only show when image is loaded
    if input_tensor is not None and original_image is not None:
        st.markdown("---")
        st.subheader("🤖 AI Analysis Results")
        
        # Analysis columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 Prediction & Trust")
            
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
                    st.subheader("� Trust Metrics")
                    metric_cols = st.columns(3)
                    
                    with metric_cols[0]:
                        st.metric("Entropy", f"{entropy:.3f}", help="Lower = more confident")
                    
                    with metric_cols[1]:
                        if 'odin_entropy' in results and results['odin_entropy'][0] is not None:
                            odin_val = results['odin_entropy'][0]
                            st.metric("ODIN Score", f"{odin_val:.3f}", help="OOD detection score")
                        else:
                            st.metric("ODIN Score", "N/A", help="ODIN computation failed or disabled")
                    
                    with metric_cols[2]:
                        if 'mahalanobis_distance' in results and results['mahalanobis_distance'][0] is not None:
                            mahal_val = results['mahalanobis_distance'][0] 
                            st.metric("Mahal. Dist.", f"{mahal_val:.3f}", help="Distance to training distribution")
                        else:
                            st.metric("Mahal. Dist.", "N/A", help="Mahalanobis detector not trained")
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.error(traceback.format_exc())
        
        with col2:
            st.subheader("🎯 Probability Distribution")
            
            if 'results' in locals():
                # Probability distribution
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
        
        # Full-width sections
        if 'results' in locals():
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
            
            # Create comprehensive trust metrics visualization
            dashboard_cols = st.columns([1, 1])
            
            with dashboard_cols[0]:
                # Trust gauge
                if 'trust_score' in locals():
                    gauge_fig = create_trust_gauge(trust_score, "Overall Trust Score")
                    st.plotly_chart(gauge_fig, use_container_width=True)
            
            with dashboard_cols[1]:
                # Detailed trust metrics with better formatting
                st.subheader("🎯 Trust Metrics Breakdown")
                
                # Entropy metric
                entropy_normalized = max(0, 1 - (entropy / np.log(10)))  # Normalize for display
                st.metric(
                    label="🎲 Prediction Certainty", 
                    value=f"{entropy_normalized:.3f}",
                    help="Higher values indicate more certain predictions. Based on entropy calculation."
                )
                
                # ODIN metric
                if 'odin_entropy' in results and results['odin_entropy'][0] is not None:
                    odin_score = max(0, 1 - (results['odin_entropy'][0] / np.log(10)))  # Normalize
                    st.metric(
                        label="🚨 OOD Detection Score", 
                        value=f"{odin_score:.3f}",
                        help="ODIN method score. Higher values suggest in-distribution samples."
                    )
                else:
                    st.metric(
                        label="🚨 OOD Detection Score", 
                        value="Not Available",
                        help="ODIN method requires gradient computation"
                    )
                
                # Mahalanobis distance metric  
                if 'mahalanobis_distance' in results and results['mahalanobis_distance'][0] is not None:
                    mahal_dist = results['mahalanobis_distance'][0]
                    # Normalize for display (lower distance is better)
                    mahal_score = max(0, 1 - (mahal_dist / 100))  # Rough normalization
                    st.metric(
                        label="📏 Distribution Similarity", 
                        value=f"{mahal_score:.3f}",
                        help=f"Raw Mahalanobis distance: {mahal_dist:.2f}. Higher scores indicate closer match to training data."
                    )
                else:
                    st.metric(
                        label="📏 Distribution Similarity", 
                        value="Computing...",
                        help="Mahalanobis distance measures similarity to training distribution"
                    )
            
            # Additional trust information
            st.subheader("📊 Trust Metrics Comparison")
            
            # Create a more informative bar chart
            trust_metrics = []
            metric_names = []
            metric_descriptions = []
            
            # Add entropy
            trust_metrics.append(max(0, 1 - (entropy / np.log(10))))
            metric_names.append("Prediction Certainty")
            metric_descriptions.append("Based on entropy")
            
            # Add ODIN if available
            if 'odin_entropy' in results and results['odin_entropy'][0] is not None:
                trust_metrics.append(max(0, 1 - (results['odin_entropy'][0] / np.log(10))))
                metric_names.append("ODIN Score")
                metric_descriptions.append("Out-of-distribution detection")
            
            # Add Mahalanobis if available
            if 'mahalanobis_distance' in results and results['mahalanobis_distance'][0] is not None:
                trust_metrics.append(max(0, 1 - (results['mahalanobis_distance'][0] / 100)))
                metric_names.append("Distribution Similarity")
                metric_descriptions.append("Mahalanobis distance")
            
            # Create DataFrame for better visualization
            if trust_metrics:
                trust_df = pd.DataFrame({
                    'Metric': metric_names,
                    'Score': trust_metrics,
                    'Description': metric_descriptions
                })
                
                # Use Streamlit's built-in bar chart with better formatting
                st.bar_chart(trust_df.set_index('Metric')['Score'])
                
                # Display table with details
                st.dataframe(
                    trust_df,
                    use_container_width=True,
                    hide_index=True
                )
    
    # Show instructions only when no image is loaded
    else:
        st.markdown("---")
        st.info("👆 **Select one of the tabs above to get started!** Choose Upload, Samples, or Test data to begin analyzing images.")
        
        # Dynamic getting started guide
        st.markdown("### 🚀 How to Get Started")
        
        guide_cols = st.columns(3)
        with guide_cols[0]:
            st.markdown("""
            **📤 Upload Your Image**
            - Click the "Upload Image" tab
            - Choose a PNG, JPG, or JPEG file
            - Image will be analyzed automatically
            """)
        
        with guide_cols[1]:
            st.markdown("""
            **🖼️ Try Sample Images**
            - Click the "Sample Images" tab
            - Choose from 5 pre-loaded samples
            - Or generate a random sample
            """)
        
        with guide_cols[2]:
            st.markdown("""
            **🧪 Use Test Data**  
            - Click the "CIFAR-10 Test" tab
            - Select clean or domain-shifted data
            - Get images from the test dataset
            """)
        
        st.markdown("---")
        st.markdown("**⚙️ Configure trust methods in the sidebar to customize your analysis!**")


if __name__ == "__main__":
    main()
