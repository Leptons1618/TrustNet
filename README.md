# TrustLens: Explainable & Trust-Aware AI 🔍

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-brightgreen.svg)](https://streamlit.io/)

TrustLens is an explainable and trust-aware AI prototype that classifies images while revealing how confident the model really is. Built for transparency and reliability in AI decision-making.

## 🚀 Features

- **Image Classification**: CNN trained on CIFAR-10 with 10 classes
- **Uncertainty Quantification**: Entropy-based confidence scoring
- **Anomaly Detection**: Multiple methods including ODIN and Mahalanobis distance
- **Explainable AI**: Grad-CAM visualizations showing decision reasoning
- **Trust Calibration**: Temperature scaling for confidence calibration
- **Interactive Interface**: Streamlit-based web application
- **Real-time Analysis**: Upload images and get instant predictions with explanations

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch
- CUDA (optional, for GPU acceleration)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/Leptons1618/TrustNet.git
cd TrustNet
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download CIFAR-10 data (will be downloaded automatically on first run)

## 🎯 Usage

### Running the Application
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Features Overview

#### 1. Image Upload & Classification
- Upload any image or select from sample images
- Get predicted class with confidence score
- View preprocessing steps

#### 2. Trust Analysis
- **Entropy Score**: Measures prediction uncertainty
- **ODIN Score**: Out-of-distribution detection
- **Mahalanobis Distance**: Anomaly detection based on feature space
- **Calibrated Confidence**: Temperature-scaled probability

#### 3. Explainability
- **Grad-CAM Heatmaps**: Visual explanation of model decisions
- **Feature Importance**: Which parts of the image matter most
- **Decision Reasoning**: Step-by-step explanation

#### 4. Trust Dashboard
- Real-time trust metrics
- Historical analysis
- Confidence calibration plots

## 🏗️ Architecture

```
TrustNet/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn.py              # CNN architecture
│   │   └── trust_model.py      # Trust-aware model wrapper
│   ├── trust_methods/
│   │   ├── __init__.py
│   │   ├── entropy.py          # Entropy-based uncertainty
│   │   ├── odin.py            # ODIN method
│   │   ├── mahalanobis.py     # Mahalanobis distance
│   │   └── temperature.py     # Temperature scaling
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_loader.py     # Data preprocessing
│   │   ├── gradcam.py         # Grad-CAM implementation
│   │   └── visualization.py   # Plotting utilities
│   └── app.py                 # Main Streamlit application
├── assets/                    # Static files and sample images
├── models/                    # Trained model files
├── tests/                     # Unit tests
├── cookbook/                  # Jupyter notebooks for development
├── requirements.txt
├── README.md
└── .gitignore
```

## 🔬 Trust Methods

### 1. Entropy-Based Uncertainty
Measures prediction uncertainty using Shannon entropy:
```
H(p) = -∑ p_i * log(p_i)
```
- Low entropy = High confidence
- High entropy = High uncertainty

### 2. ODIN (Out-of-Distribution Detection)
Uses input preprocessing and temperature scaling to detect anomalous inputs:
- Applies small perturbations to inputs
- Uses temperature scaling on logits
- Effective for detecting domain shift

### 3. Mahalanobis Distance
Measures distance from class centroids in feature space:
- Computes distance to nearest class centroid
- Uses feature representations from penultimate layer
- Effective for detecting novel classes

### 4. Temperature Scaling
Calibrates model confidence to match actual accuracy:
- Post-processing technique
- Learns optimal temperature parameter
- Improves reliability of confidence scores

## 📊 Model Performance

### CIFAR-10 Classification
- **Clean Accuracy**: ~85%
- **Domain-Shifted Accuracy**: ~70%
- **Calibration Error**: <5% (after temperature scaling)

### Trust Methods Performance
- **Entropy AUC**: 0.85+ for OOD detection
- **ODIN AUC**: 0.90+ for OOD detection
- **Mahalanobis AUC**: 0.88+ for OOD detection

## 🎨 Sample Use Cases

1. **Quality Inspection**: Detect defective products with confidence scores
2. **Medical Triage**: Classify medical images with uncertainty quantification
3. **Security Screening**: Identify suspicious items with explainable decisions
4. **Autonomous Systems**: Make safety-critical decisions with trust metrics

## 🧪 Development

### Running Tests
```bash
pytest tests/
```

### Training New Models
```bash
python src/train.py --config config/train_config.yaml
```

### Jupyter Notebooks
Explore the `cookbook/` directory for development notebooks and experiments.

## 📈 Roadmap

- [ ] Support for additional datasets (ImageNet, custom datasets)
- [ ] Ensemble methods for improved trust
- [ ] Bayesian neural networks
- [ ] Active learning integration
- [ ] Mobile app deployment
- [ ] REST API for integration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- CIFAR-10 dataset creators
- PyTorch team for the deep learning framework
- Streamlit team for the web app framework
- Research papers on uncertainty quantification and explainable AI

## 📧 Contact

For questions or support, please open an issue on GitHub or contact [anishgiri163@gmail.com].

---

**Made with ❤️ for transparent and trustworthy AI**
