"""
Demo script to test TrustLens functionality
"""

import sys
import os
import torch
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_model_loading():
    """Test if the model can be loaded"""
    print("Testing model loading...")
    
    try:
        from src.models import TrustModel
        
        device = torch.device('cpu')  # Use CPU for testing
        trust_model = TrustModel(model_path='models/trustnet_cnn.pth', device=device)
        print("Model loaded successfully!")
        return trust_model
    except Exception as e:
        print(f"Model loading failed: {e}")
        return None

def test_inference():
    """Test model inference"""
    print("\nTesting inference...")
    
    trust_model = test_model_loading()
    if trust_model is None:
        return False
    
    try:
        # Create dummy input
        dummy_input = torch.randn(1, 3, 32, 32)
        
        # Run inference
        results = trust_model.predict_with_trust(dummy_input)
        
        print("Inference successful!")
        print(f"   Prediction: {results['predictions'][0]}")
        print(f"   Confidence: {results['max_probability'][0]:.3f}")
        print(f"   Entropy: {results['entropy'][0]:.3f}")
        
        return True
    except Exception as e:
        print(f"Inference failed: {e}")
        return False

def test_data_loading():
    """Test data loading"""
    print("\n🔍 Testing data loading...")
    
    try:
        from src.utils import CIFAR10DataLoader
        
        data_loader = CIFAR10DataLoader(data_dir='cookbook/data', batch_size=4)
        loaders = data_loader.get_loaders()
        
        print("✅ Data loading successful!")
        print(f"   Classes: {data_loader.classes}")
        
        return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_trust_methods():
    """Test trust computation methods"""
    print("\n🔍 Testing trust methods...")
    
    try:
        from src.trust_methods import compute_entropy
        
        # Test entropy computation
        dummy_logits = torch.randn(4, 10)
        entropy = compute_entropy(dummy_logits)
        
        print("✅ Trust methods working!")
        print(f"   Sample entropy: {entropy[0]:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Trust methods failed: {e}")
        return False

def main():
    """Main demo function"""
    print("TrustLens Demo & Test Script")
    print("=" * 50)
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Data Loading", test_data_loading),
        ("Trust Methods", test_trust_methods),
        ("Inference", test_inference),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! TrustLens is ready to use.")
        print("\n📋 To run the application:")
        print("   streamlit run app.py")
        print("\n🌐 Then open your browser to:")
        print("   http://localhost:8501")
    else:
        print("⚠️  Some tests failed. Please check the setup.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
