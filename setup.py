"""
Setup script for TrustLens
"""

import os
import subprocess
import sys
import shutil
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. Current version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python version {version.major}.{version.minor} is compatible")
    return True


def install_requirements():
    """Install required packages"""
    requirements_file = "requirements.txt"
    if not os.path.exists(requirements_file):
        print(f"❌ {requirements_file} not found")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r {requirements_file}",
        "Installing requirements"
    )


def setup_directories():
    """Create necessary directories"""
    print("\n📁 Setting up directories...")
    
    directories = [
        "models",
        "data",
        "assets",
        "logs",
        "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created/verified directory: {directory}")
    
    return True


def copy_model_file():
    """Copy model file to models directory if it exists in cookbook"""
    cookbook_model = "cookbook/trustnet_cnn.pth"
    models_dir = "models/trustnet_cnn.pth"
    
    if os.path.exists(cookbook_model) and not os.path.exists(models_dir):
        print(f"\n📋 Copying model file...")
        try:
            shutil.copy2(cookbook_model, models_dir)
            print(f"✅ Copied model from {cookbook_model} to {models_dir}")
            return True
        except Exception as e:
            print(f"❌ Failed to copy model file: {e}")
            return False
    elif os.path.exists(models_dir):
        print(f"✅ Model file already exists at {models_dir}")
        return True
    else:
        print(f"⚠️  Model file not found. You may need to train the model first.")
        return True  # Not a critical error


def run_tests():
    """Run basic tests to verify installation"""
    print("\n🧪 Running basic tests...")
    
    test_command = f"{sys.executable} -m pytest tests/ -v"
    if shutil.which('pytest'):
        return run_command(test_command, "Running tests")
    else:
        print("⚠️  pytest not found, skipping tests")
        return True


def main():
    """Main setup function"""
    print("🚀 TrustLens Setup Script")
    print("=" * 50)
    
    steps = [
        ("Check Python version", check_python_version),
        ("Install requirements", install_requirements),
        ("Setup directories", setup_directories),
        ("Copy model file", copy_model_file),
        ("Run tests", run_tests),
    ]
    
    failed_steps = []
    
    for step_name, step_function in steps:
        if not step_function():
            failed_steps.append(step_name)
    
    print("\n" + "=" * 50)
    
    if not failed_steps:
        print("🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Run the application: streamlit run app.py")
        print("2. Open your browser to the URL shown (usually http://localhost:8501)")
        print("3. Upload an image or use sample data to test the system")
        
        if not os.path.exists("models/trustnet_cnn.pth") and not os.path.exists("cookbook/trustnet_cnn.pth"):
            print("\n⚠️  Note: No trained model found. You may need to:")
            print("   - Train a model using the notebook in cookbook/")
            print("   - Or download a pre-trained model")
    else:
        print("❌ Setup completed with some issues:")
        for step in failed_steps:
            print(f"   - {step}")
        print("\nPlease resolve these issues before running the application.")
    
    return len(failed_steps) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
