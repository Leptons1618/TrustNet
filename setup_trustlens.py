"""
Setup script for TrustLens
"""

import subprocess
import sys
import os
import shutil


def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False


def check_model_file():
    """Check if model file exists and copy if needed"""
    print("🔍 Checking for model file...")
    
    model_paths = [
        "cookbook/trustnet_cnn.pth",
        "cookbook/og_Model/trustnet_cnn.pth",
        "og_Model/trustnet_cnn.pth"
    ]
    
    target_path = "models/trustnet_cnn.pth"
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Check if target already exists
    if os.path.exists(target_path):
        print("✅ Model file already exists at models/trustnet_cnn.pth")
        return True
    
    # Look for model file in other locations
    for path in model_paths:
        if os.path.exists(path):
            print(f"📋 Copying model from {path} to {target_path}")
            shutil.copy2(path, target_path)
            print("✅ Model file copied successfully!")
            return True
    
    print("⚠️ Model file not found. You may need to:")
    print("   1. Train the model using the notebook in cookbook/")
    print("   2. Place trustnet_cnn.pth in the models/ directory")
    return False


def create_data_directory():
    """Create data directory structure"""
    print("📁 Creating data directory...")
    
    data_dir = "cookbook/data"
    os.makedirs(data_dir, exist_ok=True)
    
    print("✅ Data directory created!")
    return True


def run_demo():
    """Run the demo script"""
    print("🚀 Running demo test...")
    try:
        result = subprocess.run([sys.executable, "demo.py"], 
                              capture_output=True, text=True, timeout=60)
        
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("❌ Demo test timed out")
        return False
    except Exception as e:
        print(f"❌ Failed to run demo: {e}")
        return False


def main():
    """Main setup function"""
    print("🔧 TrustLens Setup Script")
    print("=" * 50)
    
    steps = [
        ("Installing Requirements", install_requirements),
        ("Checking Model File", check_model_file),
        ("Creating Data Directory", create_data_directory),
        ("Running Demo Test", run_demo),
    ]
    
    success_count = 0
    
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        try:
            if step_func():
                success_count += 1
            else:
                print(f"⚠️ {step_name} completed with warnings")
        except Exception as e:
            print(f"❌ {step_name} failed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Setup completed: {success_count}/{len(steps)} steps successful")
    
    if success_count >= len(steps) - 1:  # Allow demo to fail
        print("\n🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("   1. Run the application: streamlit run app.py")
        print("   2. Open your browser to: http://localhost:8501")
        print("   3. Upload an image and explore the trust analysis!")
    else:
        print("\n⚠️ Setup completed with issues. Check the output above.")
    
    return success_count >= len(steps) - 1


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
