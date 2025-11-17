#!/usr/bin/env python3
"""
One-click setup script for Aftershoot White Balance Prediction on Google Colab
Run this cell first to set up everything automatically
"""

import os
import subprocess
import sys

def install_dependencies():
    """Install all required packages"""
    print("📦 Installing dependencies...")
    
    packages = [
        "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "timm==0.9.12",
        "albumentations==1.3.1", 
        "opencv-python==4.8.1.78",
        "pandas==2.1.4",
        "numpy==1.24.4",
        "scikit-learn==1.3.2",
        "matplotlib==3.8.2",
        "seaborn==0.13.0",
        "tqdm==4.66.1",
        "Pillow==10.1.0"
    ]
    
    for package in packages:
        print(f"Installing {package.split('==')[0]}...")
        subprocess.run([sys.executable, "-m", "pip", "install"] + package.split(), 
                      capture_output=True, text=True)
    
    print("✅ All dependencies installed!")

def setup_environment():
    """Setup the working environment"""
    print("🔧 Setting up environment...")
    
    # Check if in Colab
    try:
        import google.colab
        print("🚀 Running in Google Colab")
        IN_COLAB = True
    except ImportError:
        print("💻 Running locally")
        IN_COLAB = False
    
    # Check GPU
    import torch
    if torch.cuda.is_available():
        print(f"🔥 GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"📊 CUDA version: {torch.version.cuda}")
        print(f"🧠 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ No GPU detected!")
        print("💡 Enable GPU: Runtime → Change runtime type → Hardware accelerator: GPU")
    
    return IN_COLAB

def create_project_structure():
    """Create the project directory structure"""
    print("📁 Creating project structure...")
    
    # Change to content directory
    os.chdir('/content')
    
    # Create main project directory
    project_dir = '/content/aftershoot_wb_prediction'
    os.makedirs(project_dir, exist_ok=True)
    os.chdir(project_dir)
    
    # Create subdirectories
    directories = [
        'src/data',
        'src/models',
        'src/training', 
        'src/inference',
        'src/utils',
        'configs',
        'outputs/checkpoints',
        'outputs/logs',
        'outputs/eda',
        'outputs/predictions',
        'data/Train/images',
        'data/Validation/images',
        'data/Test/images',
        'notebooks'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Create __init__.py files
    init_files = [
        'src/__init__.py',
        'src/data/__init__.py',
        'src/models/__init__.py',
        'src/training/__init__.py',
        'src/inference/__init__.py',
        'src/utils/__init__.py'
    ]
    
    for init_file in init_files:
        with open(init_file, 'w') as f:
            f.write('# Package initialization\n')
    
    print(f"✅ Project structure created in {project_dir}")
    
    return project_dir

def create_configs():
    """Create configuration files"""
    print("⚙️ Creating configuration files...")
    
    import json
    
    configs = {
        'efficientnet': {
            "model": {
                "backbone": "efficientnet_b3",
                "pretrained": True,
                "dropout_rate": 0.3,
                "mlp_hidden_dims": [256, 128, 64],
                "mlp_dropout": 0.2
            },
            "training": {
                "batch_size": 32,
                "learning_rate": 1e-4,
                "epochs": 100,
                "weight_decay": 1e-5,
                "patience": 15,
                "min_lr": 1e-7
            },
            "loss": {
                "temperature_weight": 1.0,
                "tint_weight": 1.0,
                "consistency_weight": 0.1,
                "temperature_aware_weighting": True
            },
            "augmentation": {
                "horizontal_flip_p": 0.5,
                "rotation_limit": 15,
                "brightness_limit": 0.2,
                "contrast_limit": 0.2,
                "gaussian_noise_p": 0.3,
                "blur_limit": 3,
                "blur_p": 0.2
            }
        },
        'lightweight': {
            "model": {
                "backbone": "efficientnet_b0",
                "pretrained": True,
                "dropout_rate": 0.2,
                "mlp_hidden_dims": [128, 64],
                "mlp_dropout": 0.1
            },
            "training": {
                "batch_size": 64,
                "learning_rate": 2e-4,
                "epochs": 50,
                "weight_decay": 1e-5,
                "patience": 10,
                "min_lr": 1e-7
            },
            "loss": {
                "temperature_weight": 1.0,
                "tint_weight": 1.0,
                "consistency_weight": 0.05,
                "temperature_aware_weighting": True
            },
            "augmentation": {
                "horizontal_flip_p": 0.3,
                "rotation_limit": 10,
                "brightness_limit": 0.1,
                "contrast_limit": 0.1,
                "gaussian_noise_p": 0.2,
                "blur_limit": 2,
                "blur_p": 0.1
            }
        }
    }
    
    for config_name, config_data in configs.items():
        config_path = f"configs/{config_name}.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        print(f"✅ Created {config_path}")

def mount_drive():
    """Mount Google Drive"""
    try:
        from google.colab import drive
        print("📂 Mounting Google Drive...")
        drive.mount('/content/drive')
        print("✅ Google Drive mounted!")
        return True
    except:
        print("⚠️ Could not mount Google Drive (not in Colab environment)")
        return False

def print_next_steps():
    """Print instructions for next steps"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    print("📁 Project created in: /content/aftershoot_wb_prediction")
    print("\n📋 NEXT STEPS:")
    print("1. 📤 Upload your code files:")
    print("   - Use Files panel (📁) to upload Python files")
    print("   - Or copy from Drive: !cp -r /content/drive/MyDrive/your_code/* .")
    print("\n2. 📊 Upload your data:")
    print("   - Upload to: data/Train/, data/Validation/, data/Test/")
    print("   - Required files: sliders.csv + images/ folder in each")
    print("   - Or link from Drive: !ln -s /content/drive/MyDrive/aftershoot_data data")
    print("\n3. 🔍 Run EDA:")
    print("   !python main.py --eda --config efficientnet")
    print("\n4. 🚀 Start training:")
    print("   !python main.py --train --config efficientnet")
    print("\n💡 Quick test with sample data:")
    print("   !python data_setup_colab.py  # Choose option 3")
    print("\n🔗 Useful commands:")
    print("   !nvidia-smi  # Check GPU status")
    print("   !ls -la data/  # Check data structure")
    print("   !tail outputs/logs/training_*.log  # Monitor training")
    
def main():
    """Main setup function"""
    print("🚀 Aftershoot White Balance Prediction - Colab Setup")
    print("=" * 60)
    print("Setting up complete ML environment for white balance prediction...")
    print("This will take 2-3 minutes.\n")
    
    # Run setup steps
    install_dependencies()
    IN_COLAB = setup_environment()
    project_dir = create_project_structure()
    create_configs()
    
    if IN_COLAB:
        mount_drive()
    
    print_next_steps()
    
    # Set working directory
    os.chdir(project_dir)
    print(f"\n📍 Current directory: {os.getcwd()}")
    
    return project_dir

if __name__ == "__main__":
    main()