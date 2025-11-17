#!/usr/bin/env python3
"""
Poetry-based setup script for Aftershoot White Balance Prediction on Google Colab
Installs Poetry and manages dependencies through pyproject.toml
"""

import os
import subprocess
import sys
import json
from pathlib import Path

def install_poetry():
    """Install Poetry package manager"""
    print("📦 Installing Poetry...")
    
    # Check if Poetry is already installed
    try:
        result = subprocess.run(["poetry", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Poetry already installed: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    # Install Poetry using official installer
    print("🔧 Installing Poetry package manager...")
    
    # Download and install Poetry
    install_cmd = [
        sys.executable, "-c",
        "import urllib.request; "
        "exec(urllib.request.urlopen('https://install.python-poetry.org').read())"
    ]
    
    result = subprocess.run(install_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Poetry installation failed: {result.stderr}")
        return False
    
    # Add Poetry to PATH
    poetry_bin = os.path.expanduser("~/.local/bin")
    current_path = os.environ.get("PATH", "")
    if poetry_bin not in current_path:
        os.environ["PATH"] = f"{poetry_bin}:{current_path}"
    
    # Verify installation
    try:
        result = subprocess.run([f"{poetry_bin}/poetry", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Poetry installed successfully: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ Poetry installation verification failed")
    return False

def configure_poetry():
    """Configure Poetry for Colab environment"""
    print("⚙️ Configuring Poetry for Colab...")
    
    poetry_cmd = "poetry"
    if not subprocess.run(["which", poetry_cmd], capture_output=True).returncode == 0:
        poetry_cmd = os.path.expanduser("~/.local/bin/poetry")
    
    # Configure Poetry settings for Colab
    configs = [
        ("virtualenvs.create", "false"),  # Don't create venv in Colab
        ("virtualenvs.in-project", "false"),
        ("installer.parallel", "true"),
        ("experimental.new-installer", "true")
    ]
    
    for setting, value in configs:
        cmd = [poetry_cmd, "config", setting, value]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Set {setting} = {value}")
        else:
            print(f"⚠️ Failed to set {setting}: {result.stderr}")

def install_dependencies():
    """Install dependencies using Poetry"""
    print("📦 Installing dependencies with Poetry...")
    
    poetry_cmd = "poetry"
    if not subprocess.run(["which", poetry_cmd], capture_output=True).returncode == 0:
        poetry_cmd = os.path.expanduser("~/.local/bin/poetry")
    
    # Install main dependencies
    print("Installing main dependencies...")
    result = subprocess.run([poetry_cmd, "install", "--no-dev"], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Main dependencies installation failed: {result.stderr}")
        # Fallback to pip installation
        print("📦 Falling back to pip installation...")
        return install_with_pip_fallback()
    
    print("✅ Main dependencies installed with Poetry!")
    
    # Install dev dependencies (optional)
    print("Installing development dependencies...")
    result = subprocess.run([poetry_cmd, "install", "--only", "dev"], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Development dependencies installed!")
    else:
        print(f"⚠️ Dev dependencies installation failed: {result.stderr}")
    
    return True

def install_with_pip_fallback():
    """Fallback to pip installation if Poetry fails"""
    print("📦 Using pip fallback installation...")
    
    # Read dependencies from pyproject.toml
    try:
        import tomllib
    except ImportError:
        # For Python < 3.11
        try:
            import tomli as tomllib
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "tomli"])
            import tomli as tomllib
    
    if os.path.exists("pyproject.toml"):
        with open("pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)
        
        deps = pyproject.get("tool", {}).get("poetry", {}).get("dependencies", {})
        
        # Install each dependency
        for package, version in deps.items():
            if package == "python":
                continue
            
            # Parse version constraint
            if isinstance(version, str):
                if version.startswith("^"):
                    version = f">={version[1:]}"
                elif version.startswith("~"):
                    version = f">={version[1:]}"
                
                package_spec = f"{package}{version}" if version != "*" else package
            else:
                package_spec = package
            
            print(f"Installing {package_spec}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package_spec])
        
        print("✅ Dependencies installed with pip!")
        return True
    
    print("❌ No pyproject.toml found for fallback")
    return False

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
        'notebooks',
        'tests'
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

def mount_drive():
    """Mount Google Drive"""
    try:
        from google.colab import drive
        print("📂 Mounting Google Drive...")
        drive.mount('/content/drive')
        print("✅ Google Drive mounted!")
        return True
    except Exception as e:
        print(f"⚠️ Could not mount Google Drive: {e}")
        return False

def create_poetry_scripts():
    """Create Poetry convenience scripts"""
    print("📝 Creating Poetry convenience scripts...")
    
    scripts = {
        'run_eda.sh': """#!/bin/bash
# Run EDA with Poetry
poetry run python main.py --eda --config efficientnet
""",
        'run_training.sh': """#!/bin/bash
# Run training with Poetry
poetry run python main.py --train --config efficientnet
""",
        'run_test_training.sh': """#!/bin/bash
# Run quick test training with Poetry
poetry run python main.py --train --config lightweight --epochs 5
""",
        'install_deps.sh': """#!/bin/bash
# Install dependencies with Poetry
poetry install --no-dev
""",
        'update_deps.sh': """#!/bin/bash
# Update dependencies with Poetry
poetry update
"""
    }
    
    for script_name, script_content in scripts.items():
        with open(script_name, 'w') as f:
            f.write(script_content)
        os.chmod(script_name, 0o755)  # Make executable
        print(f"✅ Created {script_name}")

def print_poetry_usage():
    """Print Poetry usage instructions"""
    print("\n" + "="*60)
    print("🎉 POETRY SETUP COMPLETE!")
    print("="*60)
    print("📁 Project created in: /content/aftershoot_wb_prediction")
    print("\n📋 POETRY COMMANDS:")
    print("🔧 Environment Management:")
    print("   poetry install                    # Install all dependencies")
    print("   poetry install --no-dev          # Install only main dependencies")
    print("   poetry update                    # Update all dependencies")
    print("   poetry show                      # List installed packages")
    print("   poetry show --tree              # Show dependency tree")
    print("\n🚀 Running Scripts:")
    print("   poetry run python main.py --eda --config efficientnet")
    print("   poetry run python main.py --train --config efficientnet")
    print("   poetry run python main.py --train --config lightweight --epochs 5")
    print("\n🛠️ Development:")
    print("   poetry add <package>             # Add new dependency")
    print("   poetry add --group dev <package> # Add dev dependency")
    print("   poetry remove <package>         # Remove dependency")
    print("   poetry shell                    # Activate virtual environment")
    print("\n📦 Package Management:")
    print("   poetry build                    # Build package")
    print("   poetry publish                  # Publish to PyPI")
    print("   poetry export -f requirements.txt --output requirements.txt")
    print("\n🎯 Quick Start Scripts:")
    print("   ./run_eda.sh                    # Run EDA")
    print("   ./run_training.sh               # Run full training")
    print("   ./run_test_training.sh          # Run quick test")
    print("\n💡 Useful Poetry Tips:")
    print("   - Dependencies are managed in pyproject.toml")
    print("   - Lock file (poetry.lock) ensures reproducible builds")
    print("   - Use poetry add instead of pip install")
    print("   - Virtual environment is managed automatically")

def main():
    """Main setup function with Poetry"""
    print("🚀 Aftershoot White Balance Prediction - Poetry Setup")
    print("=" * 60)
    print("Setting up complete ML environment with Poetry package manager...")
    print("This will take 3-5 minutes.\n")
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        print("❌ Python 3.8+ required for Poetry")
        return
    
    print(f"✅ Python {python_version.major}.{python_version.minor} detected")
    
    # Run setup steps
    IN_COLAB = setup_environment()
    project_dir = create_project_structure()
    
    if IN_COLAB:
        mount_drive()
    
    # Install and configure Poetry
    if install_poetry():
        configure_poetry()
        
        # Copy pyproject.toml if it exists
        pyproject_source = "/content/drive/MyDrive/aftershoot_code/pyproject.toml"
        if os.path.exists(pyproject_source):
            import shutil
            shutil.copy2(pyproject_source, "pyproject.toml")
            print("✅ Copied pyproject.toml from Drive")
        
        # Install dependencies
        if install_dependencies():
            create_poetry_scripts()
            print_poetry_usage()
        else:
            print("❌ Dependency installation failed")
    else:
        print("❌ Poetry installation failed, falling back to pip")
        install_with_pip_fallback()
    
    # Set working directory
    os.chdir(project_dir)
    print(f"\n📍 Current directory: {os.getcwd()}")
    
    return project_dir

if __name__ == "__main__":
    main()