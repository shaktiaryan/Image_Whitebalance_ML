# 🚀 Google Colab Setup Guide - Aftershoot White Balance Prediction

## 📋 Quick Start Checklist

### 1. 🔧 Colab Environment Setup
- [ ] Open Google Colab: https://colab.research.google.com
- [ ] Enable GPU: Runtime → Change runtime type → GPU (T4 recommended)
- [ ] Upload the provided notebook: `aftershoot_colab_setup.ipynb`

### 2. 📁 Data Preparation

#### Option A: Google Drive Upload
1. **Upload your dataset to Google Drive:**
   ```
   /MyDrive/aftershoot_data/
   ├── Train/
   │   ├── images/          # Your TIFF images
   │   └── sliders.csv      # Training dataset CSV
   ├── Validation/
   │   ├── images/
   │   └── sliders.csv
   └── Test/
       ├── images/
       └── sliders.csv
   ```

#### Option B: Direct Upload to Colab
1. Use Colab's file upload feature
2. Upload to `/content/aftershoot_wb_prediction/data/`

### 3. 📦 Package Management (Choose One)

#### Option A: Poetry (Recommended)
1. **Upload** `pyproject.toml` and `poetry_colab_setup.py`
2. **Run Poetry setup:**
   ```python
   !python poetry_colab_setup.py
   ```
3. **Install dependencies:**
   ```python
   !poetry install --no-dev
   ```

#### Option B: Traditional pip
1. **Install dependencies manually:**
   ```python
   !pip install torch torchvision timm albumentations opencv-python pandas scikit-learn
   ```

### 3. 💻 Code Upload

#### Option A: Manual File Upload
1. Use Colab's Files panel (📁)
2. Upload all Python files from your local project:
   - `main.py`
   - All files from `src/` folder
   - Configuration files

#### Option B: Google Drive Method
1. Upload entire codebase to `/MyDrive/aftershoot_code/`
2. Run copy command in notebook

#### Option C: GitHub Integration
```bash
# If you have a GitHub repository
!git clone https://github.com/yourusername/aftershoot-wb-prediction.git
!cp -r aftershoot-wb-prediction/* /content/aftershoot_wb_prediction/
```

## ⚡ Quick Commands for Colab

### Option A: Poetry Setup (Recommended)
```python
# 1. One-click Poetry setup
!python poetry_colab_setup.py

# 2. Install dependencies with Poetry
!poetry install --no-dev

# 3. Check installation
!poetry show

# 4. Run commands with Poetry
!poetry run python main.py --eda --config efficientnet
```

### Option B: Traditional pip Setup
```python
# Install dependencies
!pip install torch torchvision timm==0.9.12 albumentations opencv-python pandas scikit-learn matplotlib seaborn

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Check GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "No GPU")
```

### Quick EDA Run
```python
%cd /content/aftershoot_wb_prediction

# With Poetry
!poetry run python main.py --eda --config efficientnet

# Or traditional
!python main.py --eda --config efficientnet
```

### Training Commands
```python
# Quick test (5 epochs)
!poetry run python main.py --train --config lightweight --epochs 5

# Full training  
!poetry run python main.py --train --config efficientnet

# Or use convenience scripts
!./run_test_training.sh
!./run_training.sh
```

### Monitor Training
```python
# View training plots
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('outputs/logs/training_latest.csv')
plt.plot(df['epoch'], df['val_loss'])
plt.show()
```

## 🔍 Troubleshooting

### Common Issues & Solutions

#### 1. **"No module named" errors**
```python
!pip install --upgrade <missing-package>
```

#### 2. **GPU not detected**
- Runtime → Change runtime type → GPU
- Check: Hardware accelerator = GPU

#### 3. **Out of memory errors**
```python
# Reduce batch size in config
config["training"]["batch_size"] = 16  # Instead of 32
```

#### 4. **Data not found**
```python
# Check data path
!ls -la /content/aftershoot_wb_prediction/data
!ls -la /content/drive/MyDrive/aftershoot_data
```

#### 5. **Training too slow**
```python
# Use lightweight config for testing
!poetry run python main.py --train --config lightweight --epochs 10
```

#### 6. **Poetry not found**
```python
# Install Poetry manually
!curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
!poetry --version
```

#### 7. **Dependency conflicts**
```python
# Update dependencies
!poetry update

# Or reinstall clean
!poetry install --sync
```

## 📊 Expected Performance

### Training Times (Google Colab T4 GPU)
- **EDA**: 2-5 minutes
- **Quick test (5 epochs)**: 10-15 minutes  
- **Full training (100 epochs)**: 1-3 hours
- **Lightweight model**: 30-60 minutes

### Performance Targets
- **Temperature MAE**: < 300K
- **Tint MAE**: < 10
- **Training loss**: < 0.5
- **Validation accuracy**: > 85%

## 💾 Saving Results

### Automatic Backup to Drive
```python
import shutil
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = f"/content/drive/MyDrive/aftershoot_results_{timestamp}"

# Save important folders
!cp -r outputs/checkpoints $backup_path/checkpoints
!cp -r outputs/eda $backup_path/eda
!cp -r outputs/logs $backup_path/logs
```

### Download Individual Files
```python
# Download specific files
from google.colab import files
files.download('outputs/checkpoints/best_model.pth')
files.download('outputs/eda/target_distributions.png')
```

## 🚀 Advanced Usage

### Hyperparameter Tuning
```python
# Try different configurations
configs = ['efficientnet', 'resnet', 'convnext', 'lightweight']
for config in configs:
    !python main.py --train --config {config} --epochs 20
```

### Custom Configuration
```python
import json

# Create custom config
custom_config = {
    "model": {"backbone": "efficientnet_b2", "dropout_rate": 0.4},
    "training": {"batch_size": 24, "learning_rate": 5e-5},
    "loss": {"temperature_weight": 1.2, "consistency_weight": 0.15}
}

with open('configs/custom.json', 'w') as f:
    json.dump(custom_config, f, indent=2)
    
!python main.py --train --config custom
```

### Resume Training
```python
# Resume from checkpoint
!python main.py --train --config efficientnet --resume outputs/checkpoints/checkpoint_epoch_25.pth
```

## 📈 Monitoring & Visualization

### Real-time Training Monitoring
```python
# Monitor during training
import time
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output

while True:
    if os.path.exists('outputs/logs/training_latest.csv'):
        df = pd.read_csv('outputs/logs/training_latest.csv')
        clear_output(wait=True)
        
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(df['epoch'], df['train_loss'], label='Train')
        plt.plot(df['epoch'], df['val_loss'], label='Val')
        plt.title('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(df['epoch'], df['val_temp_mae'])
        plt.title('Temperature MAE')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Latest epoch: {df['epoch'].iloc[-1]}")
        print(f"Val loss: {df['val_loss'].iloc[-1]:.4f}")
    
    time.sleep(30)  # Update every 30 seconds
```

## ✅ Success Checklist

After running the complete notebook, you should have:

- [ ] ✅ GPU enabled and detected
- [ ] ✅ All dependencies installed  
- [ ] ✅ Dataset loaded and verified
- [ ] ✅ EDA completed with 5 visualizations
- [ ] ✅ Training completed successfully
- [ ] ✅ Model checkpoints saved
- [ ] ✅ Training logs with metrics
- [ ] ✅ Results backed up to Google Drive

## 🎯 Next Steps

1. **Model Optimization**: Try different backbones and hyperparameters
2. **Advanced Augmentation**: Add color space transformations  
3. **Ensemble Methods**: Combine multiple models
4. **Production Ready**: Export to ONNX for deployment
5. **Custom Loss Functions**: Experiment with domain-specific losses

---

**🎉 You now have a fully functional Aftershoot White Balance prediction system running on Google Colab with GPU acceleration!**