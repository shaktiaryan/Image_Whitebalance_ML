# 🎯 Aftershoot White Balance Prediction - ML Solution

**Professional ML solution for automatic white balance correction using EfficientNet-B0**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)

## 🚀 Features

- **🎯 High Accuracy**: EfficientNet-B0 backbone with 4.9M parameters
- **⚡ GPU Accelerated**: CUDA-optimized training and inference
- **📊 Comprehensive EDA**: Detailed dataset analysis and visualizations
- **🔧 Professional Workflow**: Poetry package management and reproducible environment
- **📈 Performance Tracking**: Training curves, metrics, and model evaluation
- **🎨 Visual Analysis**: Prediction scatter plots and error distributions

## 📊 Model Performance

- **Best Validation Score**: 0.049909 (Very Good Performance)
- **Temperature MAE**: ~2,684K
- **Tint MAE**: ~8.44
- **Training Data**: 816 images (90% split optimization)
- **Validation Data**: 91 images (10% split)

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA 12.x
- Git

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/shaktiaryan/Image_Whitebalance_ML.git
cd Image_Whitebalance_ML
```

2. **Create virtual environment**:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install timm albumentations opencv-python pandas numpy scikit-learn matplotlib seaborn tqdm Pillow
```

4. **Optional - Poetry setup**:
```bash
# Install Poetry (Windows)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
poetry install
```

## 📁 Project Structure

```
aftershoot_wb_prediction/
├── src/                          # Source code
│   ├── data/                     # Data loading and preprocessing
│   ├── models/                   # Model architectures
│   ├── training/                 # Training pipeline
│   ├── inference/                # Inference utilities
│   └── utils/                    # Utility functions
├── configs/                      # Configuration files
├── data/                         # Dataset
│   ├── Train/                    # Training data
│   ├── Validation/              # Validation data
│   └── Test/                    # Test data
├── outputs/                     # Training outputs
│   ├── checkpoints/             # Model checkpoints
│   ├── logs/                    # Training logs
│   ├── eda/                     # EDA visualizations
│   └── model_testing/           # Testing results
├── notebooks/                   # Jupyter notebooks
├── main.py                      # Main training script
├── test_model.py               # Model testing script
└── aftershoot_poetry_colab.ipynb  # Complete workflow notebook
```

## 🎯 Quick Start

### 1. Data Preparation
Place your images and CSV files in the appropriate directories:
- `data/Train/images/` - Training images
- `data/Train/sliders_filtered.csv` - Training labels
- `data/Validation/images/` - Validation images
- `data/Validation/sliders_inputs.csv` - Validation metadata

### 2. Run EDA (Exploratory Data Analysis)
```bash
python main.py --eda --config efficientnet
```

### 3. Train the Model
```bash
# Quick test (5 epochs)
python main.py --config lightweight --epochs 5 --device cuda

# Full training (15 epochs)
python main.py --config lightweight --epochs 15 --device cuda --output_dir outputs/training_run
```

### 4. Test the Model
```bash
python test_model.py
```

### 5. Using Jupyter Notebook
Open `aftershoot_poetry_colab.ipynb` for an interactive workflow with:
- Step-by-step setup instructions
- EDA analysis and insights
- Training monitoring
- Results visualization

## 🔧 Configuration

The project uses a flexible configuration system in `configs/base_config.py`:

```python
# Data split optimization (90/10 for maximum training data utilization)
train_val_split: float = 0.9

# Model configuration
backbone: str = "efficientnet_b0"
image_size: Tuple[int, int] = (256, 256)

# Training parameters
batch_size: int = 32
learning_rate: float = 2e-4
epochs: int = 15
```

## 📈 Training Optimization

The project implements several optimizations:

1. **Data Split Optimization**: 90/10 split instead of traditional 80/20
2. **GPU Acceleration**: CUDA-optimized training pipeline
3. **Advanced Augmentation**: Albumentations library for robust data augmentation
4. **Learning Rate Scheduling**: Adaptive learning rate with warmup
5. **Early Stopping**: Prevents overfitting with validation monitoring

## 🎨 Model Architecture

- **Backbone**: EfficientNet-B0 (pretrained on ImageNet)
- **Input Size**: 256x256 RGB images
- **Output**: Temperature (Kelvin) and Tint values
- **Loss Function**: Combined MSE loss for both outputs
- **Optimizer**: AdamW with weight decay

## 📊 Results

### Training Performance
- **Best Model**: `optimized_90_split` run
- **Validation Score**: 0.049909
- **Training Time**: ~1-3 hours (GPU dependent)
- **Memory Usage**: ~6GB VRAM (GTX 1660 Ti compatible)

### Error Analysis
- **Temperature**: 30.8% predictions within ±1000K
- **Tint**: 64.8% predictions within ±10 units
- **Overall**: Very good performance for real-world application

## 🔬 Development Workflow

1. **Setup Environment**: Virtual environment and dependencies
2. **Data Analysis**: Comprehensive EDA with visualizations
3. **Model Training**: Optimized training pipeline with monitoring
4. **Evaluation**: Detailed testing with metrics and plots
5. **Deployment**: Production-ready model checkpoints

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Aftershoot**: For the original dataset and problem formulation
- **timm**: For the excellent model library
- **PyTorch**: For the deep learning framework
- **Albumentations**: For advanced data augmentation

## 📞 Contact

**Shakti Aryan**
- GitHub: [@shaktiaryan](https://github.com/shaktiaryan)
- Project Link: [https://github.com/shaktiaryan/Image_Whitebalance_ML](https://github.com/shaktiaryan/Image_Whitebalance_ML)

---

**⭐ If you find this project helpful, please give it a star!**