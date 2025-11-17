from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import os

@dataclass
class DataConfig:
    """Data configuration"""
    train_images_path: str = "data/Train/images"
    train_csv_path: str = "data/Train/sliders.csv"
    val_images_path: str = "data/Validation/images"
    val_csv_path: str = "data/Validation/sliders_inputs.csv"
    
    # Image processing
    image_size: Tuple[int, int] = (256, 256)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Data splits
    train_val_split: float = 0.9  # 90% for training, 10% for validation
    random_seed: int = 42

@dataclass
class ModelConfig:
    """Model configuration"""
    # Image encoder
    backbone: str = "efficientnet_b3"  # or "resnet50", "convnext_base"
    pretrained: bool = True
    image_feature_dim: int = 512
    
    # Metadata encoder
    categorical_features: List[str] = None
    numerical_features: List[str] = None
    metadata_hidden_dims: List[int] = None
    
    # Fusion and output
    fusion_dim: int = 256
    dropout_rate: float = 0.3
    output_dim: int = 2  # Temperature and Tint
    
    def __post_init__(self):
        if self.categorical_features is None:
            self.categorical_features = ["camera_model", "camera_group", "flashFired"]
        
        if self.numerical_features is None:
            self.numerical_features = [
                "currTemp", "currTint", "aperture", "focalLength", 
                "isoSpeedRating", "shutterSpeed", "intensity", "ev"
            ]
        
        if self.metadata_hidden_dims is None:
            self.metadata_hidden_dims = [128, 64]

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    epochs: int = 100
    
    # Learning rate scheduling
    lr_scheduler: str = "cosine"  # "cosine", "step", "reduce_on_plateau"
    lr_warmup_epochs: int = 5
    lr_min: float = 1e-6
    
    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4
    
    # Loss function weights
    temperature_weight: float = 1.0
    tint_weight: float = 1.0
    consistency_weight: float = 0.1
    
    # Model saving
    save_best_model: bool = True
    save_checkpoint_freq: int = 10
    
    # Logging
    log_freq: int = 50
    use_wandb: bool = False
    wandb_project: str = "aftershoot-wb-prediction"

@dataclass
class AugmentationConfig:
    """Data augmentation configuration"""
    # Probability of applying each augmentation
    horizontal_flip_p: float = 0.5
    rotation_p: float = 0.3
    color_jitter_p: float = 0.3
    gaussian_blur_p: float = 0.2
    gaussian_noise_p: float = 0.2
    
    # Augmentation parameters
    rotation_limit: int = 15
    brightness_limit: float = 0.1
    contrast_limit: float = 0.1
    saturation_limit: float = 0.1
    hue_limit: int = 5
    blur_limit: int = 3
    noise_var_limit: Tuple[float, float] = (0.0, 0.01)

@dataclass
class Config:
    """Main configuration class"""
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    augmentation: AugmentationConfig
    
    # Paths
    output_dir: str = "outputs"
    checkpoint_dir: str = "outputs/checkpoints"
    logs_dir: str = "outputs/logs"
    predictions_dir: str = "outputs/predictions"
    
    # Device
    device: str = "cuda"  # Will be set automatically
    num_workers: int = 0  # 0 disables multiprocessing, safer on Windows
    
    def __post_init__(self):
        # Create output directories
        for dir_path in [self.output_dir, self.checkpoint_dir, self.logs_dir, self.predictions_dir]:
            os.makedirs(dir_path, exist_ok=True)

def get_default_config() -> Config:
    """Get default configuration"""
    return Config(
        data=DataConfig(),
        model=ModelConfig(),
        training=TrainingConfig(),
        augmentation=AugmentationConfig()
    )