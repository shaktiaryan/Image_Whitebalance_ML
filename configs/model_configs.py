from configs.base_config import Config, DataConfig, ModelConfig, TrainingConfig, AugmentationConfig

def get_efficientnet_config() -> Config:
    """Configuration for EfficientNet-based model"""
    return Config(
        data=DataConfig(),
        model=ModelConfig(
            backbone="efficientnet_b3",
            pretrained=True,
            image_feature_dim=512,
            fusion_dim=256,
            dropout_rate=0.3
        ),
        training=TrainingConfig(
            learning_rate=1e-4,
            batch_size=32,
            epochs=100
        ),
        augmentation=AugmentationConfig()
    )

def get_resnet_config() -> Config:
    """Configuration for ResNet-based model"""
    return Config(
        data=DataConfig(),
        model=ModelConfig(
            backbone="resnet50",
            pretrained=True,
            image_feature_dim=2048,
            fusion_dim=512,
            dropout_rate=0.4
        ),
        training=TrainingConfig(
            learning_rate=5e-5,
            batch_size=16,
            epochs=80
        ),
        augmentation=AugmentationConfig()
    )

def get_convnext_config() -> Config:
    """Configuration for ConvNeXt-based model"""
    return Config(
        data=DataConfig(),
        model=ModelConfig(
            backbone="convnext_base",
            pretrained=True,
            image_feature_dim=1024,
            fusion_dim=512,
            dropout_rate=0.2
        ),
        training=TrainingConfig(
            learning_rate=2e-4,
            batch_size=24,
            epochs=120,
            lr_warmup_epochs=10
        ),
        augmentation=AugmentationConfig(
            horizontal_flip_p=0.3,
            rotation_p=0.2,
            color_jitter_p=0.4
        )
    )

def get_lightweight_config() -> Config:
    """Configuration for lightweight/fast model"""
    return Config(
        data=DataConfig(),
        model=ModelConfig(
            backbone="efficientnet_b0",
            pretrained=True,
            image_feature_dim=256,
            fusion_dim=128,
            dropout_rate=0.2,
            metadata_hidden_dims=[64, 32]
        ),
        training=TrainingConfig(
            learning_rate=2e-4,
            batch_size=64,
            epochs=150
        ),
        augmentation=AugmentationConfig()
    )

# Model configuration registry
MODEL_CONFIGS = {
    "efficientnet": get_efficientnet_config,
    "resnet": get_resnet_config,
    "convnext": get_convnext_config,
    "lightweight": get_lightweight_config
}