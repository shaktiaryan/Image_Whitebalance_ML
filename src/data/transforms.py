import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from typing import Dict, Any
import numpy as np

class ImageTransforms:
    """Image transformation pipeline for training and validation"""
    
    def __init__(self, image_size: tuple = (256, 256), 
                 normalize_mean: tuple = (0.485, 0.456, 0.406),
                 normalize_std: tuple = (0.229, 0.224, 0.225)):
        self.image_size = image_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
    
    def get_train_transforms(self, aug_config) -> A.Compose:
        """Get training transforms with augmentations"""
        transforms = [
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            
            # Geometric transformations
            A.HorizontalFlip(p=aug_config.horizontal_flip_p),
            A.Rotate(limit=aug_config.rotation_limit, p=aug_config.rotation_p),
            
            # Color augmentations
            A.ColorJitter(
                brightness=aug_config.brightness_limit,
                contrast=aug_config.contrast_limit,
                saturation=aug_config.saturation_limit,
                hue=aug_config.hue_limit,
                p=aug_config.color_jitter_p
            ),
            
            # Blur and noise
            A.GaussianBlur(blur_limit=aug_config.blur_limit, p=aug_config.gaussian_blur_p),
            A.GaussNoise(var_limit=0.01, p=aug_config.gaussian_noise_p),
            
            # Normalization and tensor conversion
            A.Normalize(mean=self.normalize_mean, std=self.normalize_std),
            ToTensorV2(),
        ]
        
        return A.Compose(transforms)
    
    def get_val_transforms(self) -> A.Compose:
        """Get validation transforms (no augmentations)"""
        transforms = [
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            A.Normalize(mean=self.normalize_mean, std=self.normalize_std),
            ToTensorV2(),
        ]
        
        return A.Compose(transforms)
    
    def get_test_transforms(self) -> A.Compose:
        """Get test transforms (same as validation)"""
        return self.get_val_transforms()

class MetadataTransforms:
    """Metadata preprocessing and normalization"""
    
    def __init__(self):
        self.numerical_stats = {}
        self.categorical_mappings = {}
        self.fitted = False
    
    def fit(self, df, numerical_features, categorical_features):
        """Fit transformations on training data"""
        # Calculate statistics for numerical features
        for feature in numerical_features:
            if feature in df.columns:
                self.numerical_stats[feature] = {
                    'mean': df[feature].mean(),
                    'std': df[feature].std(),
                    'min': df[feature].min(),
                    'max': df[feature].max()
                }
        
        # Create mappings for categorical features
        for feature in categorical_features:
            if feature in df.columns:
                unique_values = df[feature].unique()
                self.categorical_mappings[feature] = {
                    val: idx for idx, val in enumerate(unique_values)
                }
                # Add unknown category
                self.categorical_mappings[feature]['<UNK>'] = len(unique_values)
        
        self.fitted = True
        return self
    
    def transform_numerical(self, df, features):
        """Transform numerical features (standardization)"""
        if not self.fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        transformed = {}
        for feature in features:
            if feature in df.columns and feature in self.numerical_stats:
                mean = self.numerical_stats[feature]['mean']
                std = self.numerical_stats[feature]['std']
                # Avoid division by zero
                std = max(std, 1e-8)
                transformed[feature] = (df[feature] - mean) / std
            else:
                transformed[feature] = np.zeros(len(df))
        
        return transformed
    
    def transform_categorical(self, df, features):
        """Transform categorical features (label encoding)"""
        if not self.fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        transformed = {}
        for feature in features:
            if feature in df.columns and feature in self.categorical_mappings:
                mapping = self.categorical_mappings[feature]
                unk_idx = mapping.get('<UNK>', 0)
                transformed[feature] = df[feature].map(mapping).fillna(unk_idx)
            else:
                transformed[feature] = np.zeros(len(df), dtype=int)
        
        return transformed
    
    def get_categorical_dims(self):
        """Get dimensions for categorical embeddings"""
        dims = {}
        for feature, mapping in self.categorical_mappings.items():
            dims[feature] = len(mapping)
        return dims

class ConsistencyAugmentation:
    """Augmentation strategy for consistency training"""
    
    def __init__(self, base_transforms, consistency_p=0.5):
        self.base_transforms = base_transforms
        self.consistency_p = consistency_p
        
        # Mild augmentations for consistency pairs
        self.mild_transforms = A.Compose([
            A.Resize(height=256, width=256),
            A.HorizontalFlip(p=0.3),
            A.Rotate(limit=5, p=0.2),
            A.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=2, p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    
    def __call__(self, image):
        """Apply consistency augmentation"""
        # Base augmented version
        augmented = self.base_transforms(image=image)['image']
        
        # Mildly augmented version for consistency
        if np.random.random() < self.consistency_p:
            consistent = self.mild_transforms(image=image)['image']
            return augmented, consistent
        
        return augmented, None