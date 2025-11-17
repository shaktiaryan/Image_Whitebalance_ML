import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import os
from typing import Dict, Any, Optional, Tuple
from sklearn.model_selection import train_test_split
import cv2

from .transforms import ImageTransforms, MetadataTransforms, ConsistencyAugmentation

class WhiteBalanceDataset(Dataset):
    """Dataset for White Balance prediction"""
    
    def __init__(self, 
                 df: pd.DataFrame,
                 images_dir: str,
                 image_transforms: Optional[Any] = None,
                 metadata_transforms: Optional[MetadataTransforms] = None,
                 numerical_features: list = None,
                 categorical_features: list = None,
                 is_training: bool = True,
                 use_consistency: bool = False):
        
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.image_transforms = image_transforms
        self.metadata_transforms = metadata_transforms
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        self.is_training = is_training
        self.use_consistency = use_consistency
        
        # Check if target columns exist (training vs validation)
        self.has_targets = 'Temperature' in df.columns and 'Tint' in df.columns
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        image_id = row['id_global']
        # Try .tif first (common extension), then .tiff
        image_path = os.path.join(self.images_dir, f"{image_id}.tif")
        if not os.path.exists(image_path):
            image_path = os.path.join(self.images_dir, f"{image_id}.tiff")
        
        try:
            # Load TIFF image
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Create a dummy image if loading fails
            image = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Apply image transforms
        if self.image_transforms:
            if self.use_consistency and isinstance(self.image_transforms, ConsistencyAugmentation):
                image_tensor, consistent_tensor = self.image_transforms(image)
            else:
                transformed = self.image_transforms(image=image)
                image_tensor = transformed['image']
                consistent_tensor = None
        else:
            # Convert to tensor if no transforms
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            consistent_tensor = None
        
        # Prepare metadata
        metadata = self._prepare_metadata(row)
        
        # Prepare output dictionary
        sample = {
            'image': image_tensor,
            'metadata': metadata,
            'id_global': image_id
        }
        
        if consistent_tensor is not None:
            sample['consistent_image'] = consistent_tensor
        
        # Add targets if available (training)
        if self.has_targets:
            sample['temperature'] = torch.tensor(row['Temperature'], dtype=torch.float32)
            sample['tint'] = torch.tensor(row['Tint'], dtype=torch.float32)
        
        return sample
    
    def _prepare_metadata(self, row) -> Dict[str, torch.Tensor]:
        """Prepare metadata features"""
        metadata = {}
        
        # Numerical features
        if self.metadata_transforms and self.numerical_features:
            row_df = pd.DataFrame([row])
            numerical_data = self.metadata_transforms.transform_numerical(
                row_df, self.numerical_features
            )
            numerical_values = []
            for feat in self.numerical_features:
                if feat in numerical_data:
                    val = numerical_data[feat]
                    if hasattr(val, 'iloc'):
                        numerical_values.append(val.iloc[0])
                    else:
                        numerical_values.append(val[0] if isinstance(val, (list, np.ndarray)) else val)
                else:
                    numerical_values.append(0.0)
            
            numerical_tensor = torch.tensor(numerical_values, dtype=torch.float32)
            metadata['numerical'] = numerical_tensor
        
        # Categorical features
        if self.metadata_transforms and self.categorical_features:
            row_df = pd.DataFrame([row])
            categorical_data = self.metadata_transforms.transform_categorical(
                row_df, self.categorical_features
            )
            for feat in self.categorical_features:
                val = categorical_data[feat]
                if hasattr(val, 'iloc'):
                    metadata[f'cat_{feat}'] = torch.tensor(
                        val.iloc[0], dtype=torch.long
                    )
                else:
                    metadata[f'cat_{feat}'] = torch.tensor(
                        val[0] if isinstance(val, (list, np.ndarray)) else val, dtype=torch.long
                    )
        
        return metadata

def create_data_loaders(config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders"""
    
    # Load training data
    train_df = pd.read_csv(config.data.train_csv_path)
    
    # Split training data into train/val
    train_data, val_data = train_test_split(
        train_df, 
        test_size=1-config.data.train_val_split,
        random_state=config.data.random_seed,
        stratify=None  # Could stratify by camera_model if needed
    )
    
    # Load test data (validation set from competition)
    test_df = pd.read_csv(config.data.val_csv_path)
    
    # Initialize transforms
    img_transforms = ImageTransforms(
        image_size=config.data.image_size,
        normalize_mean=config.data.normalize_mean,
        normalize_std=config.data.normalize_std
    )
    
    # Create metadata transformer and fit on training data
    metadata_transforms = MetadataTransforms()
    metadata_transforms.fit(
        train_data, 
        config.model.numerical_features,
        config.model.categorical_features
    )
    
    # Create datasets
    train_dataset = WhiteBalanceDataset(
        df=train_data,
        images_dir=config.data.train_images_path,
        image_transforms=img_transforms.get_train_transforms(config.augmentation),
        metadata_transforms=metadata_transforms,
        numerical_features=config.model.numerical_features,
        categorical_features=config.model.categorical_features,
        is_training=True,
        use_consistency=False  # Can be enabled for consistency training
    )
    
    val_dataset = WhiteBalanceDataset(
        df=val_data,
        images_dir=config.data.train_images_path,
        image_transforms=img_transforms.get_val_transforms(),
        metadata_transforms=metadata_transforms,
        numerical_features=config.model.numerical_features,
        categorical_features=config.model.categorical_features,
        is_training=False
    )
    
    test_dataset = WhiteBalanceDataset(
        df=test_df,
        images_dir=config.data.val_images_path,
        image_transforms=img_transforms.get_test_transforms(),
        metadata_transforms=metadata_transforms,
        numerical_features=config.model.numerical_features,
        categorical_features=config.model.categorical_features,
        is_training=False
    )
    
    # Create data loaders
    # Use pin_memory only if CUDA is available
    use_pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=use_pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=use_pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=use_pin_memory
    )
    
    return train_loader, val_loader, test_loader, metadata_transforms

def get_class_weights(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate class weights for imbalanced data (if needed)"""
    # For regression, we might want to weight samples based on target distribution
    # This is a placeholder for potential weighted sampling strategies
    weights = {
        'temperature': 1.0,
        'tint': 1.0
    }
    return weights

def analyze_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze dataset statistics"""
    analysis = {
        'num_samples': len(df),
        'temperature_stats': {
            'mean': df['Temperature'].mean() if 'Temperature' in df.columns else None,
            'std': df['Temperature'].std() if 'Temperature' in df.columns else None,
            'min': df['Temperature'].min() if 'Temperature' in df.columns else None,
            'max': df['Temperature'].max() if 'Temperature' in df.columns else None,
        },
        'tint_stats': {
            'mean': df['Tint'].mean() if 'Tint' in df.columns else None,
            'std': df['Tint'].std() if 'Tint' in df.columns else None,
            'min': df['Tint'].min() if 'Tint' in df.columns else None,
            'max': df['Tint'].max() if 'Tint' in df.columns else None,
        },
        'camera_models': df['camera_model'].value_counts().to_dict() if 'camera_model' in df.columns else None,
        'missing_values': df.isnull().sum().to_dict()
    }
    return analysis