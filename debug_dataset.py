#!/usr/bin/env python3
"""
Simple model testing script - debug batch structure
"""

import os
import sys
import torch
import pandas as pd

# Add src to path for imports
sys.path.append('src')

from configs.model_configs import MODEL_CONFIGS
from src.data.dataset import WhiteBalanceDataset
from src.data.transforms import ImageTransforms, MetadataTransforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def debug_dataset():
    """Debug the dataset to see what keys are available"""
    print("🔍 DEBUGGING DATASET STRUCTURE")
    print("=" * 50)
    
    # Load configuration
    config = MODEL_CONFIGS['lightweight']()
    config.data.train_csv_path = 'data/Train/sliders_filtered.csv'
    config.data.train_images_path = 'data/Train/images'
    
    # Load and split training data
    train_df = pd.read_csv(config.data.train_csv_path)
    train_data, val_data = train_test_split(
        train_df, 
        test_size=0.1,
        random_state=42
    )
    
    print(f"📊 Data loaded:")
    print(f"   Training: {len(train_data)} samples")
    print(f"   Validation: {len(val_data)} samples")
    
    # Initialize transforms
    img_transforms = ImageTransforms()
    metadata_transforms = MetadataTransforms()
    metadata_transforms.fit(
        train_data, 
        config.model.numerical_features,
        config.model.categorical_features
    )
    
    # Create validation dataset
    val_dataset = WhiteBalanceDataset(
        df=val_data,
        images_dir=config.data.train_images_path,
        image_transforms=img_transforms.get_val_transforms(),
        metadata_transforms=metadata_transforms,
        numerical_features=config.model.numerical_features,
        categorical_features=config.model.categorical_features,
        is_training=False
    )
    
    # Create dataloader
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    # Get first batch and examine structure
    print("\\n🔍 Examining batch structure:")
    for batch in val_dataloader:
        print("\\nBatch keys:", list(batch.keys()))
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape} ({value.dtype})")
            elif isinstance(value, dict):
                print(f"  {key}: dict with keys {list(value.keys())}")
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, torch.Tensor):
                        print(f"    {subkey}: {subvalue.shape} ({subvalue.dtype})")
            else:
                print(f"  {key}: {type(value)}")
        break
    
    # Check DataFrame columns
    print(f"\\n📋 DataFrame columns:")
    print(f"   {list(val_data.columns)}")
    print(f"\\n🎯 Target columns available:")
    if 'Temperature' in val_data.columns and 'Tint' in val_data.columns:
        print("   ✅ Temperature, Tint found")
    else:
        print("   ❌ Temperature, Tint not found")
        print("   Available columns:", [col for col in val_data.columns if 'temp' in col.lower() or 'tint' in col.lower()])

if __name__ == '__main__':
    debug_dataset()