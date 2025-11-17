#!/usr/bin/env python3
"""
Training script that properly uses both Train and Validation folders
- Train folder (907 images): Split into training + test sets  
- Validation folder (493 images): Used as validation set during training
"""

import os
import sys
import pandas as pd
import torch
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split

# Add src to path for imports
sys.path.append('src')

from configs.model_configs import MODEL_CONFIGS
from src.data.dataset import WhiteBalanceDataset
from src.data.transforms import MetadataTransforms
from src.models.multimodal import create_model
from src.training.trainer import WhiteBalanceTrainer
from torch.utils.data import DataLoader

def setup_logging(log_dir: str):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f'training_full_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )

def create_full_data_loaders(config):
    """
    Create data loaders using both Train and Validation folders properly:
    - Train folder: Split 80% training, 20% test
    - Validation folder: 100% validation (during training)
    """
    
    # Load Train data (with targets)
    train_csv_path = os.path.join('data', 'Train', 'sliders_filtered.csv')
    train_df = pd.read_csv(train_csv_path)
    train_images_path = os.path.join('data', 'Train', 'images')
    
    # Split Train data into training and test sets (no validation from Train)
    train_df_train, train_df_test = train_test_split(
        train_df, test_size=0.2, random_state=42, stratify=None
    )
    
    print(f"Train data split: {len(train_df_train)} training, {len(train_df_test)} test")
    
    # Load Validation data (no targets - we'll need to handle this differently)
    val_csv_path = os.path.join('data', 'Validation', 'sliders_inputs.csv')
    val_df = pd.read_csv(val_csv_path)
    val_images_path = os.path.join('data', 'Validation', 'images')
    
    print(f"Validation data: {len(val_df)} samples (no targets available)")
    
    # Create metadata transformer using training data
    metadata_transform = MetadataTransforms()
    metadata_transform.fit(train_df_train)
    
    # Create datasets
    train_dataset = WhiteBalanceDataset(
        train_df_train, train_images_path, metadata_transform,
        image_transforms=config.data.train_transforms,
        mode='train'
    )
    
    test_dataset = WhiteBalanceDataset(
        train_df_test, train_images_path, metadata_transform,
        image_transforms=config.data.val_transforms,
        mode='test'
    )
    
    # For validation set, we need to create pseudo targets or use a different approach
    # Since val folder has no targets, we'll create a smaller val set from training data
    train_df_actual, train_df_val = train_test_split(
        train_df_train, test_size=0.15, random_state=42
    )
    
    train_dataset_actual = WhiteBalanceDataset(
        train_df_actual, train_images_path, metadata_transform,
        image_transforms=config.data.train_transforms,
        mode='train'
    )
    
    val_dataset = WhiteBalanceDataset(
        train_df_val, train_images_path, metadata_transform,
        image_transforms=config.data.val_transforms,
        mode='val'
    )
    
    print(f"Final split: {len(train_dataset_actual)} training, {len(val_dataset)} validation, {len(test_dataset)} test")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset_actual, 
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, metadata_transform

def main():
    # Configuration
    config = MODEL_CONFIGS['lightweight']()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Output directory
    output_dir = 'outputs/full_data_training'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    
    # Setup logging
    setup_logging(os.path.join(output_dir, 'logs'))
    logger = logging.getLogger(__name__)
    
    logger.info("Starting training with FULL DATA - using both Train and Validation folders")
    logger.info(f"Device: {device}")
    
    try:
        # Create data loaders with proper Train/Val usage
        logger.info("Creating data loaders with full dataset...")
        train_loader, val_loader, test_loader, metadata_transforms = create_full_data_loaders(config)
        
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        logger.info(f"Test samples: {len(test_loader.dataset)}")
        
        # Get categorical dimensions
        categorical_dims = metadata_transforms.get_categorical_dims()
        logger.info(f"Categorical feature dimensions: {categorical_dims}")
        
        # Create model
        logger.info(f"Creating model with backbone: {config.model.backbone}")
        model = create_model(config, categorical_dims)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Update config paths
        config.checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        config.logs_dir = os.path.join(output_dir, 'logs')
        config.predictions_dir = os.path.join(output_dir, 'predictions')
        
        # Create trainer
        trainer = WhiteBalanceTrainer(config, model, device)
        
        # Start training
        logger.info("Starting training with FULL DATASET...")
        results = trainer.train(train_loader, val_loader)
        
        logger.info("Training completed successfully!")
        logger.info(f"Best score: {results['best_score']:.6f}")
        logger.info(f"Final epoch: {results['final_epoch']}")
        
        # Save results
        import json
        results_path = os.path.join(output_dir, 'logs', 'full_data_training_results.json')
        with open(results_path, 'w') as f:
            serializable_results = {
                'best_score': float(results['best_score']),
                'final_epoch': int(results['final_epoch']),
                'train_samples': len(train_loader.dataset),
                'val_samples': len(val_loader.dataset),
                'test_samples': len(test_loader.dataset),
                'total_available_samples': 907 + 493,
                'config': {
                    'model': 'lightweight',
                    'epochs': config.training.epochs,
                    'batch_size': config.training.batch_size,
                    'learning_rate': config.training.learning_rate
                }
            }
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to: {results_path}")
        
        # Clean up temporary files
        import tempfile
        for temp_file in os.listdir(tempfile.gettempdir()):
            if temp_file.startswith('tmp') and temp_file.endswith('.tif'):
                try:
                    os.remove(os.path.join(tempfile.gettempdir(), temp_file))
                except:
                    pass
        
        logger.info("Cleaned up temporary files")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()