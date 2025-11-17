#!/usr/bin/env python3
"""
Improved training script that uses more of the available training data
Instead of 907 -> 543/182/182 split, use 907 -> 725/182 split for better training
"""

import os
import sys
import pandas as pd
import torch
import logging
from datetime import datetime

# Add src to path for imports
sys.path.append('src')

from configs.model_configs import MODEL_CONFIGS
from src.data.dataset import create_data_loaders
from src.models.multimodal import create_model
from src.training.trainer import WhiteBalanceTrainer

def setup_logging(log_dir: str):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f'training_improved_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """
    Improved training with better data split:
    - Use 80% of 907 images for training (725 images)
    - Use 20% of 907 images for validation (182 images) 
    - Save validation folder (493 images) for final inference/testing
    """
    
    # Configuration
    config = MODEL_CONFIGS['lightweight']()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Output directory
    output_dir = 'outputs/improved_training'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
    
    # Setup logging
    setup_logging(os.path.join(output_dir, 'logs'))
    logger = logging.getLogger(__name__)
    
    logger.info("Starting IMPROVED training - using more training data")
    logger.info("Data strategy:")
    logger.info("- Train folder (907 images): 80% training (725) + 20% validation (182)")
    logger.info("- Validation folder (493 images): Reserved for final inference")
    logger.info(f"Device: {device}")
    
    # Update paths in config for better split
    config.data.train_images_path = os.path.join('data', 'Train', 'images')
    config.data.train_csv_path = os.path.join('data', 'Train', 'sliders_filtered.csv')
    config.data.val_images_path = os.path.join('data', 'Train', 'images')  # Same as train
    config.data.val_csv_path = os.path.join('data', 'Train', 'sliders_filtered.csv')  # Same as train
    
    # Update output paths
    config.output_dir = output_dir
    config.checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    config.logs_dir = os.path.join(output_dir, 'logs')
    config.predictions_dir = os.path.join(output_dir, 'predictions')
    
    # Modify training split to use more data for training
    config.data.train_split = 0.8  # 80% for training (725 images)
    config.data.val_split = 0.2   # 20% for validation (182 images)
    config.data.test_split = 0.0  # No test split from train data
    
    try:
        # Create data loaders with improved split
        logger.info("Creating data loaders with improved split...")
        train_loader, val_loader, test_loader, metadata_transforms = create_data_loaders(config)
        
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        if test_loader:
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
        
        # Create trainer
        trainer = WhiteBalanceTrainer(config, model, device)
        
        # Start training with more data
        logger.info("Starting training with IMPROVED DATA SPLIT...")
        results = trainer.train(train_loader, val_loader)
        
        logger.info("Training completed successfully!")
        logger.info(f"Best score: {results['best_score']:.6f}")
        logger.info(f"Final epoch: {results['final_epoch']}")
        
        # Save results with data usage info
        import json
        results_path = os.path.join(output_dir, 'logs', 'improved_training_results.json')
        with open(results_path, 'w') as f:
            serializable_results = {
                'best_score': float(results['best_score']),
                'final_epoch': int(results['final_epoch']),
                'data_usage': {
                    'train_images_used': len(train_loader.dataset),
                    'val_images_used': len(val_loader.dataset),
                    'total_train_folder_images': 907,
                    'unused_validation_folder_images': 493,
                    'training_data_percentage': len(train_loader.dataset) / 907 * 100
                },
                'config': {
                    'model': 'lightweight',
                    'epochs': config.training.epochs,
                    'batch_size': config.training.batch_size,
                    'learning_rate': config.training.learning_rate,
                    'train_split': config.data.train_split,
                    'val_split': config.data.val_split
                }
            }
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to: {results_path}")
        
        # Report data utilization
        logger.info("=" * 50)
        logger.info("DATA UTILIZATION SUMMARY:")
        logger.info(f"Training images: {len(train_loader.dataset)}/907 ({len(train_loader.dataset)/907*100:.1f}%)")
        logger.info(f"Validation images: {len(val_loader.dataset)}/907 ({len(val_loader.dataset)/907*100:.1f}%)")
        logger.info(f"Unused Validation folder: 493 images (available for inference)")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()