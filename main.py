#!/usr/bin/env python3
"""
Main training script for Aftershoot White Balance Prediction
"""

import os
import sys
import argparse
import torch
import logging
from datetime import datetime

# Add src to path for imports
sys.path.append('src')

from configs.model_configs import MODEL_CONFIGS
from src.data.dataset import create_data_loaders
from src.models.multimodal import create_model
from src.training.trainer import WhiteBalanceTrainer
from src.utils.visualization import create_eda_report

def setup_logging(log_dir: str):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    parser = argparse.ArgumentParser(description='Train White Balance Prediction Model')
    parser.add_argument('--config', type=str, default='efficientnet',
                       choices=['efficientnet', 'resnet', 'convnext', 'lightweight'],
                       help='Model configuration to use')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing the dataset')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for models and logs')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use: auto, cuda, cpu')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--eda', action='store_true',
                       help='Run exploratory data analysis')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load configuration
    config = MODEL_CONFIGS[args.config]()
    
    # Override config with command line arguments
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.wandb:
        config.training.use_wandb = True
    
    # Update data paths
    config.data.train_images_path = os.path.join(args.data_dir, 'Train', 'images')
    # Use filtered CSV if it exists, otherwise use original
    filtered_csv = os.path.join(args.data_dir, 'Train', 'sliders_filtered.csv')
    if os.path.exists(filtered_csv):
        config.data.train_csv_path = filtered_csv
        print(f"Using filtered CSV with only available images: {filtered_csv}")
    else:
        config.data.train_csv_path = os.path.join(args.data_dir, 'Train', 'sliders.csv')
        print(f"Using original CSV: {config.data.train_csv_path}")
    
    config.data.val_images_path = os.path.join(args.data_dir, 'Validation', 'images')
    config.data.val_csv_path = os.path.join(args.data_dir, 'Validation', 'sliders_inputs.csv')
    
    # Update output paths
    config.output_dir = args.output_dir
    config.checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    config.logs_dir = os.path.join(args.output_dir, 'logs')
    config.predictions_dir = os.path.join(args.output_dir, 'predictions')
    
    # Create output directories
    for dir_path in [config.output_dir, config.checkpoint_dir, config.logs_dir, config.predictions_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Setup logging
    setup_logging(config.logs_dir)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting training with configuration: {args.config}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Device: {device}")
    
    # Run EDA if requested
    if args.eda:
        logger.info("Running exploratory data analysis...")
        import pandas as pd
        from src.utils.visualization import create_eda_report
        
        # Load training data for EDA
        train_df = pd.read_csv(config.data.train_csv_path)
        
        eda_output_dir = os.path.join(args.output_dir, 'eda')
        create_eda_report(train_df, config.data.train_images_path, eda_output_dir)
        
        logger.info(f"EDA report saved to: {eda_output_dir}")
    
    try:
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader, test_loader, metadata_transforms = create_data_loaders(config)
        
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        logger.info(f"Test samples: {len(test_loader.dataset)}")
        
        # Get categorical dimensions for model initialization
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
        
        # Resume from checkpoint if specified
        if args.resume and os.path.exists(args.resume):
            logger.info(f"Resuming training from: {args.resume}")
            from src.training.trainer import load_checkpoint
            checkpoint = load_checkpoint(args.resume, model, trainer.optimizer, trainer.scheduler)
            logger.info(f"Resumed from epoch {checkpoint.get('epoch', 'unknown')}")
        
        # Start training
        logger.info("Starting training...")
        results = trainer.train(train_loader, val_loader)
        
        logger.info("Training completed successfully!")
        logger.info(f"Best score: {results['best_score']:.6f}")
        logger.info(f"Final epoch: {results['final_epoch']}")
        
        # Save final results
        import json
        results_path = os.path.join(config.logs_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            # Convert any non-serializable objects
            serializable_results = {
                'best_score': float(results['best_score']),
                'final_epoch': int(results['final_epoch']),
                'config': {
                    'model': args.config,
                    'epochs': config.training.epochs,
                    'batch_size': config.training.batch_size,
                    'learning_rate': config.training.learning_rate
                }
            }
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to: {results_path}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()