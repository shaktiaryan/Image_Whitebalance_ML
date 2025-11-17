#!/usr/bin/env python3
"""
Training script for Aftershoot White Balance using preprocessed data
"""

import os
import sys
import argparse
import torch
import logging
from datetime import datetime
import pandas as pd

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
            logging.FileHandler(os.path.join(log_dir, f'training_clean_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_preprocessed_data(data_dir="outputs/preprocessed_data"):
    """Load the preprocessed and cleaned data"""
    
    print(f"📁 Loading preprocessed data from: {data_dir}")
    
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    
    print(f"✅ Data loaded successfully:")
    print(f"   • Training: {len(train_df)} samples")
    print(f"   • Validation: {len(val_df)} samples") 
    print(f"   • Test: {len(test_df)} samples")
    print(f"   • Features: {train_df.shape[1]} (including engineered features)")
    
    return train_df, val_df, test_df

def analyze_preprocessed_features(train_df):
    """Analyze the features in preprocessed data"""
    
    print(f"\n🔍 Analyzing preprocessed features:")
    
    # Identify feature types
    numeric_features = train_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove targets and IDs
    target_features = ['Temperature', 'Tint']
    id_features = ['id_global']
    
    # Clean feature lists
    numeric_features = [f for f in numeric_features if f not in target_features + id_features]
    categorical_features = [f for f in categorical_features if f not in id_features]
    
    print(f"   📊 Numeric features: {len(numeric_features)}")
    print(f"      Examples: {numeric_features[:5]}")
    print(f"   🏷️  Categorical features: {len(categorical_features)}")
    print(f"      Examples: {categorical_features}")
    print(f"   🎯 Target features: {target_features}")
    
    # Check for engineered features
    engineered_features = [f for f in train_df.columns if any(suffix in f for suffix in ['_combo', '_adjustment', '_category', '_log', '_outlier', '_hour'])]
    print(f"   ⚙️  Engineered features: {len(engineered_features)}")
    print(f"      Examples: {engineered_features[:5]}")
    
    return numeric_features, categorical_features, target_features

def main():
    parser = argparse.ArgumentParser(description='Train White Balance Model with Clean Data')
    parser.add_argument('--config', type=str, default='efficientnet',
                       choices=['efficientnet', 'resnet', 'convnext', 'lightweight'],
                       help='Model configuration to use')
    parser.add_argument('--preprocessed_data_dir', type=str, default='outputs/preprocessed_data',
                       help='Directory containing preprocessed data')
    parser.add_argument('--images_dir', type=str, default='data/Train/images',
                       help='Directory containing the images')
    parser.add_argument('--output_dir', type=str, default='outputs/clean_training',
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
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"🚀 TRAINING WITH PREPROCESSED DATA")
    print("=" * 50)
    print(f"Using device: {device}")
    
    # Load preprocessed data
    train_df, val_df, test_df = load_preprocessed_data(args.preprocessed_data_dir)
    
    # Analyze features
    numeric_features, categorical_features, target_features = analyze_preprocessed_features(train_df)
    
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
    
    # Update feature lists in config
    config.model.numerical_features = numeric_features[:10]  # Limit features for initial training
    config.model.categorical_features = [f for f in categorical_features if f in ['flashFired', 'iso_category', 'aperture_category']]
    
    # Update paths
    config.data.train_images_path = args.images_dir
    config.data.val_images_path = args.images_dir
    
    # Save preprocessed data as temporary CSV files for the data loader
    temp_dir = os.path.join(args.output_dir, 'temp_data')
    os.makedirs(temp_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(temp_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(temp_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(temp_dir, 'test.csv'), index=False)
    
    # Update config paths
    config.data.train_csv_path = os.path.join(temp_dir, 'train.csv')
    config.data.val_csv_path = os.path.join(temp_dir, 'val.csv')
    
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
    
    logger.info(f"Starting training with CLEAN data - configuration: {args.config}")
    logger.info(f"Preprocessed data directory: {args.preprocessed_data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Using {len(config.model.numerical_features)} numerical features")
    logger.info(f"Using {len(config.model.categorical_features)} categorical features")
    
    try:
        # Create data loaders (this will use our custom train/val/test split)
        logger.info("Creating data loaders...")
        
        # We need to modify this to handle our custom splits
        from src.data.dataset import WhiteBalanceDataset, MetadataTransforms, ImageTransforms
        from torch.utils.data import DataLoader
        
        # Initialize transforms
        img_transforms = ImageTransforms(
            image_size=config.data.image_size,
            normalize_mean=config.data.normalize_mean,
            normalize_std=config.data.normalize_std
        )
        
        # Create metadata transformer and fit on training data
        metadata_transforms = MetadataTransforms()
        metadata_transforms.fit(
            train_df, 
            config.model.numerical_features,
            config.model.categorical_features
        )
        
        # Create datasets
        train_dataset = WhiteBalanceDataset(
            df=train_df,
            images_dir=config.data.train_images_path,
            image_transforms=img_transforms.get_train_transforms(config.augmentation),
            metadata_transforms=metadata_transforms,
            numerical_features=config.model.numerical_features,
            categorical_features=config.model.categorical_features,
            is_training=True
        )
        
        val_dataset = WhiteBalanceDataset(
            df=val_df,
            images_dir=config.data.val_images_path,
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
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")
        
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
        logger.info("Starting training with CLEAN DATA...")
        results = trainer.train(train_loader, val_loader)
        
        logger.info("Training completed successfully!")
        logger.info(f"Best score: {results['best_score']:.6f}")
        logger.info(f"Final epoch: {results['final_epoch']}")
        
        # Save final results
        import json
        results_path = os.path.join(config.logs_dir, 'clean_training_results.json')
        with open(results_path, 'w') as f:
            # Convert any non-serializable objects
            serializable_results = {
                'best_score': float(results['best_score']),
                'final_epoch': int(results['final_epoch']),
                'config': {
                    'model': args.config,
                    'epochs': config.training.epochs,
                    'batch_size': config.training.batch_size,
                    'learning_rate': config.training.learning_rate,
                    'numerical_features': config.model.numerical_features,
                    'categorical_features': config.model.categorical_features
                },
                'data_stats': {
                    'train_samples': len(train_dataset),
                    'val_samples': len(val_dataset),
                    'test_samples': len(test_dataset),
                    'total_features': len(config.model.numerical_features) + len(config.model.categorical_features)
                }
            }
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to: {results_path}")
        
        # Clean up temporary files
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info("Cleaned up temporary files")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()