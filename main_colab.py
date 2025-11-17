#!/usr/bin/env python3
"""
Colab-optimized main script for Aftershoot White Balance Prediction
Simplified version for Google Colab environment with better progress tracking
"""

import os
import sys
import torch
import json
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append('/content/aftershoot_wb_prediction/src')

def setup_colab_logging():
    """Setup logging for Colab environment"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('colab_training.log')
        ]
    )
    return logging.getLogger(__name__)

def check_colab_environment():
    """Check if running in Colab and setup accordingly"""
    try:
        import google.colab
        IN_COLAB = True
        print("🚀 Running in Google Colab")
    except ImportError:
        IN_COLAB = False
        print("💻 Running locally")
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🔥 GPU detected: {gpu_name}")
        device = torch.device('cuda')
    else:
        print("⚠️ No GPU detected - using CPU")
        device = torch.device('cpu')
    
    return IN_COLAB, device

def load_config(config_name):
    """Load configuration file"""
    config_path = f"configs/{config_name}.json"
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        print("Available configs:")
        for f in os.listdir("configs"):
            if f.endswith('.json'):
                print(f"  - {f.replace('.json', '')}")
        return None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"✅ Loaded config: {config_name}")
    return config

def run_eda(config, logger):
    """Run Exploratory Data Analysis"""
    logger.info("🔍 Starting EDA...")
    
    try:
        from data.dataset import create_data_loaders
        from utils.eda import generate_eda_report
        
        # Check data availability
        data_path = "data/Train/sliders.csv"
        if not os.path.exists(data_path):
            logger.error(f"❌ Training data not found: {data_path}")
            return False
        
        # Load dataset info
        df = pd.read_csv(data_path)
        logger.info(f"📊 Dataset loaded: {len(df)} samples")
        
        # Basic statistics
        print("\n" + "="*50)
        print("📊 DATASET OVERVIEW")
        print("="*50)
        print(f"Total samples: {len(df)}")
        print(f"Features: {list(df.columns)}")
        print(f"Temperature range: {df['Temperature'].min():.0f}K - {df['Temperature'].max():.0f}K")
        print(f"Tint range: {df['Tint'].min():.1f} - {df['Tint'].max():.1f}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        
        # Temperature sensitivity analysis
        print(f"\n🌡️ TEMPERATURE SENSITIVITY")
        df['temp_change'] = abs(df['Temperature'] - df['currTemp'])
        
        temp_ranges = [
            (df['currTemp'] < 3500, "Low temp (< 3500K)"),
            ((df['currTemp'] >= 3500) & (df['currTemp'] < 6000), "Mid temp (3500-6000K)"),
            (df['currTemp'] >= 6000, "High temp (> 6000K)")
        ]
        
        for mask, label in temp_ranges:
            if mask.sum() > 0:
                avg_change = df[mask]['temp_change'].mean()
                print(f"{label}: Avg change = {avg_change:.0f}K")
        
        # Generate full EDA report
        logger.info("📈 Generating EDA visualizations...")
        try:
            generate_eda_report(config)
            logger.info("✅ EDA visualizations saved to outputs/eda/")
        except Exception as e:
            logger.warning(f"⚠️ EDA visualization generation failed: {e}")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Missing required modules for EDA: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ EDA failed: {e}")
        return False

def run_training(config, logger, args):
    """Run model training"""
    logger.info("🚀 Starting training...")
    
    try:
        from data.dataset import create_data_loaders
        from models.multimodal import MultiModalWhiteBalanceModel
        from training.trainer import WhiteBalanceTrainer
        
        # Create data loaders
        logger.info("📁 Creating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(config)
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        
        # Create model
        logger.info("🏗️ Creating model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = MultiModalWhiteBalanceModel(
            backbone=config['model']['backbone'],
            pretrained=config['model']['pretrained'],
            num_numerical_features=len(train_loader.dataset.numerical_features),
            categorical_feature_dims=train_loader.dataset.categorical_feature_dims,
            dropout_rate=config['model']['dropout_rate']
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"📊 Model parameters: {total_params:,}")
        
        # Create trainer
        trainer = WhiteBalanceTrainer(
            model=model,
            config=config,
            device=device
        )
        
        # Training parameters
        epochs = args.epochs if args.epochs else config['training']['epochs']
        
        logger.info(f"🔄 Starting training for {epochs} epochs...")
        
        # Resume from checkpoint if specified
        if args.resume:
            if os.path.exists(args.resume):
                logger.info(f"📂 Resuming from checkpoint: {args.resume}")
                trainer.load_checkpoint(args.resume)
            else:
                logger.warning(f"⚠️ Checkpoint not found: {args.resume}")
        
        # Train model
        results = trainer.train(train_loader, val_loader, epochs=epochs)
        
        # Final evaluation
        logger.info("📊 Running final evaluation...")
        test_metrics = trainer.evaluate(test_loader)
        
        print("\n" + "="*50)
        print("🎉 TRAINING COMPLETED")
        print("="*50)
        print(f"Best validation loss: {results['best_val_loss']:.4f}")
        print(f"Best epoch: {results['best_epoch']}")
        print(f"Test Temperature MAE: {test_metrics['temperature_mae']:.2f}K")
        print(f"Test Tint MAE: {test_metrics['tint_mae']:.2f}")
        print(f"Final test loss: {test_metrics['total_loss']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function for Colab execution"""
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Aftershoot WB Prediction - Colab Version')
    parser.add_argument('--eda', action='store_true', help='Run exploratory data analysis')
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--config', type=str, default='efficientnet', help='Configuration name')
    parser.add_argument('--epochs', type=int, help='Number of epochs (override config)')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint path')
    
    args = parser.parse_args()
    
    # Setup environment
    print("🔧 Setting up environment...")
    IN_COLAB, device = check_colab_environment()
    logger = setup_colab_logging()
    
    # Create output directories
    output_dirs = ['outputs/checkpoints', 'outputs/logs', 'outputs/eda', 'outputs/predictions']
    for dir_path in output_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    if config is None:
        return
    
    logger.info(f"🎯 Starting Aftershoot WB Prediction")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Device: {device}")
    
    success = True
    
    # Run EDA if requested
    if args.eda:
        success &= run_eda(config, logger)
    
    # Run training if requested
    if args.train:
        success &= run_training(config, logger, args)
    
    # If neither specified, run EDA by default
    if not args.eda and not args.train:
        logger.info("No specific task specified, running EDA...")
        success &= run_eda(config, logger)
    
    if success:
        print("\n🎉 All tasks completed successfully!")
        if IN_COLAB:
            print("💾 Don't forget to save your results to Google Drive!")
    else:
        print("\n❌ Some tasks failed. Check the logs for details.")

if __name__ == "__main__":
    main()