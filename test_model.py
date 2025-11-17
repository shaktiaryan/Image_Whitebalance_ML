#!/usr/bin/env python3
"""
Model Testing Script for Aftershoot White Balance Prediction
Tests the trained model on validation set and generates predictions
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

from configs.model_configs import MODEL_CONFIGS
from src.data.dataset import WhiteBalanceDataset
from src.data.transforms import ImageTransforms, MetadataTransforms
from src.models.multimodal import create_model
from torch.utils.data import DataLoader

def load_trained_model(model_path, config, categorical_dims, device):
    """Load the trained model from checkpoint"""
    model = create_model(config, categorical_dims)
    
    if os.path.exists(model_path):
        print(f"📁 Loading model from: {model_path}")
        # Use weights_only=False for compatibility with older PyTorch checkpoints
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully!")
        return model
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")

def calculate_metrics(predictions, targets):
    """Calculate evaluation metrics"""
    temp_pred, tint_pred = predictions[:, 0], predictions[:, 1]
    temp_true, tint_true = targets[:, 0], targets[:, 1]
    
    # Mean Absolute Error
    temp_mae = np.mean(np.abs(temp_pred - temp_true))
    tint_mae = np.mean(np.abs(tint_pred - tint_true))
    overall_mae = (temp_mae + tint_mae) / 2
    
    # Root Mean Square Error
    temp_rmse = np.sqrt(np.mean((temp_pred - temp_true) ** 2))
    tint_rmse = np.sqrt(np.mean((tint_pred - tint_true) ** 2))
    overall_rmse = np.sqrt(temp_rmse ** 2 + tint_rmse ** 2)
    
    # Mean Square Error (what the training uses)
    temp_mse = np.mean((temp_pred - temp_true) ** 2)
    tint_mse = np.mean((tint_pred - tint_true) ** 2)
    overall_mse = (temp_mse + tint_mse) / 2
    
    return {
        'temperature_mae': temp_mae,
        'tint_mae': tint_mae,
        'overall_mae': overall_mae,
        'temperature_rmse': temp_rmse,
        'tint_rmse': tint_rmse,
        'overall_rmse': overall_rmse,
        'temperature_mse': temp_mse,
        'tint_mse': tint_mse,
        'overall_mse': overall_mse
    }

def test_model_on_dataset(model, dataloader, device, dataset_name):
    """Test model on a dataset and return predictions and metrics"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    print(f"🧪 Testing model on {dataset_name} dataset...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device - handle all tensor types properly
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
                elif isinstance(batch[key], dict):
                    # Handle nested dictionaries (like metadata)
                    for sub_key in batch[key]:
                        if isinstance(batch[key][sub_key], torch.Tensor):
                            batch[key][sub_key] = batch[key][sub_key].to(device)
            
            # Get predictions
            outputs = model(batch)
            
            # Handle model outputs - extract temperature and tint predictions
            if isinstance(outputs, dict) and 'temperature' in outputs and 'tint' in outputs:
                temp_pred = outputs['temperature'].cpu().numpy()
                tint_pred = outputs['tint'].cpu().numpy()
                predictions = np.column_stack([temp_pred, tint_pred])
            elif isinstance(outputs, dict):
                # If output is a dict with other structure
                if 'output' in outputs:
                    predictions = outputs['output'].cpu().numpy()
                else:
                    # Get first tensor value from dict
                    predictions = list(outputs.values())[0].cpu().numpy()
                    if predictions.ndim == 1:
                        # If 1D, assume it needs reshaping
                        predictions = predictions.reshape(-1, 1)
            else:
                # If output is a tensor
                predictions = outputs.cpu().numpy()
                if predictions.ndim == 1:
                    predictions = predictions.reshape(-1, 1)
            
            # Get targets - combine temperature and tint
            temperature = batch['temperature'].cpu().numpy()
            tint = batch['tint'].cpu().numpy()
            targets = np.column_stack([temperature, tint])
            
            all_predictions.append(predictions)
            all_targets.append(targets)
            
            if batch_idx % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")
    
    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate metrics
    metrics = calculate_metrics(all_predictions, all_targets)
    
    return all_predictions, all_targets, metrics

def create_visualizations(predictions, targets, metrics, output_dir, dataset_name):
    """Create visualization plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract temperature and tint
    temp_pred, tint_pred = predictions[:, 0], predictions[:, 1]
    temp_true, tint_true = targets[:, 0], targets[:, 1]
    
    # 1. Prediction vs True scatter plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Temperature scatter plot
    ax1.scatter(temp_true, temp_pred, alpha=0.6, s=20)
    ax1.plot([temp_true.min(), temp_true.max()], [temp_true.min(), temp_true.max()], 'r--', lw=2)
    ax1.set_xlabel('True Temperature')
    ax1.set_ylabel('Predicted Temperature')
    ax1.set_title(f'Temperature Predictions - {dataset_name}\nMAE: {metrics["temperature_mae"]:.3f}, RMSE: {metrics["temperature_rmse"]:.3f}')
    ax1.grid(True, alpha=0.3)
    
    # Tint scatter plot
    ax2.scatter(tint_true, tint_pred, alpha=0.6, s=20)
    ax2.plot([tint_true.min(), tint_true.max()], [tint_true.min(), tint_true.max()], 'r--', lw=2)
    ax2.set_xlabel('True Tint')
    ax2.set_ylabel('Predicted Tint')
    ax2.set_title(f'Tint Predictions - {dataset_name}\nMAE: {metrics["tint_mae"]:.3f}, RMSE: {metrics["tint_rmse"]:.3f}')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_predictions_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Error distribution histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    temp_errors = temp_pred - temp_true
    tint_errors = tint_pred - tint_true
    
    # Temperature error distribution
    ax1.hist(temp_errors, bins=50, alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Temperature Prediction Error')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Temperature Error Distribution - {dataset_name}\nMean: {np.mean(temp_errors):.3f}, Std: {np.std(temp_errors):.3f}')
    ax1.grid(True, alpha=0.3)
    
    # Tint error distribution
    ax2.hist(tint_errors, bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Tint Prediction Error')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Tint Error Distribution - {dataset_name}\nMean: {np.mean(tint_errors):.3f}, Std: {np.std(tint_errors):.3f}')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_error_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualizations saved to: {output_dir}")

def main():
    """Main testing function"""
    print("🔬 AFTERSHOOT WHITE BALANCE MODEL TESTING")
    print("=" * 60)
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Using device: {device}")
    
    # Load model configuration
    config = MODEL_CONFIGS['lightweight']()
    
    # Update paths
    config.data.train_csv_path = 'data/Train/sliders_filtered.csv'
    config.data.train_images_path = 'data/Train/images'
    config.data.val_csv_path = 'data/Validation/sliders_inputs.csv'
    config.data.val_images_path = 'data/Validation/images'
    
    # Find the best model checkpoint
    model_paths = [
        'outputs/complete_90_split/checkpoints/best_model.pt',
        'outputs/improved_90_split/checkpoints/best_model.pt',
        'outputs/clean_training/checkpoints/best_model.pt'
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("❌ No trained model found! Please train a model first.")
        return
    
    print(f"📁 Found model: {model_path}")
    
    # Load and prepare test data
    print("📊 Loading test data...")
    
    # Load training data to fit metadata transforms and create test sets
    train_df = pd.read_csv(config.data.train_csv_path)
    
    # Split training data using same split as training (90/10)
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(
        train_df, 
        test_size=0.1,  # 10% for validation (same as training)
        random_state=42,
        stratify=None
    )
    
    print(f"📈 Test datasets prepared:")
    print(f"   Validation set: {len(val_data)} samples (from training split)")
    
    # Initialize transforms
    img_transforms = ImageTransforms(
        image_size=config.data.image_size,
        normalize_mean=config.data.normalize_mean,
        normalize_std=config.data.normalize_std
    )
    
    # Create and fit metadata transformer on training data
    metadata_transforms = MetadataTransforms()
    metadata_transforms.fit(
        train_data, 
        config.model.numerical_features,
        config.model.categorical_features
    )
    
    # Get categorical dimensions and load model
    categorical_dims = metadata_transforms.get_categorical_dims()
    model = load_trained_model(model_path, config, categorical_dims, device)
    
    # Create validation dataset for testing
    val_dataset = WhiteBalanceDataset(
        df=val_data,
        images_dir=config.data.train_images_path,
        image_transforms=img_transforms.get_val_transforms(),
        metadata_transforms=metadata_transforms,
        numerical_features=config.model.numerical_features,
        categorical_features=config.model.categorical_features,
        is_training=False
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Test the model
    output_dir = 'outputs/model_testing'
    os.makedirs(output_dir, exist_ok=True)
    
    print("🧪 Starting model evaluation...")
    
    # Test on validation set
    val_predictions, val_targets, val_metrics = test_model_on_dataset(
        model, val_dataloader, device, "Validation"
    )
    
    # Create visualizations
    create_visualizations(val_predictions, val_targets, val_metrics, output_dir, "Validation")
    
    # Print detailed results
    print("\\n" + "="*60)
    print("📊 MODEL TESTING RESULTS")
    print("="*60)
    
    print(f"\\n🎯 VALIDATION SET RESULTS ({len(val_data)} samples):")
    print(f"   Temperature MAE: {val_metrics['temperature_mae']:.4f}")
    print(f"   Temperature RMSE: {val_metrics['temperature_rmse']:.4f}")
    print(f"   Tint MAE: {val_metrics['tint_mae']:.4f}")
    print(f"   Tint RMSE: {val_metrics['tint_rmse']:.4f}")
    print(f"   Overall MAE: {val_metrics['overall_mae']:.4f}")
    print(f"   Overall RMSE: {val_metrics['overall_rmse']:.4f}")
    print(f"   Overall MSE: {val_metrics['overall_mse']:.6f} (training metric)")
    
    # Save predictions and metrics
    results = {
        'model_path': model_path,
        'test_date': datetime.now().isoformat(),
        'device': device,
        'validation_metrics': {
            'temperature_mae': float(val_metrics['temperature_mae']),
            'tint_mae': float(val_metrics['tint_mae']),
            'overall_mae': float(val_metrics['overall_mae']),
            'temperature_rmse': float(val_metrics['temperature_rmse']),
            'tint_rmse': float(val_metrics['tint_rmse']),
            'overall_rmse': float(val_metrics['overall_rmse']),
            'temperature_mse': float(val_metrics['temperature_mse']),
            'tint_mse': float(val_metrics['tint_mse']),
            'overall_mse': float(val_metrics['overall_mse'])
        },
        'model_config': {
            'backbone': config.model.backbone,
            'data_split': '90/10',
            'training_samples': len(train_data),
            'validation_samples': len(val_data)
        }
    }
    
    results_path = os.path.join(output_dir, 'testing_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save predictions CSV
    predictions_df = pd.DataFrame({
        'temperature_true': val_targets[:, 0],
        'tint_true': val_targets[:, 1],
        'temperature_pred': val_predictions[:, 0],
        'tint_pred': val_predictions[:, 1],
        'temperature_error': val_predictions[:, 0] - val_targets[:, 0],
        'tint_error': val_predictions[:, 1] - val_targets[:, 1]
    })
    predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    
    print(f"\\n💾 Results saved to: {output_dir}")
    print(f"   📊 Metrics: testing_results.json")
    print(f"   📈 Predictions: predictions.csv") 
    print(f"   📊 Visualizations: *_scatter.png, *_error_distribution.png")
    
    print("\\n✅ Model testing completed successfully!")

if __name__ == '__main__':
    main()