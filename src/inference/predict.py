import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple
from tqdm import tqdm
import argparse

from ..models.multimodal import create_model
from ..data.dataset import create_data_loaders
from ..training.trainer import load_checkpoint
from configs.model_configs import MODEL_CONFIGS

class WhiteBalancePredictor:
    """Inference pipeline for White Balance prediction"""
    
    def __init__(self, model_path: str, config_name: str = 'efficientnet', 
                 device: str = 'cuda'):
        self.device = device
        self.model_path = model_path
        self.config_name = config_name
        
        # Load config and model
        self.config = MODEL_CONFIGS[config_name]()
        self.model = None
        self.metadata_transforms = None
        
        # Set device
        if device == 'cuda' and not torch.cuda.is_available():
            self.device = 'cpu'
            print("CUDA not available, using CPU")
    
    def load_model(self, categorical_dims: Dict[str, int] = None):
        """Load trained model from checkpoint"""
        
        # Create model
        self.model = create_model(self.config, categorical_dims)
        self.model.to(self.device)
        
        # Load checkpoint
        if os.path.exists(self.model_path):
            checkpoint = load_checkpoint(self.model_path, self.model)
            print(f"Model loaded from {self.model_path}")
            print(f"Best score: {checkpoint.get('best_score', 'N/A')}")
            print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
        else:
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        
        self.model.eval()
    
    def predict_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """Predict on a batch of data"""
        
        with torch.no_grad():
            # Move batch to device
            device_batch = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    device_batch[key] = value.to(self.device)
                elif isinstance(value, dict):
                    device_batch[key] = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in value.items()
                    }
                else:
                    device_batch[key] = value
            
            # Get predictions
            outputs = self.model(device_batch)
            
            # Convert to numpy
            predictions = {
                'temperature': outputs['temperature'].cpu().numpy(),
                'tint': outputs['tint'].cpu().numpy(),
                'id_global': batch['id_global']
            }
        
        return predictions
    
    def predict_dataloader(self, dataloader) -> pd.DataFrame:
        """Predict on entire dataloader"""
        
        all_predictions = []
        all_ids = []
        
        print("Generating predictions...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Predicting'):
                predictions = self.predict_batch(batch)
                
                all_predictions.append({
                    'temperature': predictions['temperature'],
                    'tint': predictions['tint']
                })
                all_ids.extend(predictions['id_global'])
        
        # Combine all predictions
        all_temp_pred = np.concatenate([p['temperature'] for p in all_predictions])
        all_tint_pred = np.concatenate([p['tint'] for p in all_predictions])
        
        # Create DataFrame
        results_df = pd.DataFrame({
            'id_global': all_ids,
            'Temperature': all_temp_pred,
            'Tint': all_tint_pred
        })
        
        return results_df
    
    def postprocess_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-process predictions (round to integers, clamp to valid ranges)"""
        
        processed_df = df.copy()
        
        # Round to nearest integer as required
        processed_df['Temperature'] = processed_df['Temperature'].round().astype(int)
        processed_df['Tint'] = processed_df['Tint'].round().astype(int)
        
        # Clamp to valid ranges
        processed_df['Temperature'] = processed_df['Temperature'].clip(2000, 50000)
        processed_df['Tint'] = processed_df['Tint'].clip(-150, 150)
        
        print(f"Temperature range: {processed_df['Temperature'].min()} - {processed_df['Temperature'].max()}")
        print(f"Tint range: {processed_df['Tint'].min()} - {processed_df['Tint'].max()}")
        
        return processed_df
    
    def create_submission(self, predictions_df: pd.DataFrame, 
                         output_path: str) -> str:
        """Create submission file in the required format"""
        
        # Ensure correct order and format
        submission_df = predictions_df[['id_global', 'Temperature', 'Tint']].copy()
        submission_df = submission_df.sort_values('id_global').reset_index(drop=True)
        
        # Set index as required
        submission_df.set_index('id_global', inplace=True)
        
        # Save to CSV
        submission_df.to_csv(output_path)
        
        print(f"Submission saved to: {output_path}")
        print(f"Submission shape: {submission_df.shape}")
        print(f"Expected shape: (493, 2)")
        
        # Validate submission format
        if submission_df.shape[0] != 493:
            print(f"WARNING: Expected 493 predictions, got {submission_df.shape[0]}")
        
        if submission_df.shape[1] != 2:
            print(f"WARNING: Expected 2 columns, got {submission_df.shape[1]}")
        
        return output_path

class EnsemblePredictor:
    """Ensemble predictor combining multiple models"""
    
    def __init__(self, model_configs: List[Tuple[str, str]], device: str = 'cuda'):
        """
        Args:
            model_configs: List of (model_path, config_name) tuples
            device: Device to run inference on
        """
        self.device = device
        self.predictors = []
        
        # Initialize individual predictors
        for model_path, config_name in model_configs:
            predictor = WhiteBalancePredictor(model_path, config_name, device)
            self.predictors.append(predictor)
    
    def load_models(self, categorical_dims: Dict[str, int] = None):
        """Load all ensemble models"""
        for i, predictor in enumerate(self.predictors):
            print(f"Loading model {i+1}/{len(self.predictors)}")
            predictor.load_model(categorical_dims)
    
    def predict_dataloader(self, dataloader, 
                          ensemble_method: str = 'mean') -> pd.DataFrame:
        """Predict using ensemble of models"""
        
        all_predictions = []
        
        # Get predictions from each model
        for i, predictor in enumerate(self.predictors):
            print(f"Generating predictions from model {i+1}/{len(self.predictors)}")
            pred_df = predictor.predict_dataloader(dataloader)
            all_predictions.append(pred_df)
        
        # Combine predictions
        if ensemble_method == 'mean':
            # Simple average
            final_df = all_predictions[0].copy()
            
            for i in range(1, len(all_predictions)):
                final_df['Temperature'] += all_predictions[i]['Temperature']
                final_df['Tint'] += all_predictions[i]['Tint']
            
            final_df['Temperature'] /= len(all_predictions)
            final_df['Tint'] /= len(all_predictions)
            
        elif ensemble_method == 'weighted_mean':
            # Weighted average (weights based on validation performance)
            weights = [0.4, 0.35, 0.25]  # Example weights
            
            if len(weights) != len(all_predictions):
                weights = [1.0 / len(all_predictions)] * len(all_predictions)
            
            final_df = all_predictions[0].copy()
            final_df['Temperature'] *= weights[0]
            final_df['Tint'] *= weights[0]
            
            for i in range(1, len(all_predictions)):
                final_df['Temperature'] += weights[i] * all_predictions[i]['Temperature']
                final_df['Tint'] += weights[i] * all_predictions[i]['Tint']
        
        elif ensemble_method == 'median':
            # Median ensemble
            temp_stack = np.stack([df['Temperature'].values for df in all_predictions])
            tint_stack = np.stack([df['Tint'].values for df in all_predictions])
            
            final_df = all_predictions[0].copy()
            final_df['Temperature'] = np.median(temp_stack, axis=0)
            final_df['Tint'] = np.median(tint_stack, axis=0)
        
        else:
            raise ValueError(f"Unknown ensemble method: {ensemble_method}")
        
        return final_df

def run_inference(config_name: str = 'efficientnet',
                 model_path: str = None,
                 data_dir: str = 'data',
                 output_dir: str = 'outputs/predictions',
                 batch_size: int = 32,
                 device: str = 'cuda'):
    """Run complete inference pipeline"""
    
    # Set default model path
    if model_path is None:
        model_path = f'outputs/checkpoints/best_model.pt'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load config
    config = MODEL_CONFIGS[config_name]()
    config.training.batch_size = batch_size
    
    # Update data paths
    config.data.val_images_path = os.path.join(data_dir, 'Validation/images')
    config.data.val_csv_path = os.path.join(data_dir, 'Validation/sliders_inputs.csv')
    config.data.train_images_path = os.path.join(data_dir, 'Train/images')
    config.data.train_csv_path = os.path.join(data_dir, 'Train/sliders.csv')
    
    # Create data loaders (we need train data for metadata transformer fitting)
    print("Loading data...")
    _, _, test_loader, metadata_transforms = create_data_loaders(config)
    
    # Get categorical dimensions
    categorical_dims = metadata_transforms.get_categorical_dims()
    
    # Create predictor
    predictor = WhiteBalancePredictor(model_path, config_name, device)
    predictor.metadata_transforms = metadata_transforms
    
    # Load model
    print("Loading model...")
    predictor.load_model(categorical_dims)
    
    # Generate predictions
    print("Generating predictions...")
    predictions_df = predictor.predict_dataloader(test_loader)
    
    # Post-process predictions
    print("Post-processing predictions...")
    final_predictions = predictor.postprocess_predictions(predictions_df)
    
    # Create submission file
    submission_path = os.path.join(output_dir, 'submission.csv')
    predictor.create_submission(final_predictions, submission_path)
    
    # Save raw predictions for analysis
    raw_path = os.path.join(output_dir, 'raw_predictions.csv')
    predictions_df.to_csv(raw_path, index=False)
    
    print("Inference completed successfully!")
    
    return submission_path, raw_path

def main():
    """Command line interface for inference"""
    parser = argparse.ArgumentParser(description='White Balance Prediction Inference')
    parser.add_argument('--config', type=str, default='efficientnet',
                       choices=['efficientnet', 'resnet', 'convnext', 'lightweight'],
                       help='Model configuration to use')
    parser.add_argument('--model_path', type=str, 
                       default='outputs/checkpoints/best_model.pt',
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing the dataset')
    parser.add_argument('--output_dir', type=str, default='outputs/predictions',
                       help='Directory to save predictions')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run inference on')
    parser.add_argument('--ensemble', action='store_true',
                       help='Use ensemble of multiple models')
    
    args = parser.parse_args()
    
    if args.ensemble:
        # Example ensemble configuration
        model_configs = [
            ('outputs/checkpoints/best_model_efficientnet.pt', 'efficientnet'),
            ('outputs/checkpoints/best_model_resnet.pt', 'resnet'),
            ('outputs/checkpoints/best_model_convnext.pt', 'convnext')
        ]
        
        # Run ensemble inference
        print("Running ensemble inference...")
        # Implementation would be similar to single model inference
        print("Ensemble inference not fully implemented in this version")
    else:
        # Run single model inference
        submission_path, raw_path = run_inference(
            config_name=args.config,
            model_path=args.model_path,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            device=args.device
        )
        
        print(f"Submission file: {submission_path}")
        print(f"Raw predictions: {raw_path}")

if __name__ == '__main__':
    main()