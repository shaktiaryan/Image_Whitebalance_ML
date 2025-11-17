import torch
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

class WhiteBalanceMetrics:
    """Metrics for evaluating white balance prediction"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics"""
        self.temperature_predictions = []
        self.temperature_targets = []
        self.tint_predictions = []
        self.tint_targets = []
        self.losses = []
    
    def update(self, predictions: Dict[str, torch.Tensor], 
               targets: Dict[str, torch.Tensor],
               loss: float = None):
        """Update metrics with new predictions and targets"""
        
        # Convert to numpy and store
        temp_pred = predictions['temperature'].detach().cpu().numpy()
        temp_target = targets['temperature'].detach().cpu().numpy()
        tint_pred = predictions['tint'].detach().cpu().numpy()
        tint_target = targets['tint'].detach().cpu().numpy()
        
        self.temperature_predictions.extend(temp_pred.flatten())
        self.temperature_targets.extend(temp_target.flatten())
        self.tint_predictions.extend(tint_pred.flatten())
        self.tint_targets.extend(tint_target.flatten())
        
        if loss is not None:
            self.losses.append(loss)
    
    def compute_mae_score(self) -> Dict[str, float]:
        """Compute MAE score using competition formula: 1 / (1 + MAE)"""
        
        temp_mae = mean_absolute_error(self.temperature_targets, self.temperature_predictions)
        tint_mae = mean_absolute_error(self.tint_targets, self.tint_predictions)
        
        # Competition scoring formula
        temp_score = 1 / (1 + temp_mae)
        tint_score = 1 / (1 + tint_mae)
        
        # Combined score (average)
        combined_score = (temp_score + tint_score) / 2
        
        return {
            'temperature_mae': temp_mae,
            'temperature_score': temp_score,
            'tint_mae': tint_mae,
            'tint_score': tint_score,
            'combined_score': combined_score,
            'average_loss': np.mean(self.losses) if self.losses else 0.0
        }
    
    def compute_detailed_metrics(self) -> Dict[str, float]:
        """Compute additional detailed metrics"""
        
        temp_pred = np.array(self.temperature_predictions)
        temp_target = np.array(self.temperature_targets)
        tint_pred = np.array(self.tint_predictions)
        tint_target = np.array(self.tint_targets)
        
        metrics = {}
        
        # Temperature metrics
        temp_mae = mean_absolute_error(temp_target, temp_pred)
        temp_mse = np.mean((temp_pred - temp_target) ** 2)
        temp_rmse = np.sqrt(temp_mse)
        temp_mape = np.mean(np.abs((temp_target - temp_pred) / np.clip(temp_target, 1, None))) * 100
        
        # Temperature correlation
        temp_corr = np.corrcoef(temp_pred, temp_target)[0, 1]
        
        metrics.update({
            'temp_mae': temp_mae,
            'temp_mse': temp_mse,
            'temp_rmse': temp_rmse,
            'temp_mape': temp_mape,
            'temp_correlation': temp_corr
        })
        
        # Tint metrics
        tint_mae = mean_absolute_error(tint_target, tint_pred)
        tint_mse = np.mean((tint_pred - tint_target) ** 2)
        tint_rmse = np.sqrt(tint_mse)
        tint_mape = np.mean(np.abs((tint_target - tint_pred) / np.clip(np.abs(tint_target), 1, None))) * 100
        
        # Tint correlation
        tint_corr = np.corrcoef(tint_pred, tint_target)[0, 1]
        
        metrics.update({
            'tint_mae': tint_mae,
            'tint_mse': tint_mse,
            'tint_rmse': tint_rmse,
            'tint_mape': tint_mape,
            'tint_correlation': tint_corr
        })
        
        # Competition scores
        temp_score = 1 / (1 + temp_mae)
        tint_score = 1 / (1 + tint_mae)
        combined_score = (temp_score + tint_score) / 2
        
        metrics.update({
            'temp_score': temp_score,
            'tint_score': tint_score,
            'combined_score': combined_score
        })
        
        return metrics
    
    def analyze_by_temperature_range(self) -> Dict[str, Dict[str, float]]:
        """Analyze metrics by temperature ranges"""
        
        temp_pred = np.array(self.temperature_predictions)
        temp_target = np.array(self.temperature_targets)
        tint_pred = np.array(self.tint_predictions)
        tint_target = np.array(self.tint_targets)
        
        # Define temperature ranges
        ranges = {
            'low': (2000, 3500),      # Low temperature range
            'medium': (3500, 6000),   # Medium temperature range  
            'high': (6000, 50000)     # High temperature range
        }
        
        range_metrics = {}
        
        for range_name, (min_temp, max_temp) in ranges.items():
            # Find samples in this range
            mask = (temp_target >= min_temp) & (temp_target < max_temp)
            
            if np.sum(mask) > 0:
                range_temp_pred = temp_pred[mask]
                range_temp_target = temp_target[mask]
                range_tint_pred = tint_pred[mask]
                range_tint_target = tint_target[mask]
                
                temp_mae = mean_absolute_error(range_temp_target, range_temp_pred)
                tint_mae = mean_absolute_error(range_tint_target, range_tint_pred)
                
                range_metrics[range_name] = {
                    'count': np.sum(mask),
                    'temp_mae': temp_mae,
                    'tint_mae': tint_mae,
                    'temp_score': 1 / (1 + temp_mae),
                    'tint_score': 1 / (1 + tint_mae),
                    'temp_range': (min_temp, max_temp)
                }
            else:
                range_metrics[range_name] = {
                    'count': 0,
                    'temp_mae': 0,
                    'tint_mae': 0,
                    'temp_score': 0,
                    'tint_score': 0,
                    'temp_range': (min_temp, max_temp)
                }
        
        return range_metrics
    
    def plot_predictions(self, save_path: str = None, show: bool = True):
        """Plot prediction vs target scatter plots"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Temperature plot
        ax1.scatter(self.temperature_targets, self.temperature_predictions, 
                   alpha=0.6, s=30)
        ax1.plot([min(self.temperature_targets), max(self.temperature_targets)],
                [min(self.temperature_targets), max(self.temperature_targets)], 
                'r--', lw=2, label='Perfect prediction')
        ax1.set_xlabel('True Temperature (K)')
        ax1.set_ylabel('Predicted Temperature (K)')
        ax1.set_title('Temperature Predictions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Tint plot
        ax2.scatter(self.tint_targets, self.tint_predictions, 
                   alpha=0.6, s=30)
        ax2.plot([min(self.tint_targets), max(self.tint_targets)],
                [min(self.tint_targets), max(self.tint_targets)], 
                'r--', lw=2, label='Perfect prediction')
        ax2.set_xlabel('True Tint')
        ax2.set_ylabel('Predicted Tint')
        ax2.set_title('Tint Predictions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_error_distribution(self, save_path: str = None, show: bool = True):
        """Plot error distribution histograms"""
        
        temp_errors = np.array(self.temperature_predictions) - np.array(self.temperature_targets)
        tint_errors = np.array(self.tint_predictions) - np.array(self.tint_targets)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Temperature error distribution
        ax1.hist(temp_errors, bins=50, alpha=0.7, edgecolor='black')
        ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect prediction')
        ax1.axvline(np.mean(temp_errors), color='orange', linestyle='--', 
                   linewidth=2, label=f'Mean error: {np.mean(temp_errors):.1f}K')
        ax1.set_xlabel('Temperature Error (K)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Temperature Prediction Errors')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Tint error distribution
        ax2.hist(tint_errors, bins=50, alpha=0.7, edgecolor='black')
        ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect prediction')
        ax2.axvline(np.mean(tint_errors), color='orange', linestyle='--', 
                   linewidth=2, label=f'Mean error: {np.mean(tint_errors):.1f}')
        ax2.set_xlabel('Tint Error')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Tint Prediction Errors')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()

class ConsistencyMetrics:
    """Metrics for evaluating prediction consistency"""
    
    def __init__(self):
        self.feature_similarities = []
        self.prediction_differences = []
    
    def update(self, features1: torch.Tensor, features2: torch.Tensor,
               pred1: Dict[str, torch.Tensor], pred2: Dict[str, torch.Tensor]):
        """Update consistency metrics"""
        
        # Feature similarity (cosine similarity)
        similarities = torch.nn.functional.cosine_similarity(features1, features2, dim=1)
        self.feature_similarities.extend(similarities.detach().cpu().numpy())
        
        # Prediction differences
        temp_diff = torch.abs(pred1['temperature'] - pred2['temperature'])
        tint_diff = torch.abs(pred1['tint'] - pred2['tint'])
        
        self.prediction_differences.extend(temp_diff.detach().cpu().numpy())
        self.prediction_differences.extend(tint_diff.detach().cpu().numpy())
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute consistency metrics"""
        return {
            'avg_feature_similarity': np.mean(self.feature_similarities),
            'avg_prediction_difference': np.mean(self.prediction_differences),
            'consistency_score': np.mean(self.feature_similarities) - 
                               0.1 * np.mean(self.prediction_differences)
        }

def compute_competition_score(temperature_pred: List[float], 
                            temperature_true: List[float],
                            tint_pred: List[float], 
                            tint_true: List[float]) -> Dict[str, float]:
    """Compute the exact competition scoring metrics"""
    
    temp_mae = mean_absolute_error(temperature_true, temperature_pred)
    tint_mae = mean_absolute_error(tint_true, tint_pred)
    
    # Competition formula: 1 / (1 + MAE)
    temp_score = 1 / (1 + temp_mae)
    tint_score = 1 / (1 + tint_mae)
    
    # Final score (assuming equal weighting)
    final_score = (temp_score + tint_score) / 2
    
    return {
        'temperature_mae': temp_mae,
        'temperature_score': temp_score,
        'tint_mae': tint_mae, 
        'tint_score': tint_score,
        'final_score': final_score
    }