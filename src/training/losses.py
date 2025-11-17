import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple

class TemperatureAwareLoss(nn.Module):
    """Loss function that accounts for non-linear temperature sensitivity"""
    
    def __init__(self, base_temp: float = 5000.0, scale_factor: float = 1000.0):
        super().__init__()
        self.base_temp = base_temp
        self.scale_factor = scale_factor
    
    def temperature_weight(self, temperature: torch.Tensor) -> torch.Tensor:
        """
        Calculate weights based on temperature sensitivity
        Lower temperatures get higher weights due to higher sensitivity
        
        Args:
            temperature: Temperature values [B]
        Returns:
            weights: Sensitivity weights [B]
        """
        # Normalize temperature for weight calculation
        temp_normalized = temperature / self.scale_factor
        
        # Weight inversely proportional to temperature (with smoothing)
        # Higher weights for lower temperatures
        weights = self.base_temp / (temperature + 1e-6)
        
        # Normalize weights to have reasonable scale
        weights = torch.clamp(weights, min=0.5, max=5.0)
        
        return weights
    
    def forward(self, pred_temp: torch.Tensor, true_temp: torch.Tensor) -> torch.Tensor:
        """
        Calculate temperature-aware loss
        
        Args:
            pred_temp: Predicted temperature [B]
            true_temp: True temperature [B]
        Returns:
            loss: Weighted temperature loss
        """
        # Calculate base loss (MAE)
        base_loss = F.l1_loss(pred_temp, true_temp, reduction='none')
        
        # Calculate temperature-dependent weights
        weights = self.temperature_weight(true_temp)
        
        # Apply weights
        weighted_loss = base_loss * weights
        
        return weighted_loss.mean()

class ConsistencyLoss(nn.Module):
    """Loss function for promoting consistency between similar images"""
    
    def __init__(self, feature_weight: float = 1.0, prediction_weight: float = 1.0):
        super().__init__()
        self.feature_weight = feature_weight
        self.prediction_weight = prediction_weight
    
    def forward(self, 
                features1: torch.Tensor, 
                features2: torch.Tensor,
                pred1: torch.Tensor,
                pred2: torch.Tensor) -> torch.Tensor:
        """
        Calculate consistency loss between two versions of similar images
        
        Args:
            features1: Features from first image [B, D]
            features2: Features from second image [B, D]
            pred1: Predictions from first image [B, 2] (temp, tint)
            pred2: Predictions from second image [B, 2] (temp, tint)
        Returns:
            loss: Consistency loss
        """
        # Feature consistency loss (cosine similarity)
        feature_loss = 1 - F.cosine_similarity(features1, features2, dim=1).mean()
        
        # Prediction consistency loss (L1)
        prediction_loss = F.l1_loss(pred1, pred2)
        
        # Combined loss
        total_loss = (self.feature_weight * feature_loss + 
                     self.prediction_weight * prediction_loss)
        
        return total_loss

class FocalLoss(nn.Module):
    """Focal loss for handling hard examples"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate focal loss (adapted for regression)
        
        Args:
            pred: Predictions [B]
            target: Targets [B]
        Returns:
            loss: Focal loss
        """
        # Calculate L1 loss
        l1_loss = F.l1_loss(pred, target, reduction='none')
        
        # Calculate focusing term
        # Normalize error for focusing calculation
        normalized_error = l1_loss / (l1_loss.mean() + 1e-8)
        focal_weight = self.alpha * torch.pow(normalized_error, self.gamma)
        
        # Apply focal weighting
        focal_loss = focal_weight * l1_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CombinedWhiteBalanceLoss(nn.Module):
    """Combined loss function for white balance prediction"""
    
    def __init__(self, 
                 temp_weight: float = 1.0,
                 tint_weight: float = 1.0, 
                 consistency_weight: float = 0.1,
                 use_temperature_aware: bool = True,
                 use_focal: bool = False):
        super().__init__()
        
        self.temp_weight = temp_weight
        self.tint_weight = tint_weight
        self.consistency_weight = consistency_weight
        self.use_temperature_aware = use_temperature_aware
        self.use_focal = use_focal
        
        # Loss components
        if use_temperature_aware:
            self.temp_loss_fn = TemperatureAwareLoss()
        else:
            self.temp_loss_fn = nn.L1Loss()
        
        if use_focal:
            self.tint_loss_fn = FocalLoss()
        else:
            self.tint_loss_fn = nn.L1Loss()
        
        self.consistency_loss_fn = ConsistencyLoss()
    
    def forward(self, outputs: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate combined loss
        
        Args:
            outputs: Model outputs dictionary
            targets: Target values dictionary
        Returns:
            loss_dict: Dictionary of individual and total losses
        """
        losses = {}
        
        # Temperature loss
        temp_loss = self.temp_loss_fn(outputs['temperature'], targets['temperature'])
        losses['temperature_loss'] = temp_loss
        
        # Tint loss
        tint_loss = self.tint_loss_fn(outputs['tint'], targets['tint'])
        losses['tint_loss'] = tint_loss
        
        # Base prediction loss
        prediction_loss = self.temp_weight * temp_loss + self.tint_weight * tint_loss
        losses['prediction_loss'] = prediction_loss
        
        # Consistency loss (if consistent predictions available)
        consistency_loss = torch.tensor(0.0, device=temp_loss.device)
        if ('consistent_temperature' in outputs and 
            'consistent_tint' in outputs and
            self.consistency_weight > 0):
            
            # Features consistency
            if 'fused_features' in outputs and 'consistent_fused_features' in outputs:
                feature_consistency = self.consistency_loss_fn(
                    outputs['fused_features'],
                    outputs['consistent_fused_features'],
                    torch.stack([outputs['temperature'], outputs['tint']], dim=1),
                    torch.stack([outputs['consistent_temperature'], outputs['consistent_tint']], dim=1)
                )
                consistency_loss = consistency_loss + feature_consistency
        
        losses['consistency_loss'] = consistency_loss
        
        # Total loss
        total_loss = prediction_loss + self.consistency_weight * consistency_loss
        losses['total_loss'] = total_loss
        
        return losses

class AdaptiveLoss(nn.Module):
    """Adaptive loss that adjusts weights during training"""
    
    def __init__(self, initial_temp_weight: float = 1.0, 
                 initial_tint_weight: float = 1.0):
        super().__init__()
        
        # Learnable loss weights
        self.log_temp_weight = nn.Parameter(torch.log(torch.tensor(initial_temp_weight)))
        self.log_tint_weight = nn.Parameter(torch.log(torch.tensor(initial_tint_weight)))
    
    def forward(self, temp_loss: torch.Tensor, 
                tint_loss: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate adaptive weighted loss
        
        Args:
            temp_loss: Temperature loss
            tint_loss: Tint loss
        Returns:
            total_loss: Weighted total loss
            weights: Current weights dictionary
        """
        # Calculate adaptive weights
        temp_weight = torch.exp(self.log_temp_weight)
        tint_weight = torch.exp(self.log_tint_weight)
        
        # Adaptive weighted loss with uncertainty
        total_loss = (temp_loss / (2 * temp_weight) + self.log_temp_weight + 
                     tint_loss / (2 * tint_weight) + self.log_tint_weight)
        
        weights = {
            'temp_weight': temp_weight.item(),
            'tint_weight': tint_weight.item()
        }
        
        return total_loss, weights

class RobustLoss(nn.Module):
    """Robust loss function less sensitive to outliers"""
    
    def __init__(self, loss_type: str = 'huber', delta: float = 1.0):
        super().__init__()
        self.loss_type = loss_type
        self.delta = delta
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate robust loss
        
        Args:
            pred: Predictions
            target: Targets
        Returns:
            loss: Robust loss value
        """
        if self.loss_type == 'huber':
            return F.smooth_l1_loss(pred, target, beta=self.delta)
        elif self.loss_type == 'cauchy':
            diff = pred - target
            return torch.log(1 + (diff / self.delta) ** 2).mean()
        elif self.loss_type == 'welsch':
            diff = pred - target
            return (1 - torch.exp(-(diff / self.delta) ** 2)).mean()
        else:
            return F.l1_loss(pred, target)

def create_loss_function(config) -> nn.Module:
    """Factory function to create loss functions"""
    
    loss_type = getattr(config.training, 'loss_type', 'combined')
    
    if loss_type == 'combined':
        return CombinedWhiteBalanceLoss(
            temp_weight=config.training.temperature_weight,
            tint_weight=config.training.tint_weight,
            consistency_weight=config.training.consistency_weight,
            use_temperature_aware=True,
            use_focal=False
        )
    elif loss_type == 'temperature_aware':
        return TemperatureAwareLoss()
    elif loss_type == 'robust':
        return RobustLoss(loss_type='huber')
    elif loss_type == 'adaptive':
        return AdaptiveLoss()
    else:
        return nn.L1Loss()