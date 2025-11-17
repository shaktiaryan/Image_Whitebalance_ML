import torch
import torch.nn as nn
from typing import Dict, Any, List
import torch.nn.functional as F

from .cnn_backbone import ImageEncoder, MetadataEncoder, AttentionFusion

class MultiModalWhiteBalanceNet(nn.Module):
    """Multi-modal neural network for White Balance prediction"""
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Image encoder
        self.image_encoder = ImageEncoder(
            backbone=config.model.backbone,
            pretrained=config.model.pretrained,
            feature_dim=config.model.image_feature_dim
        )
        
        # Metadata encoder (will be initialized after data loading)
        self.metadata_encoder = None
        self.metadata_feature_dim = None
        
        # Fusion layer
        self.fusion_type = getattr(config.model, 'fusion_type', 'concat')  # 'concat' or 'attention'
        
        if self.fusion_type == 'attention':
            self.fusion = AttentionFusion(
                image_dim=config.model.image_feature_dim,
                metadata_dim=128,  # Will be updated
                hidden_dim=config.model.fusion_dim
            )
            fusion_output_dim = config.model.fusion_dim
        else:
            # Simple concatenation
            self.fusion = None
            fusion_output_dim = config.model.image_feature_dim + 128  # Will be updated
        
        # Output heads
        self.temperature_head = self._create_output_head(
            fusion_output_dim, 1, config.model.dropout_rate
        )
        self.tint_head = self._create_output_head(
            fusion_output_dim, 1, config.model.dropout_rate
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def initialize_metadata_encoder(self, categorical_dims: Dict[str, int]):
        """Initialize metadata encoder with categorical dimensions"""
        if self.config.model.numerical_features or self.config.model.categorical_features:
            metadata_output_dim = 128  # Can be made configurable
            self.metadata_encoder = MetadataEncoder(
                numerical_features=self.config.model.numerical_features,
                categorical_features=self.config.model.categorical_features,
                categorical_dims=categorical_dims,
                hidden_dims=self.config.model.metadata_hidden_dims,
                output_dim=metadata_output_dim,
                dropout_rate=self.config.model.dropout_rate
            )
            self.metadata_feature_dim = metadata_output_dim
            
            # Update fusion layer if needed
            if self.fusion_type == 'attention':
                self.fusion = AttentionFusion(
                    image_dim=self.config.model.image_feature_dim,
                    metadata_dim=metadata_output_dim,
                    hidden_dim=self.config.model.fusion_dim
                )
                fusion_output_dim = self.config.model.fusion_dim
            else:
                fusion_output_dim = self.config.model.image_feature_dim + metadata_output_dim
            
            # Recreate output heads with correct input dimension
            self.temperature_head = self._create_output_head(
                fusion_output_dim, 1, self.config.model.dropout_rate
            )
            self.tint_head = self._create_output_head(
                fusion_output_dim, 1, self.config.model.dropout_rate
            )
    
    def _create_output_head(self, input_dim: int, output_dim: int, dropout_rate: float) -> nn.Module:
        """Create output head for regression"""
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim // 4, output_dim)
        )
    
    def _initialize_weights(self):
        """Initialize output head weights"""
        for head in [self.temperature_head, self.tint_head]:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            batch: Dictionary containing 'image' and 'metadata'
            
        Returns:
            Dictionary with 'temperature' and 'tint' predictions
        """
        # Extract image features
        image_features = self.image_encoder(batch['image'])
        
        # Extract metadata features
        if self.metadata_encoder is not None and 'metadata' in batch:
            metadata_features = self.metadata_encoder(batch['metadata'])
        else:
            # Create dummy metadata features if not available
            batch_size = image_features.size(0)
            metadata_features = torch.zeros(
                batch_size, 128, device=image_features.device
            )
        
        # Fuse features
        if self.fusion_type == 'attention' and self.fusion is not None:
            fused_features = self.fusion(image_features, metadata_features)
        else:
            # Simple concatenation
            fused_features = torch.cat([image_features, metadata_features], dim=1)
        
        # Predict temperature and tint
        temperature = self.temperature_head(fused_features).squeeze(-1)
        tint = self.tint_head(fused_features).squeeze(-1)
        
        return {
            'temperature': temperature,
            'tint': tint,
            'image_features': image_features,
            'metadata_features': metadata_features,
            'fused_features': fused_features
        }

class ConsistencyNet(nn.Module):
    """Network with consistency regularization"""
    
    def __init__(self, base_net: MultiModalWhiteBalanceNet):
        super().__init__()
        self.base_net = base_net
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass with consistency handling"""
        # Regular forward pass
        outputs = self.base_net(batch)
        
        # If consistent images are provided, process them too
        if 'consistent_image' in batch:
            consistent_batch = batch.copy()
            consistent_batch['image'] = batch['consistent_image']
            
            consistent_outputs = self.base_net(consistent_batch)
            
            outputs.update({
                'consistent_temperature': consistent_outputs['temperature'],
                'consistent_tint': consistent_outputs['tint'],
                'consistent_fused_features': consistent_outputs['fused_features']
            })
        
        return outputs

class TemperatureAwareNet(MultiModalWhiteBalanceNet):
    """Network with temperature-aware processing"""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Add temperature conditioning
        self.temp_conditioning = nn.Sequential(
            nn.Linear(1, 64),  # Current temperature input
            nn.ReLU(inplace=True),
            nn.Linear(64, 32)
        )
        
        # Update fusion dimension to include temperature conditioning
        if self.fusion_type == 'attention':
            original_dim = config.model.fusion_dim
        else:
            original_dim = config.model.image_feature_dim + 128
        
        new_fusion_dim = original_dim + 32  # Add temperature conditioning
        
        # Recreate output heads
        self.temperature_head = self._create_output_head(
            new_fusion_dim, 1, config.model.dropout_rate
        )
        self.tint_head = self._create_output_head(
            new_fusion_dim, 1, config.model.dropout_rate
        )
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass with temperature conditioning"""
        # Get base features
        outputs = super().forward(batch)
        fused_features = outputs['fused_features']
        
        # Add current temperature conditioning if available
        if 'metadata' in batch and 'numerical' in batch['metadata']:
            # Assume first feature is current temperature (currTemp)
            curr_temp = batch['metadata']['numerical'][:, 0:1]  # [B, 1]
            temp_cond = self.temp_conditioning(curr_temp)
            
            # Combine with fused features
            enhanced_features = torch.cat([fused_features, temp_cond], dim=1)
        else:
            enhanced_features = fused_features
        
        # Re-predict with enhanced features
        temperature = self.temperature_head(enhanced_features).squeeze(-1)
        tint = self.tint_head(enhanced_features).squeeze(-1)
        
        outputs.update({
            'temperature': temperature,
            'tint': tint,
            'enhanced_features': enhanced_features
        })
        
        return outputs

def create_model(config, categorical_dims: Dict[str, int] = None) -> nn.Module:
    """Factory function to create models"""
    
    model_type = getattr(config.model, 'type', 'multimodal')
    
    if model_type == 'multimodal':
        model = MultiModalWhiteBalanceNet(config)
    elif model_type == 'temperature_aware':
        model = TemperatureAwareNet(config)
    elif model_type == 'consistency':
        base_model = MultiModalWhiteBalanceNet(config)
        model = ConsistencyNet(base_model)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Initialize metadata encoder if categorical dimensions are provided
    if categorical_dims is not None and hasattr(model, 'initialize_metadata_encoder'):
        model.initialize_metadata_encoder(categorical_dims)
    elif categorical_dims is not None and hasattr(model, 'base_net'):
        model.base_net.initialize_metadata_encoder(categorical_dims)
    
    return model