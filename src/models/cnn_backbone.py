import torch
import torch.nn as nn
import timm
from typing import Dict, Any
import torchvision.models as models

class ImageEncoder(nn.Module):
    """CNN backbone for image feature extraction"""
    
    def __init__(self, backbone: str = "efficientnet_b3", 
                 pretrained: bool = True, 
                 feature_dim: int = 512):
        super().__init__()
        
        self.backbone_name = backbone
        self.feature_dim = feature_dim
        
        # Load backbone model
        if backbone.startswith('efficientnet'):
            self.backbone = timm.create_model(backbone, pretrained=pretrained)
            backbone_feature_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()  # Remove classification head
        
        elif backbone.startswith('resnet'):
            if backbone == 'resnet50':
                self.backbone = models.resnet50(pretrained=pretrained)
            elif backbone == 'resnet101':
                self.backbone = models.resnet101(pretrained=pretrained)
            else:
                self.backbone = models.resnet18(pretrained=pretrained)
            
            backbone_feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove classification head
        
        elif backbone.startswith('convnext'):
            self.backbone = timm.create_model(backbone, pretrained=pretrained)
            backbone_feature_dim = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()  # Remove classification head
        
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Feature projection layer
        self.feature_projection = nn.Sequential(
            nn.Linear(backbone_feature_dim, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Initialize projection layers
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize projection layer weights"""
        for m in self.feature_projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor [B, C, H, W]
        Returns:
            features: Image features [B, feature_dim]
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Project to desired dimension
        features = self.feature_projection(features)
        
        return features

class MetadataEncoder(nn.Module):
    """Neural network for metadata feature encoding"""
    
    def __init__(self, 
                 numerical_features: list,
                 categorical_features: list,
                 categorical_dims: Dict[str, int],
                 hidden_dims: list = [128, 64],
                 output_dim: int = 128,
                 dropout_rate: float = 0.3):
        super().__init__()
        
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.categorical_dims = categorical_dims
        
        # Categorical embeddings
        self.embeddings = nn.ModuleDict()
        embedding_total_dim = 0
        
        for feature in categorical_features:
            if feature in categorical_dims:
                vocab_size = categorical_dims[feature]
                # Embedding dimension based on vocabulary size
                embed_dim = min(16, vocab_size // 2 + 1)
                self.embeddings[feature] = nn.Embedding(vocab_size, embed_dim)
                embedding_total_dim += embed_dim
        
        # Calculate input dimension
        input_dim = len(numerical_features) + embedding_total_dim
        
        # MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.1)
    
    def forward(self, metadata: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            metadata: Dictionary containing numerical and categorical features
        Returns:
            features: Encoded metadata features [B, output_dim]
        """
        features = []
        
        # Process numerical features
        if 'numerical' in metadata and len(self.numerical_features) > 0:
            numerical = metadata['numerical']
            features.append(numerical)
        
        # Process categorical features
        for feature in self.categorical_features:
            feature_key = f'cat_{feature}'
            if feature_key in metadata and feature in self.embeddings:
                cat_input = metadata[feature_key]
                embedded = self.embeddings[feature](cat_input)
                features.append(embedded)
        
        # Concatenate all features
        if features:
            combined_features = torch.cat(features, dim=1)
        else:
            # Handle case with no features
            batch_size = next(iter(metadata.values())).size(0)
            combined_features = torch.zeros(batch_size, 1, device=next(iter(metadata.values())).device)
        
        # Pass through MLP
        output = self.mlp(combined_features)
        
        return output

class AttentionFusion(nn.Module):
    """Attention-based fusion of image and metadata features"""
    
    def __init__(self, image_dim: int, metadata_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.image_projection = nn.Linear(image_dim, hidden_dim)
        self.metadata_projection = nn.Linear(metadata_dim, hidden_dim)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, image_features: torch.Tensor, 
                metadata_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_features: [B, image_dim]
            metadata_features: [B, metadata_dim]
        Returns:
            fused_features: [B, hidden_dim]
        """
        # Project to same dimension
        img_proj = self.image_projection(image_features)  # [B, hidden_dim]
        meta_proj = self.metadata_projection(metadata_features)  # [B, hidden_dim]
        
        # Add sequence dimension for attention
        img_seq = img_proj.unsqueeze(1)  # [B, 1, hidden_dim]
        meta_seq = meta_proj.unsqueeze(1)  # [B, 1, hidden_dim]
        
        # Concatenate for attention
        features = torch.cat([img_seq, meta_seq], dim=1)  # [B, 2, hidden_dim]
        
        # Apply self-attention
        attended, _ = self.attention(features, features, features)
        
        # Global average pooling
        pooled = attended.mean(dim=1)  # [B, hidden_dim]
        
        # Layer normalization and residual connection
        output = self.layer_norm(pooled + img_proj + meta_proj)
        
        return output

def create_backbone(backbone_name: str, pretrained: bool = True) -> nn.Module:
    """Factory function to create backbone models"""
    
    if backbone_name.startswith('efficientnet'):
        model = timm.create_model(backbone_name, pretrained=pretrained)
        feature_dim = model.classifier.in_features
        model.classifier = nn.Identity()
    
    elif backbone_name.startswith('resnet'):
        if backbone_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
        elif backbone_name == 'resnet101':
            model = models.resnet101(pretrained=pretrained)
        else:
            model = models.resnet18(pretrained=pretrained)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    
    elif backbone_name.startswith('convnext'):
        model = timm.create_model(backbone_name, pretrained=pretrained)
        feature_dim = model.head.fc.in_features
        model.head.fc = nn.Identity()
    
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    
    return model, feature_dim