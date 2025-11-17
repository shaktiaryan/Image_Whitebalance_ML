import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple
import cv2
from PIL import Image
import os

# Set style
plt.style.use('default')
sns.set_palette("husl")

class DataVisualizer:
    """Visualization utilities for the white balance dataset"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        
    def plot_target_distributions(self, df: pd.DataFrame, save_path: str = None):
        """Plot distribution of target variables"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Temperature distribution
        ax1.hist(df['Temperature'], bins=50, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Temperature (K)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Temperature Distribution')
        ax1.axvline(df['Temperature'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["Temperature"].mean():.0f}K')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Tint distribution
        ax2.hist(df['Tint'], bins=50, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Tint')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Tint Distribution')
        ax2.axvline(df['Tint'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["Tint"].mean():.1f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_temperature_vs_features(self, df: pd.DataFrame, save_path: str = None):
        """Plot temperature vs various input features"""
        
        numerical_features = ['currTemp', 'currTint', 'aperture', 'focalLength', 
                            'isoSpeedRating', 'shutterSpeed', 'intensity', 'ev']
        
        # Filter features that exist in dataframe
        available_features = [f for f in numerical_features if f in df.columns]
        
        if not available_features:
            print("No numerical features found for plotting")
            return
        
        n_features = len(available_features)
        cols = 3
        rows = (n_features + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for i, feature in enumerate(available_features):
            ax = axes[i]
            
            # Scatter plot with alpha for overlapping points
            ax.scatter(df[feature], df['Temperature'], alpha=0.6, s=20)
            ax.set_xlabel(feature)
            ax.set_ylabel('Temperature (K)')
            ax.set_title(f'Temperature vs {feature}')
            ax.grid(True, alpha=0.3)
            
            # Add correlation coefficient
            corr = df[feature].corr(df['Temperature'])
            ax.text(0.05, 0.95, f'Corr: {corr:.3f}', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide empty subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_categorical_analysis(self, df: pd.DataFrame, save_path: str = None):
        """Analyze categorical features"""
        
        categorical_features = ['camera_model', 'camera_group', 'flashFired']
        available_features = [f for f in categorical_features if f in df.columns]
        
        if not available_features:
            print("No categorical features found")
            return
        
        n_features = len(available_features)
        fig, axes = plt.subplots(n_features, 2, figsize=(15, 6*n_features))
        
        if n_features == 1:
            axes = axes.reshape(1, -1)
        
        for i, feature in enumerate(available_features):
            # Count plot
            df[feature].value_counts().plot(kind='bar', ax=axes[i, 0])
            axes[i, 0].set_title(f'{feature} Distribution')
            axes[i, 0].set_xlabel(feature)
            axes[i, 0].set_ylabel('Count')
            axes[i, 0].tick_params(axis='x', rotation=45)
            
            # Box plot of temperature by category
            df.boxplot(column='Temperature', by=feature, ax=axes[i, 1])
            axes[i, 1].set_title(f'Temperature by {feature}')
            axes[i, 1].set_xlabel(feature)
            axes[i, 1].set_ylabel('Temperature (K)')
            axes[i, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_correlation_matrix(self, df: pd.DataFrame, save_path: str = None):
        """Plot correlation matrix of numerical features"""
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numerical_cols].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm',
                   center=0, square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_interactive_scatter(self, df: pd.DataFrame, save_path: str = None):
        """Create interactive scatter plot with Plotly"""
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Current Temperature vs Target Temperature', 
                          'Current Tint vs Target Tint')
        )
        
        # Temperature scatter
        fig.add_trace(
            go.Scatter(
                x=df['currTemp'], y=df['Temperature'],
                mode='markers',
                marker=dict(size=8, opacity=0.6),
                name='Temperature',
                hovertemplate='Current: %{x}<br>Target: %{y}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Tint scatter
        fig.add_trace(
            go.Scatter(
                x=df['currTint'], y=df['Tint'],
                mode='markers',
                marker=dict(size=8, opacity=0.6),
                name='Tint',
                hovertemplate='Current: %{x}<br>Target: %{y}<extra></extra>'
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Current Temperature (K)", row=1, col=1)
        fig.update_yaxes(title_text="Target Temperature (K)", row=1, col=1)
        fig.update_xaxes(title_text="Current Tint", row=1, col=2)
        fig.update_yaxes(title_text="Target Tint", row=1, col=2)
        
        fig.update_layout(height=500, showlegend=False)
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
    
    def plot_sample_images(self, df: pd.DataFrame, images_dir: str, 
                          n_samples: int = 12, save_path: str = None):
        """Plot sample images with their metadata"""
        
        # Sample random images
        sample_df = df.sample(n_samples).reset_index(drop=True)
        
        cols = 4
        rows = (n_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
        axes = axes.flatten() if n_samples > 1 else [axes]
        
        for i, (idx, row) in enumerate(sample_df.iterrows()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Load and display image - try .tif first, then .tiff
            image_path = f"{images_dir}/{row['id_global']}.tif"
            if not os.path.exists(image_path):
                image_path = f"{images_dir}/{row['id_global']}.tiff"
            try:
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    ax.imshow(image)
                else:
                    # Create placeholder if image not found
                    ax.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center')
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
            except Exception as e:
                ax.text(0.5, 0.5, f'Error:\n{str(e)[:20]}', ha='center', va='center')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            
            # Set title with metadata
            title = f"ID: {row['id_global']}\n"
            title += f"Temp: {row.get('Temperature', 'N/A')}K, "
            title += f"Tint: {row.get('Tint', 'N/A')}\n"
            title += f"Camera: {row.get('camera_model', 'N/A')[:15]}"
            
            ax.set_title(title, fontsize=8)
            ax.axis('off')
        
        # Hide empty subplots
        for i in range(n_samples, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

class TrainingVisualizer:
    """Visualization utilities for training progress"""
    
    def plot_training_curves(self, train_history: List[Dict], 
                           val_history: List[Dict], save_path: str = None):
        """Plot training curves"""
        
        epochs = range(1, len(train_history) + 1)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Combined score
        ax1.plot(epochs, [h['combined_score'] for h in train_history], 
                'b-', label='Train', linewidth=2)
        ax1.plot(epochs, [h['combined_score'] for h in val_history], 
                'r-', label='Validation', linewidth=2)
        ax1.set_title('Competition Score (Higher is Better)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss curves
        ax2.plot(epochs, [h.get('avg_loss', 0) for h in train_history], 
                'b-', label='Train', linewidth=2)
        ax2.plot(epochs, [h.get('avg_loss', 0) for h in val_history], 
                'r-', label='Validation', linewidth=2)
        ax2.set_title('Loss (Lower is Better)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Temperature MAE
        ax3.plot(epochs, [h['temperature_mae'] for h in train_history], 
                'b-', label='Train', linewidth=2)
        ax3.plot(epochs, [h['temperature_mae'] for h in val_history], 
                'r-', label='Validation', linewidth=2)
        ax3.set_title('Temperature MAE (Lower is Better)')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('MAE (K)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Tint MAE
        ax4.plot(epochs, [h['tint_mae'] for h in train_history], 
                'b-', label='Train', linewidth=2)
        ax4.plot(epochs, [h['tint_mae'] for h in val_history], 
                'r-', label='Validation', linewidth=2)
        ax4.set_title('Tint MAE (Lower is Better)')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('MAE')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def create_eda_report(df: pd.DataFrame, images_dir: str, output_dir: str):
    """Create comprehensive EDA report"""
    
    visualizer = DataVisualizer()
    
    # Basic statistics
    print("=== DATASET STATISTICS ===")
    print(f"Number of samples: {len(df)}")
    print(f"Number of features: {df.shape[1]}")
    print("\nTarget variable statistics:")
    if 'Temperature' in df.columns:
        print(f"Temperature - Mean: {df['Temperature'].mean():.1f}K, "
              f"Std: {df['Temperature'].std():.1f}K, "
              f"Range: {df['Temperature'].min():.0f}-{df['Temperature'].max():.0f}K")
    if 'Tint' in df.columns:
        print(f"Tint - Mean: {df['Tint'].mean():.1f}, "
              f"Std: {df['Tint'].std():.1f}, "
              f"Range: {df['Tint'].min():.0f}-{df['Tint'].max():.0f}")
    
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    # Create visualizations
    os.makedirs(output_dir, exist_ok=True)
    
    if 'Temperature' in df.columns and 'Tint' in df.columns:
        visualizer.plot_target_distributions(
            df, save_path=os.path.join(output_dir, 'target_distributions.png')
        )
    
    visualizer.plot_temperature_vs_features(
        df, save_path=os.path.join(output_dir, 'feature_correlations.png')
    )
    
    visualizer.plot_categorical_analysis(
        df, save_path=os.path.join(output_dir, 'categorical_analysis.png')
    )
    
    visualizer.plot_correlation_matrix(
        df, save_path=os.path.join(output_dir, 'correlation_matrix.png')
    )
    
    if os.path.exists(images_dir):
        visualizer.plot_sample_images(
            df, images_dir, save_path=os.path.join(output_dir, 'sample_images.png')
        )
    
    print(f"\nEDA visualizations saved to: {output_dir}")