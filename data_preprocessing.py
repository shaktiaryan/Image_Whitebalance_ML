#!/usr/bin/env python3
"""
Comprehensive Data Cleaning and Preprocessing for Aftershoot White Balance
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

def load_and_analyze_data(data_dir):
    """Load and perform initial analysis of the dataset"""
    
    print("🔍 STEP 1: Loading and Analyzing Data")
    print("=" * 50)
    
    # Load filtered CSV (only images that exist)
    train_csv = os.path.join(data_dir, 'Train', 'sliders_filtered.csv')
    if not os.path.exists(train_csv):
        train_csv = os.path.join(data_dir, 'Train', 'sliders.csv')
    
    df = pd.read_csv(train_csv)
    images_dir = os.path.join(data_dir, 'Train', 'images')
    
    print(f"📁 Loaded dataset: {train_csv}")
    print(f"📊 Dataset shape: {df.shape}")
    print(f"🖼️  Images directory: {images_dir}")
    
    # Basic info
    print(f"\n📈 Dataset Statistics:")
    print(f"   • Total samples: {len(df)}")
    print(f"   • Features: {df.shape[1]}")
    print(f"   • Missing values: {df.isnull().sum().sum()}")
    
    return df, images_dir

def analyze_features(df):
    """Analyze feature distributions and quality"""
    
    print(f"\n🔬 STEP 2: Feature Analysis")
    print("=" * 50)
    
    # Categorical features
    categorical_features = ['flashFired']
    
    # Numerical features  
    numerical_features = [
        'aperture', 'focalLength', 'isoSpeedRating', 
        'shutterSpeed', 'currTemp', 'currTint'
    ]
    
    # Target variables
    target_features = ['Temperature', 'Tint']
    
    # Time features
    time_features = ['copyCreationTime', 'captureTime', 'touchTime']
    
    print(f"📋 Feature Categories:")
    print(f"   • Categorical: {categorical_features}")
    print(f"   • Numerical: {numerical_features}")
    print(f"   • Targets: {target_features}")
    print(f"   • Time: {time_features}")
    
    # Analyze distributions
    print(f"\n📊 Target Variable Analysis:")
    for target in target_features:
        stats = df[target].describe()
        print(f"   {target}:")
        print(f"      Mean: {stats['mean']:.1f}, Std: {stats['std']:.1f}")
        print(f"      Range: {stats['min']:.0f} - {stats['max']:.0f}")
        print(f"      Skewness: {df[target].skew():.2f}")
    
    # Check for outliers
    print(f"\n🚨 Outlier Analysis:")
    for feature in numerical_features + target_features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[feature] < Q1 - 1.5*IQR) | (df[feature] > Q3 + 1.5*IQR)]
        print(f"   {feature}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")
    
    return categorical_features, numerical_features, target_features, time_features

def check_image_quality(df, images_dir, sample_size=100):
    """Check image quality and consistency"""
    
    print(f"\n📸 STEP 3: Image Quality Analysis")
    print("=" * 50)
    
    # Sample random images to check
    sample_ids = df['id_global'].sample(min(sample_size, len(df)), random_state=42)
    
    image_stats = {
        'total_checked': 0,
        'successful_loads': 0,
        'failed_loads': 0,
        'sizes': [],
        'formats': [],
        'corrupted': []
    }
    
    print(f"🔍 Checking {len(sample_ids)} sample images...")
    
    for img_id in sample_ids:
        image_stats['total_checked'] += 1
        
        # Try to load image
        img_path = os.path.join(images_dir, f"{img_id}.tif")
        if not os.path.exists(img_path):
            img_path = os.path.join(images_dir, f"{img_id}.tiff")
        
        try:
            # Try with OpenCV
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is not None:
                height, width, channels = img.shape
                image_stats['sizes'].append((width, height))
                image_stats['successful_loads'] += 1
            else:
                # Try with PIL as backup
                with Image.open(img_path) as pil_img:
                    image_stats['sizes'].append(pil_img.size)
                    image_stats['successful_loads'] += 1
                    
        except Exception as e:
            image_stats['failed_loads'] += 1
            image_stats['corrupted'].append(img_id)
    
    # Analyze image sizes
    if image_stats['sizes']:
        sizes_array = np.array(image_stats['sizes'])
        unique_sizes = np.unique(sizes_array, axis=0)
        
        print(f"📊 Image Quality Results:")
        print(f"   • Successful loads: {image_stats['successful_loads']}/{image_stats['total_checked']}")
        print(f"   • Failed loads: {image_stats['failed_loads']}")
        print(f"   • Unique image sizes: {len(unique_sizes)}")
        
        if len(unique_sizes) <= 10:
            print(f"   • Common sizes: {unique_sizes[:5].tolist()}")
        
        # Check if all images have same size
        if len(unique_sizes) == 1:
            print(f"   ✅ All images have consistent size: {unique_sizes[0]}")
        else:
            print(f"   ⚠️  Multiple image sizes detected - will need resizing")
    
    return image_stats

def detect_anomalies(df):
    """Detect anomalies in the dataset"""
    
    print(f"\n🔍 STEP 4: Anomaly Detection")
    print("=" * 50)
    
    anomalies = []
    
    # 1. Temperature anomalies (extremely high/low values)
    temp_q99 = df['Temperature'].quantile(0.99)
    temp_q01 = df['Temperature'].quantile(0.01)
    temp_anomalies = df[(df['Temperature'] > temp_q99) | (df['Temperature'] < temp_q01)]
    print(f"🌡️  Temperature anomalies: {len(temp_anomalies)} samples")
    print(f"   Range: {temp_q01:.0f}K - {temp_q99:.0f}K (99% of data)")
    
    # 2. Tint anomalies
    tint_q99 = df['Tint'].quantile(0.99)
    tint_q01 = df['Tint'].quantile(0.01)
    tint_anomalies = df[(df['Tint'] > tint_q99) | (df['Tint'] < tint_q01)]
    print(f"🎨 Tint anomalies: {len(tint_anomalies)} samples")
    print(f"   Range: {tint_q01:.1f} - {tint_q99:.1f} (99% of data)")
    
    # 3. ISO anomalies (extremely high ISO values might be noisy)
    if 'isoSpeedRating' in df.columns:
        iso_q95 = df['isoSpeedRating'].quantile(0.95)
        iso_anomalies = df[df['isoSpeedRating'] > iso_q95]
        print(f"📷 High ISO samples (>95th percentile): {len(iso_anomalies)}")
        print(f"   Threshold: ISO {iso_q95:.0f}")
    
    # 4. Aperture anomalies
    if 'aperture' in df.columns:
        # Very wide apertures (f/1.0 or wider) might be rare/special cases
        wide_aperture = df[df['aperture'] <= 1.2]
        print(f"🔍 Wide aperture samples (≤f/1.2): {len(wide_aperture)}")
    
    # 5. Check for duplicate images
    duplicates = df[df.duplicated(subset=['id_global'], keep=False)]
    print(f"🔄 Duplicate image IDs: {len(duplicates)}")
    
    return {
        'temperature_anomalies': temp_anomalies,
        'tint_anomalies': tint_anomalies,
        'iso_anomalies': iso_anomalies if 'isoSpeedRating' in df.columns else pd.DataFrame(),
        'duplicates': duplicates
    }

def create_features(df):
    """Create new engineered features"""
    
    print(f"\n⚙️ STEP 5: Feature Engineering")
    print("=" * 50)
    
    df_processed = df.copy()
    
    # 1. Camera settings combinations
    if 'aperture' in df.columns and 'shutterSpeed' in df.columns and 'isoSpeedRating' in df.columns:
        # Exposure value (simplified)
        df_processed['exposure_combo'] = (
            np.log2(df_processed['aperture']**2) - 
            np.log2(df_processed['shutterSpeed']) + 
            np.log2(df_processed['isoSpeedRating']/100)
        )
        print("📷 Created exposure_combo feature")
    
    # 2. Temperature differences
    if 'currTemp' in df.columns and 'Temperature' in df.columns:
        df_processed['temp_adjustment'] = df_processed['Temperature'] - df_processed['currTemp']
        print("🌡️  Created temp_adjustment feature")
    
    # 3. Tint differences  
    if 'currTint' in df.columns and 'Tint' in df.columns:
        df_processed['tint_adjustment'] = df_processed['Tint'] - df_processed['currTint']
        print("🎨 Created tint_adjustment feature")
    
    # 4. Categorical binning
    if 'isoSpeedRating' in df.columns:
        df_processed['iso_category'] = pd.cut(
            df_processed['isoSpeedRating'], 
            bins=[0, 200, 800, 3200, np.inf], 
            labels=['low', 'medium', 'high', 'very_high']
        )
        print("📊 Created iso_category feature")
    
    if 'aperture' in df.columns:
        df_processed['aperture_category'] = pd.cut(
            df_processed['aperture'],
            bins=[0, 2.0, 4.0, 8.0, np.inf],
            labels=['wide', 'moderate', 'narrow', 'very_narrow']
        )
        print("🔍 Created aperture_category feature")
    
    # 5. Time-based features (if needed)
    time_cols = ['copyCreationTime', 'captureTime', 'touchTime']
    for col in time_cols:
        if col in df.columns:
            try:
                df_processed[f'{col}_hour'] = pd.to_datetime(df_processed[col]).dt.hour
                print(f"⏰ Created {col}_hour feature")
            except:
                pass  # Skip if time parsing fails
    
    new_features = [col for col in df_processed.columns if col not in df.columns]
    print(f"✅ Total new features created: {len(new_features)}")
    if new_features:
        print(f"   New features: {new_features}")
    
    return df_processed

def preprocess_data(df, categorical_features, numerical_features):
    """Preprocess features for training"""
    
    print(f"\n🛠️ STEP 6: Data Preprocessing")
    print("=" * 50)
    
    df_processed = df.copy()
    
    # 1. Handle categorical variables
    print("🏷️  Processing categorical features...")
    for feature in categorical_features:
        if feature in df_processed.columns:
            # Convert to category and get dummies if needed
            df_processed[feature] = df_processed[feature].astype('category')
            print(f"   {feature}: {df_processed[feature].nunique()} categories")
    
    # 2. Handle numerical features
    print("🔢 Processing numerical features...")
    
    # Detect skewed features
    skewed_features = []
    for feature in numerical_features:
        if feature in df_processed.columns:
            skewness = df_processed[feature].skew()
            if abs(skewness) > 1.0:  # Highly skewed
                skewed_features.append(feature)
                print(f"   {feature}: skewness = {skewness:.2f} (highly skewed)")
    
    # Apply log transformation to highly skewed features
    for feature in skewed_features:
        if (df_processed[feature] > 0).all():  # Only if all values are positive
            df_processed[f'{feature}_log'] = np.log1p(df_processed[feature])
            print(f"   Applied log transform to {feature}")
    
    # 3. Outlier handling (optional - create flags)
    print("🚨 Creating outlier flags...")
    for feature in numerical_features:
        if feature in df_processed.columns:
            Q1 = df_processed[feature].quantile(0.25)
            Q3 = df_processed[feature].quantile(0.75)
            IQR = Q3 - Q1
            
            # Create outlier flag
            outlier_mask = (
                (df_processed[feature] < Q1 - 2*IQR) | 
                (df_processed[feature] > Q3 + 2*IQR)
            )
            df_processed[f'{feature}_outlier'] = outlier_mask.astype(int)
            
            outlier_count = outlier_mask.sum()
            if outlier_count > 0:
                print(f"   {feature}: {outlier_count} outliers flagged")
    
    return df_processed

def create_splits(df, test_size=0.2, val_size=0.2, random_state=42):
    """Create train/validation/test splits"""
    
    print(f"\n📊 STEP 7: Creating Data Splits")
    print("=" * 50)
    
    # First split: train+val vs test
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_state, shuffle=True
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for reduced dataset
    train, val = train_test_split(
        train_val, test_size=val_size_adjusted, random_state=random_state, shuffle=True
    )
    
    print(f"📈 Data split results:")
    print(f"   • Training: {len(train)} samples ({len(train)/len(df)*100:.1f}%)")
    print(f"   • Validation: {len(val)} samples ({len(val)/len(df)*100:.1f}%)")
    print(f"   • Test: {len(test)} samples ({len(test)/len(df)*100:.1f}%)")
    
    # Save splits
    output_dir = "outputs/preprocessed_data"
    os.makedirs(output_dir, exist_ok=True)
    
    train.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"💾 Saved splits to: {output_dir}/")
    
    return train, val, test

def create_visualizations(df, output_dir="outputs/preprocessing_analysis"):
    """Create comprehensive visualizations"""
    
    print(f"\n📊 STEP 8: Creating Visualizations")
    print("=" * 50)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    fig_size = (12, 8)
    
    # 1. Target variable distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Temperature distribution
    axes[0,0].hist(df['Temperature'], bins=50, alpha=0.7, color='orange')
    axes[0,0].set_title('Temperature Distribution')
    axes[0,0].set_xlabel('Temperature (K)')
    axes[0,0].set_ylabel('Frequency')
    
    # Temperature box plot
    axes[0,1].boxplot(df['Temperature'])
    axes[0,1].set_title('Temperature Box Plot')
    axes[0,1].set_ylabel('Temperature (K)')
    
    # Tint distribution
    axes[1,0].hist(df['Tint'], bins=50, alpha=0.7, color='blue')
    axes[1,0].set_title('Tint Distribution')
    axes[1,0].set_xlabel('Tint')
    axes[1,0].set_ylabel('Frequency')
    
    # Tint box plot
    axes[1,1].boxplot(df['Tint'])
    axes[1,1].set_title('Tint Box Plot')
    axes[1,1].set_ylabel('Tint')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'target_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature correlation with targets
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_with_targets = df[numeric_cols].corr()[['Temperature', 'Tint']].drop(['Temperature', 'Tint'])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_with_targets, annot=True, cmap='RdBu_r', center=0, ax=ax)
    ax.set_title('Feature Correlation with Target Variables')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_target_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Camera settings distribution
    camera_features = ['aperture', 'isoSpeedRating', 'shutterSpeed', 'focalLength']
    available_camera_features = [f for f in camera_features if f in df.columns]
    
    if available_camera_features:
        n_features = len(available_camera_features)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, feature in enumerate(available_camera_features[:4]):
            if i < len(axes):
                axes[i].hist(df[feature].dropna(), bins=30, alpha=0.7)
                axes[i].set_title(f'{feature} Distribution')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Frequency')
        
        # Hide unused subplots
        for i in range(len(available_camera_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'camera_settings.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"📊 Visualizations saved to: {output_dir}/")

def main():
    """Main preprocessing pipeline"""
    
    print("🚀 AFTERSHOOT WHITE BALANCE - DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Configuration
    data_dir = "data"
    
    # Step 1: Load and analyze data
    df, images_dir = load_and_analyze_data(data_dir)
    
    # Step 2: Feature analysis
    categorical_features, numerical_features, target_features, time_features = analyze_features(df)
    
    # Step 3: Image quality check
    image_stats = check_image_quality(df, images_dir, sample_size=50)
    
    # Step 4: Anomaly detection
    anomalies = detect_anomalies(df)
    
    # Step 5: Feature engineering
    df_engineered = create_features(df)
    
    # Step 6: Data preprocessing
    df_processed = preprocess_data(df_engineered, categorical_features, numerical_features)
    
    # Step 7: Create splits
    train_df, val_df, test_df = create_splits(df_processed)
    
    # Step 8: Create visualizations
    create_visualizations(df_processed)
    
    print(f"\n✅ PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"📁 Final dataset shape: {df_processed.shape}")
    print(f"🏷️  Original features: {df.shape[1]}")
    print(f"⚙️  Engineered features: {df_processed.shape[1] - df.shape[1]}")
    print(f"💾 Processed data saved to: outputs/preprocessed_data/")
    print(f"📊 Analysis saved to: outputs/preprocessing_analysis/")
    
    return df_processed, train_df, val_df, test_df

if __name__ == "__main__":
    main()