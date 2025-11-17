#!/usr/bin/env python3
"""
View EDA results and key insights
"""

import pandas as pd
import os

def show_eda_insights():
    """Show key insights from the EDA analysis"""
    
    print("🔍 AFTERSHOOT WHITE BALANCE EDA INSIGHTS")
    print("=" * 50)
    
    # Load and analyze the data
    if os.path.exists('data/Train/sliders.csv'):
        df = pd.read_csv('data/Train/sliders.csv')
        
        print(f"\n📊 DATASET OVERVIEW")
        print(f"   Total samples: {len(df)}")
        print(f"   Features: {len(df.columns)}")
        print(f"   Missing values: {df.isnull().sum().sum()}")
        
        print(f"\n🎯 TARGET VARIABLES")
        print(f"   Temperature range: {df['Temperature'].min():.0f}K - {df['Temperature'].max():.0f}K")
        print(f"   Temperature mean: {df['Temperature'].mean():.1f}K ± {df['Temperature'].std():.1f}K")
        print(f"   Tint range: {df['Tint'].min():.1f} - {df['Tint'].max():.1f}")
        print(f"   Tint mean: {df['Tint'].mean():.2f} ± {df['Tint'].std():.2f}")
        
        # Temperature sensitivity analysis
        print(f"\n🌡️ TEMPERATURE SENSITIVITY ANALYSIS")
        df['temp_change'] = df['Temperature'] - df['currTemp']
        df['temp_change_abs'] = abs(df['temp_change'])
        
        # Define ranges
        low_temp_mask = df['currTemp'] < 3500
        mid_temp_mask = (df['currTemp'] >= 3500) & (df['currTemp'] < 6000)
        high_temp_mask = df['currTemp'] >= 6000
        
        print(f"   Low temp (< 3500K): Avg change = {df[low_temp_mask]['temp_change_abs'].mean():.0f}K")
        print(f"   Mid temp (3500-6000K): Avg change = {df[mid_temp_mask]['temp_change_abs'].mean():.0f}K") 
        print(f"   High temp (> 6000K): Avg change = {df[high_temp_mask]['temp_change_abs'].mean():.0f}K")
        
        print(f"\n📸 CAMERA & SETTINGS")
        print(f"   Unique cameras: {df['flashFired'].nunique()}")
        print(f"   Flash usage: {(df['flashFired'] == 1).sum()}/{len(df)} ({(df['flashFired'] == 1).mean()*100:.1f}%)")
        print(f"   ISO range: {df['isoSpeedRating'].min()} - {df['isoSpeedRating'].max()}")
        print(f"   Aperture range: f/{df['aperture'].min():.1f} - f/{df['aperture'].max():.1f}")
        
        print(f"\n🔗 FEATURE CORRELATIONS WITH TARGETS")
        # Select only numeric columns for correlation
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if 'Temperature' in numeric_cols and 'Tint' in numeric_cols:
            correlations_temp = df[numeric_cols].corr()['Temperature'].abs().sort_values(ascending=False)
            correlations_tint = df[numeric_cols].corr()['Tint'].abs().sort_values(ascending=False)
            
            print(f"   Strongest Temperature correlations:")
            for feat, corr in correlations_temp.head(5).items():
                if feat != 'Temperature':
                    print(f"     {feat}: {corr:.3f}")
            
            print(f"   Strongest Tint correlations:")
            for feat, corr in correlations_tint.head(5).items():
                if feat != 'Tint':
                    print(f"     {feat}: {corr:.3f}")
    
    # Check EDA visualizations
    eda_dir = 'outputs/eda'
    if os.path.exists(eda_dir):
        print(f"\n📈 GENERATED VISUALIZATIONS")
        viz_files = os.listdir(eda_dir)
        for viz_file in viz_files:
            print(f"   ✅ {viz_file}")
        print(f"\n   📁 View them in: {os.path.abspath(eda_dir)}")
    
    print(f"\n🚀 MODELING RECOMMENDATIONS")
    print(f"   1. Use temperature-aware loss weighting")
    print(f"   2. Include current WB (currTemp, currTint) as strong predictors")
    print(f"   3. Engineer flash/camera interaction features")
    print(f"   4. Apply consistency regularization")
    print(f"   5. Handle missing images gracefully")

if __name__ == "__main__":
    show_eda_insights()