#!/usr/bin/env python3
"""
Quick verification script to test the new 90/10 data split
"""

import os
import sys

# Add src to path for imports
sys.path.append('src')

from configs.model_configs import MODEL_CONFIGS
from src.data.dataset import create_data_loaders

def test_data_split():
    """Test the new data split configuration"""
    
    print("🔍 Testing new 90/10 data split configuration...")
    print("=" * 50)
    
    # Load lightweight config
    config = MODEL_CONFIGS['lightweight']()
    
    # Update paths to use filtered CSV
    config.data.train_csv_path = os.path.join('data', 'Train', 'sliders_filtered.csv')
    
    print(f"📊 Data split configuration:")
    print(f"   train_val_split: {config.data.train_val_split}")
    print(f"   Expected training: {int(907 * config.data.train_val_split)} images ({config.data.train_val_split * 100:.0f}%)")
    print(f"   Expected validation: {int(907 * (1 - config.data.train_val_split))} images ({(1 - config.data.train_val_split) * 100:.0f}%)")
    print()
    
    try:
        # Create data loaders to test the split
        print("🚀 Creating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(config)
        
        print("✅ Data loaders created successfully!")
        print()
        print("📈 ACTUAL DATA SPLIT RESULTS:")
        print(f"   Training samples: {len(train_loader.dataset)}")
        print(f"   Validation samples: {len(val_loader.dataset)}")
        print(f"   Test samples: {len(test_loader.dataset)}")
        print()
        
        # Calculate improvement
        previous_training = 543
        new_training = len(train_loader.dataset)
        improvement = new_training - previous_training
        
        print("📊 IMPROVEMENT ANALYSIS:")
        print(f"   Previous training samples: {previous_training}")
        print(f"   New training samples: {new_training}")
        print(f"   Additional training data: +{improvement} images ({improvement/previous_training*100:.1f}% increase)")
        print()
        
        print("🎯 SUMMARY:")
        if improvement > 200:
            print(f"   ✅ Excellent! You've gained {improvement} more training images")
            print("   ✅ This should significantly improve model performance")
        elif improvement > 100:
            print(f"   ✅ Good! You've gained {improvement} more training images") 
            print("   ✅ This should improve model performance")
        else:
            print(f"   ⚠️  Only gained {improvement} training images")
            
        print("   ✅ Ready for training with improved data split!")
        
    except Exception as e:
        print(f"❌ Error testing data split: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_data_split()