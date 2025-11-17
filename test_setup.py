#!/usr/bin/env python3
"""
Quick test script to verify the complete setup
"""

import sys
import os
sys.path.append('src')

def test_imports():
    """Test all critical imports"""
    try:
        import torch
        import torchvision
        import pandas as pd
        import numpy as np
        import cv2
        import timm
        from configs.model_configs import MODEL_CONFIGS
        print("✅ All critical imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_data():
    """Test data availability and structure"""
    try:
        import pandas as pd
        
        # Check files exist
        required_files = [
            'data/Train/sliders.csv',
            'data/Train/images',
            'data/Validation/sliders_inputs.csv',
            'data/Validation/images'
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                print(f"❌ Missing: {file_path}")
                return False
        
        # Test CSV loading
        train_df = pd.read_csv('data/Train/sliders.csv')
        val_df = pd.read_csv('data/Validation/sliders_inputs.csv')
        
        print(f"✅ Data structure verified")
        print(f"   Training samples: {len(train_df)}")
        print(f"   Validation samples: {len(val_df)}")
        print(f"   Training columns: {len(train_df.columns)}")
        
        # Check for required columns
        required_cols = ['id_global', 'Temperature', 'Tint']
        missing_cols = [col for col in required_cols if col not in train_df.columns]
        if missing_cols:
            print(f"❌ Missing columns in training data: {missing_cols}")
            return False
            
        return True
    except Exception as e:
        print(f"❌ Data test error: {e}")
        return False

def test_model_creation():
    """Test model creation"""
    try:
        import torch
        import timm
        from configs.model_configs import get_efficientnet_config
        
        # Test basic model creation
        model = timm.create_model('efficientnet_b0', pretrained=False)
        
        # Test config loading
        config = get_efficientnet_config()
        
        # Test tensor operations
        dummy_input = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            output = model(dummy_input)
        
        print("✅ Model creation and inference test successful")
        print(f"   Model output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"❌ Model test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Aftershoot White Balance Prediction Setup")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Data", test_data),
        ("Model Creation", test_model_creation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} test failed")
    
    print(f"\n🎯 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("✅ All tests passed! Your setup is ready for training.")
        print("\nNext steps:")
        print("1. Run EDA: python main.py --eda --config efficientnet")
        print("2. Start training: python main.py --config efficientnet --epochs 10")
    else:
        print("❌ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()