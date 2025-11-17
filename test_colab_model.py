#!/usr/bin/env python3
"""
Comprehensive model testing script for Colab environment
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime

# Add src to path
sys.path.append('src')

def test_trained_model():
    """Test the trained model and generate comprehensive results"""
    print("🔬 COMPREHENSIVE MODEL TESTING")
    print("=" * 60)

    # Find the best model
    model_paths = [
        'outputs/optimized_90_split/checkpoints/best_model.pt',
        'outputs/checkpoints/best_model.pt',
        'outputs/complete_90_split/checkpoints/best_model.pt'
    ]

    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break

    if not model_path:
        print("❌ No trained model found! Train a model first.")
        return

    print(f"📁 Found model: {model_path}")

    # Test with Poetry (with proper UTF-8 encoding)
    try:
        import subprocess
        result = subprocess.run([
            'poetry', 'run', 'python', 'test_model.py'
        ], capture_output=True, text=True, cwd='.', encoding='utf-8', errors='replace')

        if result.returncode == 0:
            print("✅ Model testing completed successfully!")
            if result.stdout and result.stdout.strip():
                print(result.stdout)
            else:
                print("💡 Test completed but no output captured - check outputs/model_testing/ for results")
        else:
            print("❌ Model testing failed:")
            if result.stderr:
                print(result.stderr)
            else:
                print("No error details available")

    except Exception as e:
        print(f"❌ Error running test: {e}")
        print("💡 Try running test_model.py directly")

if __name__ == '__main__':
    test_trained_model()
