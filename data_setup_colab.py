#!/usr/bin/env python3
"""
Data preparation helper for Google Colab
Handles downloading, uploading, and organizing Aftershoot WB prediction data
"""

import os
import shutil
import zipfile
import pandas as pd
from pathlib import Path
from google.colab import files, drive

def mount_drive():
    """Mount Google Drive"""
    print("📂 Mounting Google Drive...")
    drive.mount('/content/drive', force_remount=True)
    print("✅ Google Drive mounted!")
    return "/content/drive/MyDrive"

def upload_data_manual():
    """Upload data files manually through Colab interface"""
    print("📤 Manual data upload process:")
    print("1. Use the Files panel (📁) on the left")
    print("2. Create folder structure:")
    print("   /content/aftershoot_wb_prediction/data/")
    print("   ├── Train/")
    print("   │   ├── images/          # Upload TIFF images here")
    print("   │   └── sliders.csv      # Upload training CSV here")
    print("   ├── Validation/")
    print("   │   ├── images/")
    print("   │   └── sliders.csv")
    print("   └── Test/")
    print("       ├── images/")
    print("       └── sliders.csv")
    print("\n3. Alternatively, upload ZIP files and extract them")
    
    # Option to upload ZIP files
    print("\n📦 Or upload ZIP files containing your dataset:")
    uploaded = files.upload()
    
    for filename in uploaded.keys():
        if filename.endswith('.zip'):
            print(f"📦 Extracting {filename}...")
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall('/content/aftershoot_wb_prediction/data/')
            print(f"✅ Extracted {filename}")
            os.remove(filename)  # Clean up

def setup_data_from_drive(drive_path="/content/drive/MyDrive/aftershoot_data"):
    """Setup data from Google Drive"""
    print(f"📂 Setting up data from Google Drive: {drive_path}")
    
    if not os.path.exists(drive_path):
        print(f"❌ Data path not found: {drive_path}")
        print("Please upload your data to Google Drive first!")
        return False
    
    # Create symbolic link or copy data
    data_link = "/content/aftershoot_wb_prediction/data"
    
    if os.path.exists(data_link):
        os.remove(data_link)
    
    os.symlink(drive_path, data_link)
    print(f"✅ Data linked from {drive_path}")
    
    return True

def verify_data_structure():
    """Verify the data structure is correct"""
    print("🔍 Verifying data structure...")
    
    data_path = "/content/aftershoot_wb_prediction/data"
    required_files = [
        "Train/sliders.csv",
        "Validation/sliders.csv", 
        "Test/sliders.csv"
    ]
    
    required_dirs = [
        "Train/images",
        "Validation/images",
        "Test/images"
    ]
    
    issues = []
    
    for file_path in required_files:
        full_path = os.path.join(data_path, file_path)
        if os.path.exists(full_path):
            df = pd.read_csv(full_path)
            print(f"✅ {file_path}: {len(df)} samples")
        else:
            issues.append(f"❌ Missing: {file_path}")
    
    for dir_path in required_dirs:
        full_path = os.path.join(data_path, dir_path)
        if os.path.exists(full_path):
            image_count = len([f for f in os.listdir(full_path) if f.endswith(('.tiff', '.tif'))])
            print(f"✅ {dir_path}: {image_count} images")
        else:
            issues.append(f"❌ Missing: {dir_path}")
    
    if issues:
        print("\n⚠️ Data structure issues:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("\n🎉 Data structure looks good!")
        return True

def create_sample_data():
    """Create sample data for testing (when real data is not available)"""
    print("🎨 Creating sample data for testing...")
    
    import numpy as np
    from PIL import Image
    
    # Create directories
    os.makedirs("/content/aftershoot_wb_prediction/data/Train/images", exist_ok=True)
    os.makedirs("/content/aftershoot_wb_prediction/data/Validation/images", exist_ok=True)
    os.makedirs("/content/aftershoot_wb_prediction/data/Test/images", exist_ok=True)
    
    # Generate sample metadata
    def generate_sample_csv(n_samples, split_name):
        np.random.seed(42 if split_name == 'Train' else 123 if split_name == 'Validation' else 456)
        
        data = {
            'id_global': [f"{split_name.lower()}_{i:04d}" for i in range(n_samples)],
            'copyCreationTime': ['2024-01-01T12:00:00.000'] * n_samples,
            'captureTime': ['2024-01-01T12:00:00.000'] * n_samples,
            'touchTime': ['2024-01-01T12:00:00.000'] * n_samples,
            'grayscale': [0] * n_samples,
            'aperture': np.random.uniform(1.4, 8.0, n_samples),
            'flashFired': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'focalLength': np.random.uniform(10, 200, n_samples),
            'isoSpeedRating': np.random.choice([100, 200, 400, 800, 1600, 3200], n_samples),
            'shutterSpeed': np.random.uniform(1/1000, 1/10, n_samples),
            'currTemp': np.random.uniform(2000, 10000, n_samples),
            'currTint': np.random.uniform(-50, 50, n_samples),
            'Temperature': np.random.uniform(2000, 10000, n_samples),
            'Tint': np.random.uniform(-90, 40, n_samples),
        }
        
        df = pd.DataFrame(data)
        csv_path = f"/content/aftershoot_wb_prediction/data/{split_name}/sliders.csv"
        df.to_csv(csv_path, index=False)
        
        # Create sample TIFF images
        for i in range(min(n_samples, 10)):  # Only create 10 sample images per split
            # Generate random image
            img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_path = f"/content/aftershoot_wb_prediction/data/{split_name}/images/{data['id_global'][i]}.tiff"
            img.save(img_path)
        
        print(f"✅ Created {split_name}: {n_samples} samples, 10 sample images")
    
    # Generate datasets
    generate_sample_csv(100, 'Train')
    generate_sample_csv(20, 'Validation')  
    generate_sample_csv(20, 'Test')
    
    print("🎉 Sample data created successfully!")
    print("⚠️ Note: This is synthetic data for testing only")

def main():
    """Main data setup function"""
    print("🚀 Aftershoot WB Prediction - Data Setup")
    print("=" * 50)
    
    # Create project structure
    os.makedirs("/content/aftershoot_wb_prediction", exist_ok=True)
    os.chdir("/content/aftershoot_wb_prediction")
    
    print("Choose your data setup method:")
    print("1. Use data from Google Drive")
    print("2. Upload data manually") 
    print("3. Create sample data for testing")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        drive_root = mount_drive()
        drive_path = input(f"Enter data path in Drive (default: {drive_root}/aftershoot_data): ").strip()
        if not drive_path:
            drive_path = f"{drive_root}/aftershoot_data"
        setup_data_from_drive(drive_path)
        
    elif choice == "2":
        upload_data_manual()
        
    elif choice == "3":
        create_sample_data()
        
    else:
        print("Invalid choice!")
        return
    
    # Verify data structure
    verify_data_structure()
    
    print("\n🎉 Data setup complete!")
    print("Next steps:")
    print("1. Upload your code files")
    print("2. Run EDA: !python main_colab.py --eda")
    print("3. Start training: !python main_colab.py --train")

if __name__ == "__main__":
    main()