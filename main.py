import os
import cv2
import numpy as np

# Quick test
def test_pannuke_loading(pannuke_path):
    fold1_images = os.path.join(pannuke_path, "Fold 2", "images", "fold2", "images.npy")
    
    if os.path.exists(fold1_images):
        images = np.load(fold1_images)
        print(f"Successfully loaded: {images.shape}")
        print(f"Data type: {images.dtype}")
        print(f"Value range: [{images.min()}, {images.max()}]")
        
        # Save first few samples
        for i in range(3):
            img = images[i]
            
            # Convert float64 to uint8 properly
            if img.dtype == np.float64 or img.dtype == np.float32:
                if img.max() <= 1.0:
                    # Values in [0, 1] range
                    img = (img * 255.0).astype(np.uint8)
                else:
                    # Values in [0, 255] range but wrong dtype
                    img = np.clip(img, 0, 255).astype(np.uint8)
            
            print(f"Sample {i} - Shape: {img.shape}, dtype: {img.dtype}, range: [{img.min()}, {img.max()}]")
            
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"test_pannuke_{i}.png", img_bgr)
            
        print("✓ Test samples saved")
    else:
        print(f"❌ File not found: {fold1_images}")

# Run test
test_pannuke_loading("/root/CycleGAN/data/PanNuke")