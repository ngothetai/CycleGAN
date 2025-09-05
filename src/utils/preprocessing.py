# prepare_histology_dataset.py
import os
import cv2
import numpy as np
from pathlib import Path
import random
from sklearn.model_selection import train_test_split

def extract_patches(image, patch_size=256, stride=128, min_tissue_ratio=0.3):
    """Extract patches và filter out background"""
    h, w = image.shape[:2]
    patches = []
    
    for y in range(0, h-patch_size+1, stride):
        for x in range(0, w-patch_size+1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            
            # Filter out mostly background patches
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            tissue_pixels = np.sum(gray < 200)  # Non-white pixels
            tissue_ratio = tissue_pixels / (patch_size * patch_size)
            
            if tissue_ratio > min_tissue_ratio:
                patches.append(patch)
    
    return patches

def process_deeplif_dataset(deeplif_path, output_dir):
    """Process DeepLIIF IHC images"""
    ihc_dir = os.path.join(deeplif_path, "DeepLIIF_Training_Set")
    patches = []
    
    for filename in os.listdir(ihc_dir):
        if filename.lower().endswith(('.png', '.tif', '.jpg')):
            img_path = os.path.join(ihc_dir, filename)
            img = cv2.imread(img_path)
            
            if img is not None:
                img_patches = extract_patches(img)
                patches.extend(img_patches)
    
    # Split train/test
    train_patches, test_patches = train_test_split(patches, test_size=0.2, random_state=42)
    
    # Save train patches
    os.makedirs(f"{output_dir}/trainA", exist_ok=True)
    for i, patch in enumerate(train_patches):
        cv2.imwrite(f"{output_dir}/trainA/ihc_{i:06d}.png", patch)
    
    # Save test patches  
    os.makedirs(f"{output_dir}/testA", exist_ok=True)
    for i, patch in enumerate(test_patches):
        cv2.imwrite(f"{output_dir}/testA/ihc_{i:06d}.png", patch)
    
    print(f"DeepLIIF: {len(train_patches)} train, {len(test_patches)} test patches")

def process_monuseg_dataset(monuseg_path, output_dir):
    """Process MoNuSeg H&E images"""
    tissue_dir = os.path.join(monuseg_path, "Tissue Images")
    patches = []
    
    for filename in os.listdir(tissue_dir):
        if filename.lower().endswith(('.tif', '.png', '.jpg')):
            img_path = os.path.join(tissue_dir, filename)
            img = cv2.imread(img_path)
            
            if img is not None:
                img_patches = extract_patches(img)
                patches.extend(img_patches)
    
    return patches

def process_pannuke_dataset(pannuke_path):
    """Process PanNuke H&E images from .npy files"""
    patches = []
    
    print("Processing PanNuke dataset...")
    
    # Detect available folds
    available_folds = []
    for item in os.listdir(pannuke_path):
        item_path = os.path.join(pannuke_path, item)
        if os.path.isdir(os.path.join(item_path, "images")) and ("fold" in item.lower() or "Fold" in item):
            available_folds.append(item)
    
    available_folds.sort()
    print(f"Found folds: {available_folds}")
    
    for fold_name in available_folds:
        fold_dir = os.path.join(pannuke_path, fold_name)
        images_file = os.path.join(fold_dir, "images", "images.npy")
        print(f"Processing {fold_name}...")
        
        try:
            # Load numpy arrays
            images = np.load(images_file)
            print(f"Loaded {len(images)} images from {fold_name}")
            print(f"Image dtype: {images.dtype}, shape: {images.shape}")
            
            fold_patches = 0
            
            # Process each image
            for i, img_array in enumerate(images):
                # Fix data type issue
                if img_array.dtype in [np.float64, np.float32]:
                    if img_array.max() <= 1.0:
                        # Normalize [0,1] to [0,255]
                        img_array = (img_array * 255.0).astype(np.uint8)
                    else:
                        # Already [0,255] range, just convert type
                        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                elif img_array.dtype != np.uint8:
                    img_array = img_array.astype(np.uint8)
                
                # Convert RGB to BGR (OpenCV format)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # Quality check - filter out mostly background
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                tissue_pixels = np.sum(gray < 200)
                tissue_ratio = tissue_pixels / (256 * 256)
                
                if tissue_ratio > 0.3:  # Keep patches with >30% tissue
                    patches.append(img_bgr)
                    fold_patches += 1
                
                # Debug: save first few samples
                if i < 3 and fold_name == available_folds[0]:
                    cv2.imwrite(f"debug_pannuke_{fold_name}_{i}.png", img_bgr)
            
            print(f"Extracted {fold_patches} good patches from {fold_name}")
            
        except Exception as e:
            print(f"Error processing {fold_name}: {e}")
    
    print(f"Total PanNuke patches: {len(patches)}")
    return patches

def main():
    # Paths to your downloaded datasets
    DEEPLIF_PATH = "data/DeepLIIF"
    MONUSEG_PATH = "data/MoNuSeg"  
    PANNUKE_PATH = "data/PanNuke"
    OUTPUT_DIR = "datasets/histology_stain"
    
    # Process IHC domain (DeepLIIF)
    print("Processing DeepLIIF (IHC domain)...")
    process_deeplif_dataset(DEEPLIF_PATH, OUTPUT_DIR)
    
    # Process H&E domain (MoNuSeg + PanNuke)
    print("Processing MoNuSeg (H&E domain)...")
    he_patches_monuseg = process_monuseg_dataset(MONUSEG_PATH, OUTPUT_DIR)
    
    print("Processing PanNuke (H&E domain)...")  
    he_patches_pannuke = process_pannuke_dataset(PANNUKE_PATH)
    
    # Combine H&E patches
    all_he_patches = he_patches_monuseg + he_patches_pannuke
    random.shuffle(all_he_patches)
    
    # Split H&E train/test
    train_he, test_he = train_test_split(all_he_patches, test_size=0.2, random_state=42)
    
    # Save H&E patches
    os.makedirs(f"{OUTPUT_DIR}/trainB", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/testB", exist_ok=True)
    
    for i, patch in enumerate(train_he):
        cv2.imwrite(f"{OUTPUT_DIR}/trainB/he_{i:06d}.png", patch)
        
    for i, patch in enumerate(test_he):
        cv2.imwrite(f"{OUTPUT_DIR}/testB/he_{i:06d}.png", patch)
    
    print(f"H&E: {len(train_he)} train, {len(test_he)} test patches")
    print("Dataset preparation completed!")

if __name__ == "__main__":
    main()