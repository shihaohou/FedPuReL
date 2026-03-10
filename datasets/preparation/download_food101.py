#!/usr/bin/env python3
"""
Food-101 Dataset Download Script

This script automatically downloads and extracts the Food-101 dataset.
The Food-101 dataset consists of 101 food categories with 101,000 images.
For each class, 250 manually reviewed test images and 750 training images are provided.

Dataset Information:
- Total images: 101,000
- Classes: 101 food categories
- Training images per class: 750
- Test images per class: 250
- Image format: JPEG
- Maximum side length: 512 pixels

Official source: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
"""

import os
import sys
import requests
import tarfile
from pathlib import Path
from tqdm import tqdm
import hashlib


def download_file(url, filename, chunk_size=8192):
    """Download a file with progress bar."""
    print(f"Downloading {filename}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                file.write(chunk)
                pbar.update(len(chunk))
    
    print(f"✓ Downloaded {filename}")


def verify_file_integrity(filepath, expected_size_mb=None):
    """Verify file integrity by checking size."""
    if not os.path.exists(filepath):
        return False
    
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"File size: {file_size_mb:.1f} MB")
    
    if expected_size_mb and abs(file_size_mb - expected_size_mb) > 50:  # Allow 50MB tolerance
        print(f"Warning: File size differs significantly from expected {expected_size_mb} MB")
        return False
    
    return True


def extract_tar_file(tar_path, extract_to):
    """Extract tar file with progress bar."""
    print(f"Extracting {tar_path}...")
    
    with tarfile.open(tar_path, 'r') as tar:
        members = tar.getmembers()
        
        with tqdm(total=len(members), desc="Extracting") as pbar:
            for member in members:
                tar.extract(member, extract_to)
                pbar.update(1)
    
    print(f"✓ Extracted to {extract_to}")


def organize_food101_dataset(data_dir):
    """Organize the Food-101 dataset structure."""
    food101_dir = os.path.join(data_dir, "food-101")
    
    if not os.path.exists(food101_dir):
        print("Error: food-101 directory not found after extraction")
        return False
    
    # Check if images directory exists
    images_dir = os.path.join(food101_dir, "images")
    if not os.path.exists(images_dir):
        print("Error: images directory not found in food-101")
        return False
    
    # Count classes and images
    classes = [d for d in os.listdir(images_dir) 
               if os.path.isdir(os.path.join(images_dir, d)) and not d.startswith('.')]
    
    total_images = 0
    for class_name in classes:
        class_dir = os.path.join(images_dir, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_images += len(images)
    
    print(f"✓ Dataset organized successfully!")
    print(f"  - Classes: {len(classes)}")
    print(f"  - Total images: {total_images}")
    print(f"  - Images directory: {images_dir}")
    
    return True


def main():
    """Main function to download and setup Food-101 dataset."""
    # Create DATA directory
    data_dir = os.path.join(os.getcwd(), "DATA")
    os.makedirs(data_dir, exist_ok=True)
    print(f"Created DATA directory: {data_dir}")
    
    # Food-101 dataset URL
    dataset_url = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    tar_filename = os.path.join(data_dir, "food-101.tar.gz")
    
    # Check if dataset already exists
    food101_dir = os.path.join(data_dir, "food-101")
    if os.path.exists(food101_dir):
        images_dir = os.path.join(food101_dir, "images")
        if os.path.exists(images_dir):
            print("Food-101 dataset already exists!")
            if organize_food101_dataset(data_dir):
                print("Dataset verification completed.")
                return
    
    try:
        # Download the dataset
        if not os.path.exists(tar_filename):
            download_file(dataset_url, tar_filename)
        else:
            print(f"Archive already exists: {tar_filename}")
        
        # Verify file integrity
        if not verify_file_integrity(tar_filename, expected_size_mb=4800):  # ~4.7GB expected
            print("File integrity check failed. Please re-download.")
            return
        
        # Extract the dataset
        extract_tar_file(tar_filename, data_dir)
        
        # Organize and verify dataset
        if organize_food101_dataset(data_dir):
            print("\n🎉 Food-101 dataset setup completed successfully!")
            print(f"Dataset location: {food101_dir}")
            print("\nYou can now run the federated learning script:")
            print("python federated_main.py --config-file configs/datasets/food101_LT.yaml")
        
        # Clean up tar file to save space
        if os.path.exists(tar_filename):
            os.remove(tar_filename)
            print(f"✓ Cleaned up archive file: {tar_filename}")
    
    except requests.RequestException as e:
        print(f"Download error: {e}")
        sys.exit(1)
    except tarfile.TarError as e:
        print(f"Extraction error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()