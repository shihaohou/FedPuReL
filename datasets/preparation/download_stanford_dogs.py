#!/usr/bin/env python3
"""
Stanford Dogs Dataset Download Script

This script automatically downloads and extracts the Stanford Dogs dataset.
The Stanford Dogs dataset contains images of 120 breeds of dogs from around the world.

Dataset Information:
- Total images: 20,580
- Classes: 120 dog breeds
- Training images: ~12,000
- Test images: ~8,580
- Image format: JPEG
- Contains bounding box annotations

Official source: http://vision.stanford.edu/aditya86/ImageNetDogs/
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
    """Verify downloaded file integrity."""
    if not os.path.exists(filepath):
        print(f"✗ File not found: {filepath}")
        return False
    
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")
    
    if expected_size_mb and abs(file_size_mb - expected_size_mb) > 50:  # Allow 50MB tolerance
        print(f"✗ File size mismatch. Expected ~{expected_size_mb}MB, got {file_size_mb:.2f}MB")
        print("Please re-download the file.")
        return False
    
    print("✓ File integrity check passed")
    return True


def extract_tar_file(tar_path, extract_to):
    """Extract tar file with progress."""
    print(f"Extracting {tar_path}...")
    
    # Try different compression formats
    formats_to_try = ['r:gz', 'r:bz2', 'r:xz', 'r']
    
    for fmt in formats_to_try:
        try:
            with tarfile.open(tar_path, fmt) as tar:
                members = tar.getmembers()
                with tqdm(total=len(members), desc="Extracting") as pbar:
                    for member in members:
                        tar.extract(member, extract_to)
                        pbar.update(1)
                print(f"✓ Extracted to {extract_to}")
                return
        except (tarfile.ReadError, tarfile.CompressionError):
            continue
    
    # If all formats fail, raise an error
    raise Exception(f"Could not extract {tar_path} - unsupported format or corrupted file")


def organize_stanford_dogs_dataset(data_dir):
    """Organize the Stanford Dogs dataset structure."""
    print("Organizing Stanford Dogs dataset structure...")
    
    # Expected structure after extraction:
    # stanford_dogs/
    # ├── Images/
    # │   ├── n02085620-Chihuahua/
    # │   ├── n02085782-Japanese_spaniel/
    # │   └── ...
    # ├── Annotation/
    # │   ├── n02085620-Chihuahua/
    # │   └── ...
    # ├── train_list.mat
    # ├── test_list.mat
    # └── file_list.mat
    
    stanford_dogs_dir = os.path.join(data_dir, "stanford_dogs")
    
    # Check if main directories exist
    images_dir = os.path.join(stanford_dogs_dir, "Images")
    annotation_dir = os.path.join(stanford_dogs_dir, "Annotation")
    
    if os.path.exists(images_dir) and os.path.exists(annotation_dir):
        print("✓ Stanford Dogs dataset structure is correct")
        
        # Count breed directories
        breed_dirs = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]
        print(f"✓ Found {len(breed_dirs)} dog breed directories")
        
        # Count total images
        total_images = 0
        for breed_dir in breed_dirs:
            breed_path = os.path.join(images_dir, breed_dir)
            images = [f for f in os.listdir(breed_path) if f.lower().endswith(('.jpg', '.jpeg'))]
            total_images += len(images)
        
        print(f"✓ Total images: {total_images}")
        
        # Check for required .mat files
        required_files = ["train_list.mat", "test_list.mat", "file_list.mat"]
        for file in required_files:
            file_path = os.path.join(stanford_dogs_dir, file)
            if os.path.exists(file_path):
                print(f"✓ Found {file}")
            else:
                print(f"✗ Missing {file}")
        
        return True
    else:
        print("✗ Stanford Dogs dataset structure is incorrect")
        return False


def main():
    """Main function to download and setup Stanford Dogs dataset."""
    # Configuration
    data_dir = os.path.join(os.getcwd(), "DATA")
    stanford_dogs_dir = os.path.join(data_dir, "stanford_dogs")
    
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    # Download URLs
    images_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
    annotation_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar"
    lists_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar"
    
    # File paths
    images_tar = os.path.join(data_dir, "images.tar")
    annotation_tar = os.path.join(data_dir, "annotation.tar")
    lists_tar = os.path.join(data_dir, "lists.tar")
    
    try:
        # Download images
        if not os.path.exists(images_tar):
            download_file(images_url, images_tar)
        else:
            print(f"✓ Images file already exists: {images_tar}")
        
        # Verify images file (expected ~750MB)
        if not verify_file_integrity(images_tar, expected_size_mb=750):
            print("Re-downloading images...")
            os.remove(images_tar)
            download_file(images_url, images_tar)
        
        # Download annotations
        if not os.path.exists(annotation_tar):
            download_file(annotation_url, annotation_tar)
        else:
            print(f"✓ Annotations file already exists: {annotation_tar}")
        
        # Verify annotations file (expected ~20MB)
        if not verify_file_integrity(annotation_tar, expected_size_mb=20):
            print("Re-downloading annotations...")
            os.remove(annotation_tar)
            download_file(annotation_url, annotation_tar)
        
        # Download lists
        if not os.path.exists(lists_tar):
            download_file(lists_url, lists_tar)
        else:
            print(f"✓ Lists file already exists: {lists_tar}")
        
        # Verify lists file (expected ~1MB)
        if not verify_file_integrity(lists_tar, expected_size_mb=1):
            print("Re-downloading lists...")
            os.remove(lists_tar)
            download_file(lists_url, lists_tar)
        
        # Create stanford_dogs directory
        os.makedirs(stanford_dogs_dir, exist_ok=True)
        
        # Extract files
        print("\nExtracting Stanford Dogs dataset...")
        
        # Extract images
        if not os.path.exists(os.path.join(stanford_dogs_dir, "Images")):
            extract_tar_file(images_tar, stanford_dogs_dir)
        else:
            print("✓ Images already extracted")
        
        # Extract annotations
        if not os.path.exists(os.path.join(stanford_dogs_dir, "Annotation")):
            extract_tar_file(annotation_tar, stanford_dogs_dir)
        else:
            print("✓ Annotations already extracted")
        
        # Extract lists
        if not os.path.exists(os.path.join(stanford_dogs_dir, "train_list.mat")):
            extract_tar_file(lists_tar, stanford_dogs_dir)
        else:
            print("✓ Lists already extracted")
        
        # Organize and verify dataset
        if organize_stanford_dogs_dataset(data_dir):
            print("\n🎉 Stanford Dogs dataset setup completed successfully!")
            print(f"Dataset location: {stanford_dogs_dir}")
            print("\nYou can now run the federated learning training:")
            print("python main.py --config configs/stanford_dogs_LT.yaml")
        else:
            print("\n❌ Dataset setup failed. Please check the extracted files.")
            return 1
        
        # Clean up tar files (optional)
        cleanup = input("\nDelete downloaded tar files? (y/N): ").lower().strip()
        if cleanup == 'y':
            for tar_file in [images_tar, annotation_tar, lists_tar]:
                if os.path.exists(tar_file):
                    os.remove(tar_file)
                    print(f"✓ Deleted {tar_file}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during download/extraction: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())