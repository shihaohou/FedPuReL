#!/usr/bin/env python3
"""
Stanford Cars Dataset Download Script

This script automatically downloads and extracts the Stanford Cars dataset.
The Stanford Cars dataset contains 16,185 images of 196 classes of cars.

Dataset Information:
- Total images: 16,185
- Classes: 196 car classes (Make, Model, Year)
- Training images: 8,144
- Test images: 8,041
- Image format: JPEG
- Contains bounding box annotations

Official source: https://ai.stanford.edu/~jkrause/cars/car_dataset.html
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


def organize_stanford_cars_dataset(data_dir):
    """Organize the Stanford Cars dataset structure."""
    print("Organizing Stanford Cars dataset structure...")
    
    # Expected structure after extraction:
    # stanford_cars/
    # ├── cars_train/
    # │   ├── 00001.jpg
    # │   ├── 00002.jpg
    # │   └── ...
    # ├── cars_test/
    # │   ├── 00001.jpg
    # │   ├── 00002.jpg
    # │   └── ...
    # └── devkit/
    #     ├── cars_train_annos.mat
    #     ├── cars_test_annos_withlabels.mat
    #     └── cars_meta.mat
    
    stanford_cars_dir = os.path.join(data_dir, "stanford_cars")
    
    # Check if main directories exist
    train_dir = os.path.join(stanford_cars_dir, "cars_train")
    test_dir = os.path.join(stanford_cars_dir, "cars_test")
    devkit_dir = os.path.join(stanford_cars_dir, "devkit")
    
    if os.path.exists(train_dir) and os.path.exists(test_dir) and os.path.exists(devkit_dir):
        print("✓ Stanford Cars dataset structure is correct")
        
        # Count training images
        train_images = [f for f in os.listdir(train_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
        print(f"✓ Training images: {len(train_images)}")
        
        # Count test images
        test_images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
        print(f"✓ Test images: {len(test_images)}")
        
        print(f"✓ Total images: {len(train_images) + len(test_images)}")
        
        # Check for required annotation files
        required_files = [
            "cars_train_annos.mat",
            "cars_test_annos_withlabels.mat",
            "cars_meta.mat"
        ]
        for file in required_files:
            file_path = os.path.join(devkit_dir, file)
            if os.path.exists(file_path):
                print(f"✓ Found {file}")
            else:
                print(f"✗ Missing {file}")
        
        return True
    else:
        print("✗ Stanford Cars dataset structure is incorrect")
        return False


def download_from_mirrors(urls, filename, chunk_size=8192):
    """Try downloading from a list of mirror URLs sequentially."""
    last_error = None
    for url in urls:
        try:
            print(f"Trying {url} ...")
            download_file(url, filename, chunk_size=chunk_size)
            return True
        except requests.HTTPError as e:
            print(f"✗ HTTP error for {url}: {e}")
            last_error = e
        except Exception as e:
            print(f"✗ Error for {url}: {e}")
            last_error = e
    print("✗ All mirrors failed.")
    if last_error:
        raise last_error
    return False


def main():
    """Main function to download and setup Stanford Cars dataset."""
    # Configuration
    data_dir = os.path.join(os.getcwd(), "DATA")
    stanford_cars_dir = os.path.join(data_dir, "stanford_cars")
    
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    # Download URLs (use working mirrors first, then legacy links)
    train_urls = [
        "http://imagenet.stanford.edu/internal/car196/cars_train.tgz",
        "https://ai.stanford.edu/~jkrause/car196/cars_train.tgz",
    ]
    test_urls = [
        "http://imagenet.stanford.edu/internal/car196/cars_test.tgz",
        "https://ai.stanford.edu/~jkrause/car196/cars_test.tgz",
    ]
    devkit_urls = [
        "https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz",
    ]
    test_labels_urls = [
        "http://imagenet.stanford.edu/internal/car196/cars_test_annos_withlabels.mat",
        "https://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat",
    ]
    
    # File paths
    train_tar = os.path.join(data_dir, "cars_train.tgz")
    test_tar = os.path.join(data_dir, "cars_test.tgz")
    devkit_tar = os.path.join(data_dir, "car_devkit.tgz")
    test_labels_file = os.path.join(data_dir, "cars_test_annos_withlabels.mat")
    
    try:
        # Download training images
        if not os.path.exists(train_tar):
            download_from_mirrors(train_urls, train_tar)
        else:
            print(f"✓ Training images file already exists: {train_tar}")
        
        # Verify training images file (expected ~1700-1900MB)
        if not verify_file_integrity(train_tar, expected_size_mb=1800):
            print("Re-downloading training images...")
            os.remove(train_tar)
            download_from_mirrors(train_urls, train_tar)
        
        # Download test images
        if not os.path.exists(test_tar):
            download_from_mirrors(test_urls, test_tar)
        else:
            print(f"✓ Test images file already exists: {test_tar}")
        
        # Verify test images file (expected ~1700-1900MB)
        if not verify_file_integrity(test_tar, expected_size_mb=1800):
            print("Re-downloading test images...")
            os.remove(test_tar)
            download_from_mirrors(test_urls, test_tar)
        
        # Download devkit
        if not os.path.exists(devkit_tar):
            download_from_mirrors(devkit_urls, devkit_tar)
        else:
            print(f"✓ Devkit file already exists: {devkit_tar}")
        
        # Verify devkit file (expected ~1MB)
        if not verify_file_integrity(devkit_tar, expected_size_mb=1):
            print("Re-downloading devkit...")
            os.remove(devkit_tar)
            download_from_mirrors(devkit_urls, devkit_tar)
        
        # Download test labels
        if not os.path.exists(test_labels_file):
            download_from_mirrors(test_labels_urls, test_labels_file)
        else:
            print(f"✓ Test labels file already exists: {test_labels_file}")

        # Create stanford_cars directory
        os.makedirs(stanford_cars_dir, exist_ok=True)
        
        # Extract files
        print("\nExtracting Stanford Cars dataset...")
        
        # Extract training images
        if not os.path.exists(os.path.join(stanford_cars_dir, "cars_train")):
            extract_tar_file(train_tar, stanford_cars_dir)
        else:
            print("✓ Training images already extracted")
        
        # Extract test images
        if not os.path.exists(os.path.join(stanford_cars_dir, "cars_test")):
            extract_tar_file(test_tar, stanford_cars_dir)
        else:
            print("✓ Test images already extracted")
        
        # Extract devkit
        if not os.path.exists(os.path.join(stanford_cars_dir, "devkit")):
            extract_tar_file(devkit_tar, stanford_cars_dir)
        else:
            print("✓ Devkit already extracted")
        
        # Move test labels to devkit directory
        devkit_dir = os.path.join(stanford_cars_dir, "devkit")
        test_labels_dest = os.path.join(devkit_dir, "cars_test_annos_withlabels.mat")
        if not os.path.exists(test_labels_dest):
            import shutil
            shutil.move(test_labels_file, test_labels_dest)
            print("✓ Moved test labels to devkit directory")
        
        # Organize and verify dataset
        if organize_stanford_cars_dataset(data_dir):
            print("\n🎉 Stanford Cars dataset setup completed successfully!")
            print(f"Dataset location: {stanford_cars_dir}")
            print("\nYou can now run the federated learning training:")
            print("python main.py --config configs/stanford_cars_LT.yaml")
        else:
            print("\n❌ Dataset setup failed. Please check the extracted files.")
            return 1
        
        # Clean up tar files (optional)
        cleanup = input("\nDelete downloaded tar files? (y/N): ").lower().strip()
        if cleanup == 'y':
            for tar_file in [train_tar, test_tar, devkit_tar]:
                if os.path.exists(tar_file):
                    os.remove(tar_file)
                    print(f"✓ Deleted {tar_file}")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error during download/extraction: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())