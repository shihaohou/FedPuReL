#!/usr/bin/env python3
"""
FGVC-Aircraft Dataset Download Script

This script automatically downloads and extracts the FGVC-Aircraft dataset.
The FGVC-Aircraft dataset contains images of aircraft, with 100 different aircraft model variants.

Dataset Information:
- Total images: 10,200
- Classes: 100 aircraft model variants
- Training images: 6,667
- Validation images: 3,333  
- Test images: 3,333
- Image format: JPEG
- Hierarchical labels: Manufacturer, Family, Variant

Official source: https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/
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
                
                print(f"✓ Extracted {tar_path}")
                return True
                
        except (tarfile.ReadError, tarfile.CompressionError):
            continue
    
    print(f"✗ Failed to extract {tar_path}")
    return False


def organize_aircraft_dataset(data_dir):
    """Organize and verify the FGVC-Aircraft dataset structure."""
    aircraft_dir = os.path.join(data_dir, "fgvc-aircraft-2013b")
    
    if not os.path.exists(aircraft_dir):
        print(f"✗ Aircraft directory not found: {aircraft_dir}")
        return False
    
    # Check for required directories and files
    required_items = [
        "data/images",
        "data/images_variant_train.txt",
        "data/images_variant_val.txt", 
        "data/images_variant_test.txt",
        "data/variants.txt"
    ]
    
    missing_items = []
    for item in required_items:
        item_path = os.path.join(aircraft_dir, item)
        if not os.path.exists(item_path):
            missing_items.append(item)
    
    if missing_items:
        print("✗ Missing required files/directories:")
        for item in missing_items:
            print(f"  - {item}")
        return False
    
    # Count images
    images_dir = os.path.join(aircraft_dir, "data", "images")
    if os.path.exists(images_dir):
        image_count = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"✓ Found {image_count} images")
        
        if image_count < 10000:  # Expected ~10,200 images
            print(f"⚠ Warning: Expected ~10,200 images, found {image_count}")
    
    # Count classes from variants.txt
    variants_file = os.path.join(aircraft_dir, "data", "variants.txt")
    if os.path.exists(variants_file):
        with open(variants_file, 'r') as f:
            variants = [line.strip() for line in f if line.strip()]
        print(f"✓ Found {len(variants)} aircraft variants")
        
        if len(variants) != 100:
            print(f"⚠ Warning: Expected 100 variants, found {len(variants)}")
    
    print("✓ FGVC-Aircraft dataset structure verified")
    return True


def main():
    """Main function to download and setup FGVC-Aircraft dataset."""
    # Configuration
    data_dir = os.path.join(os.getcwd(), "DATA")
    aircraft_dir = os.path.join(data_dir, "fgvc-aircraft-2013b")
    
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    # Download URL
    dataset_url = "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"
    
    # File path
    dataset_tar = os.path.join(data_dir, "fgvc-aircraft-2013b.tar.gz")
    
    try:
        # Download dataset
        if not os.path.exists(dataset_tar):
            download_file(dataset_url, dataset_tar)
        else:
            print(f"✓ Dataset file already exists: {dataset_tar}")
        
        # Verify dataset file (expected ~2.6GB)
        if not verify_file_integrity(dataset_tar, expected_size_mb=2600):
            print("Re-downloading dataset...")
            os.remove(dataset_tar)
            download_file(dataset_url, dataset_tar)
        
        # Extract dataset
        print("\nExtracting FGVC-Aircraft dataset...")
        
        if not os.path.exists(aircraft_dir):
            if not extract_tar_file(dataset_tar, data_dir):
                print("✗ Failed to extract dataset")
                return 1
        else:
            print("✓ Dataset already extracted")
        
        # Organize and verify dataset
        if organize_aircraft_dataset(data_dir):
            print("\n🎉 FGVC-Aircraft dataset setup completed successfully!")
            print(f"Dataset location: {aircraft_dir}")
            print("\nDataset structure:")
            print("├── data/")
            print("│   ├── images/                    # All aircraft images")
            print("│   ├── images_variant_train.txt   # Training split")
            print("│   ├── images_variant_val.txt     # Validation split")
            print("│   ├── images_variant_test.txt    # Test split")
            print("│   └── variants.txt               # List of 100 variants")
            print("\nYou can now run the federated learning training:")
            print("python federated_main.py --config configs/datasets/aircraft_LT.yaml")
        else:
            print("\n❌ Dataset setup failed. Please check the extracted files.")
            return 1
        
        # Clean up tar file (optional)
        cleanup = input("\nDelete downloaded tar file? (y/N): ").lower().strip()
        if cleanup == 'y':
            if os.path.exists(dataset_tar):
                os.remove(dataset_tar)
                print(f"✓ Deleted {dataset_tar}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during download/extraction: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Ensure you have sufficient disk space (~3GB)")
        print("3. Try running the script again")
        print("4. If the problem persists, you can manually download from:")
        print("   https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/")
        return 1


if __name__ == "__main__":
    print("FGVC-Aircraft Dataset Downloader")
    print("=" * 40)
    print("This script will download the FGVC-Aircraft dataset (~2.6GB)")
    print("The dataset contains 10,200 images of 100 aircraft model variants")
    print()
    
    # Ask for confirmation
    confirm = input("Continue with download? (y/N): ").lower().strip()
    if confirm != 'y':
        print("Download cancelled.")
        sys.exit(0)
    
    sys.exit(main())