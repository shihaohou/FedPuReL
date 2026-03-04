#!/usr/bin/env python3
"""
Oxford Pets Dataset Downloader
自动下载并设置Oxford Pets数据集
"""

import os
import urllib.request
import tarfile
import shutil
from pathlib import Path

def download_file(url, filepath):
    """下载文件并显示进度"""
    print(f"正在下载: {url}")
    print(f"保存到: {filepath}")
    
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            print(f"\r下载进度: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='')
        else:
            print(f"\r已下载: {downloaded} bytes", end='')
    
    urllib.request.urlretrieve(url, filepath, reporthook=show_progress)
    print()  # 换行

def extract_tar_gz(filepath, extract_to):
    """解压tar.gz文件"""
    print(f"正在解压: {filepath}")
    with tarfile.open(filepath, 'r:gz') as tar:
        tar.extractall(extract_to)
    print(f"解压完成到: {extract_to}")

def setup_oxford_pets_dataset(data_root="DATA"):
    """
    设置Oxford Pets数据集
    
    Args:
        data_root: 数据集根目录，默认为"DATA"
    """
    # 创建目录结构
    data_root = Path(data_root)
    oxford_pets_dir = data_root / "oxford_pets"
    oxford_pets_dir.mkdir(parents=True, exist_ok=True)
    
    # 下载URLs
    images_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    annotations_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
    
    # 下载文件
    images_file = oxford_pets_dir / "images.tar.gz"
    annotations_file = oxford_pets_dir / "annotations.tar.gz"
    
    try:
        # 下载图像数据
        if not images_file.exists():
            download_file(images_url, images_file)
        else:
            print(f"图像文件已存在: {images_file}")
        
        # 下载标注数据
        if not annotations_file.exists():
            download_file(annotations_url, annotations_file)
        else:
            print(f"标注文件已存在: {annotations_file}")
        
        # 解压文件
        if not (oxford_pets_dir / "images").exists():
            extract_tar_gz(images_file, oxford_pets_dir)
        else:
            print("图像文件已解压")
            
        if not (oxford_pets_dir / "annotations").exists():
            extract_tar_gz(annotations_file, oxford_pets_dir)
        else:
            print("标注文件已解压")
        
        # 清理压缩文件（可选）
        print("\n清理压缩文件...")
        if images_file.exists():
            images_file.unlink()
            print(f"已删除: {images_file}")
        if annotations_file.exists():
            annotations_file.unlink()
            print(f"已删除: {annotations_file}")
        
        # 验证数据集结构
        images_dir = oxford_pets_dir / "images"
        annotations_dir = oxford_pets_dir / "annotations"
        
        if images_dir.exists() and annotations_dir.exists():
            image_count = len(list(images_dir.glob("*.jpg")))
            print(f"\n✅ 数据集设置成功!")
            print(f"📁 数据集位置: {oxford_pets_dir.absolute()}")
            print(f"🖼️  图像数量: {image_count}")
            print(f"📝 标注文件: {list(annotations_dir.glob('*.txt'))}")
            
            return True
        else:
            print("❌ 数据集设置失败: 缺少必要的目录")
            return False
            
    except Exception as e:
        print(f"❌ 下载过程中出现错误: {e}")
        return False

if __name__ == "__main__":
    print("🐱🐶 Oxford Pets 数据集下载器")
    print("=" * 50)
    
    # 设置数据集
    success = setup_oxford_pets_dataset()
    
    if success:
        print("\n🎉 数据集准备完成! 现在可以运行训练命令了。")
        print("\n建议的测试命令:")
        print("python federated_main.py --root DATA/ --model fedavg --dataset OxfordPets_LT --seed 1 --num_users 5 --frac 0.4 --lr 0.001 --csc False --gamma 1 --trainer FedClip --round 2 --partition homo --beta 1 --n_ctx 4 --dataset-config-file configs/datasets/oxford_pets_LT.yaml --config-file configs/trainers/PromptFL/vit_b16.yaml --output-dir output/test/oxford_pets_LT/ --imb_factor 0.01 --imb_type exp --ctx_init False --train_batch_size 16 --test_batch_size 32 --num_classes 37 --n_general 0 --fusion_frac 0.5 --fusion_mode wise --fusion_loss_alpha 0.99")
    else:
        print("\n❌ 数据集设置失败，请检查网络连接或手动下载。")