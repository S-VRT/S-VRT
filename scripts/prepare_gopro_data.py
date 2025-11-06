#!/usr/bin/env python3
"""
重组GoPro数据集结构以匹配VRT项目要求

从: trainsets/gopro_spike/GOPRO_Large/train/VIDEO/sharp|blur/*.png
到: trainsets/GoPro/train_GT|train_GT_blurred/VIDEO/*.png
"""

import os
import shutil
from pathlib import Path

def prepare_gopro_dataset(use_symlinks=True):
    """
    重组GoPro数据集
    
    Args:
        use_symlinks: 如果为True，创建符号链接（节省空间）；如果为False，复制文件
    """
    
    # 源路径
    source_base = Path('trainsets/gopro_spike/GOPRO_Large')
    
    # 目标路径
    target_base = Path('trainsets/GoPro')
    
    # 处理训练集
    print("=" * 60)
    print("重组训练集...")
    print("=" * 60)
    
    source_train = source_base / 'train'
    target_gt = target_base / 'train_GT'
    target_blur = target_base / 'train_GT_blurred'
    
    # 创建目标目录
    target_gt.mkdir(parents=True, exist_ok=True)
    target_blur.mkdir(parents=True, exist_ok=True)
    
    # 遍历所有视频序列
    video_folders = sorted([d for d in source_train.iterdir() if d.is_dir()])
    
    for video_folder in video_folders:
        video_name = video_folder.name
        print(f"\n处理视频序列: {video_name}")
        
        # 源文件夹
        sharp_folder = video_folder / 'sharp'
        blur_folder = video_folder / 'blur'
        
        # 目标文件夹
        target_gt_video = target_gt / video_name
        target_blur_video = target_blur / video_name
        
        # 创建目标视频文件夹
        target_gt_video.mkdir(exist_ok=True)
        target_blur_video.mkdir(exist_ok=True)
        
        # 获取所有图片文件
        sharp_images = sorted(sharp_folder.glob('*.png'))
        blur_images = sorted(blur_folder.glob('*.png'))
        
        print(f"  - Sharp图像: {len(sharp_images)}张")
        print(f"  - Blur图像: {len(blur_images)}张")
        
        # 处理sharp图像
        for img in sharp_images:
            target_file = target_gt_video / img.name
            if not target_file.exists():
                if use_symlinks:
                    # 创建符号链接（相对路径）
                    rel_path = os.path.relpath(img, target_file.parent)
                    os.symlink(rel_path, target_file)
                else:
                    # 复制文件
                    shutil.copy2(img, target_file)
        
        # 处理blur图像
        for img in blur_images:
            target_file = target_blur_video / img.name
            if not target_file.exists():
                if use_symlinks:
                    # 创建符号链接（相对路径）
                    rel_path = os.path.relpath(img, target_file.parent)
                    os.symlink(rel_path, target_file)
                else:
                    # 复制文件
                    shutil.copy2(img, target_file)
        
        print(f"  ✓ 完成")
    
    # 处理测试集
    print("\n" + "=" * 60)
    print("重组测试集...")
    print("=" * 60)
    
    source_test = source_base / 'test'
    target_test_gt = Path('testsets/GoPro11/test_GT')
    target_test_blur = Path('testsets/GoPro11/test_GT_blurred')
    
    # 创建目标目录
    target_test_gt.mkdir(parents=True, exist_ok=True)
    target_test_blur.mkdir(parents=True, exist_ok=True)
    
    # 遍历所有测试视频序列
    test_video_folders = sorted([d for d in source_test.iterdir() if d.is_dir()])
    
    for video_folder in test_video_folders:
        video_name = video_folder.name
        print(f"\n处理测试视频: {video_name}")
        
        # 源文件夹
        sharp_folder = video_folder / 'sharp'
        blur_folder = video_folder / 'blur'
        
        if not sharp_folder.exists() or not blur_folder.exists():
            print(f"  ⚠ 跳过（文件夹不完整）")
            continue
        
        # 目标文件夹
        target_test_gt_video = target_test_gt / video_name
        target_test_blur_video = target_test_blur / video_name
        
        # 创建目标视频文件夹
        target_test_gt_video.mkdir(exist_ok=True)
        target_test_blur_video.mkdir(exist_ok=True)
        
        # 获取所有图片文件
        sharp_images = sorted(sharp_folder.glob('*.png'))
        blur_images = sorted(blur_folder.glob('*.png'))
        
        print(f"  - Sharp图像: {len(sharp_images)}张")
        print(f"  - Blur图像: {len(blur_images)}张")
        
        # 处理sharp图像
        for img in sharp_images:
            target_file = target_test_gt_video / img.name
            if not target_file.exists():
                if use_symlinks:
                    rel_path = os.path.relpath(img, target_file.parent)
                    os.symlink(rel_path, target_file)
                else:
                    shutil.copy2(img, target_file)
        
        # 处理blur图像
        for img in blur_images:
            target_file = target_test_blur_video / img.name
            if not target_file.exists():
                if use_symlinks:
                    rel_path = os.path.relpath(img, target_file.parent)
                    os.symlink(rel_path, target_file)
                else:
                    shutil.copy2(img, target_file)
        
        print(f"  ✓ 完成")
    
    print("\n" + "=" * 60)
    print("✓ 数据重组完成！")
    print("=" * 60)
    print(f"\n训练集位置:")
    print(f"  - GT: {target_gt}")
    print(f"  - Blurred: {target_blur}")
    print(f"\n测试集位置:")
    print(f"  - GT: {target_test_gt}")
    print(f"  - Blurred: {target_test_blur}")
    print(f"\n下一步: 运行以下命令创建LMDB数据库")
    print(f"  python create_lmdb.py --dataset gopro")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='准备GoPro数据集')
    parser.add_argument('--copy', action='store_true', 
                        help='复制文件而不是创建符号链接（需要更多磁盘空间）')
    args = parser.parse_args()
    
    use_symlinks = not args.copy
    
    if use_symlinks:
        print("使用符号链接模式（节省磁盘空间）")
    else:
        print("使用复制模式（需要更多磁盘空间）")
    
    prepare_gopro_dataset(use_symlinks=use_symlinks)

