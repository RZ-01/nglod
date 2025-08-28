import numpy as np
import tifffile
import os

def extract_original_slices():
    """从原始体积数据中提取指定的切片"""
    
    # 要提取的切片索引（与推理结果对应）
    slice_indices = [463, 328, 306, 489, 370]
    
    # 读取原始体积数据
    print("Loading original volume...")
    vol = tifffile.imread('Mouse_Heart_Angle0_patch.tif')
    print(f"Original volume shape: {vol.shape}")
    print(f"Original volume dtype: {vol.dtype}")
    print(f"Original volume range: [{vol.min()}, {vol.max()}]")
    
    # 归一化到[0,1]（与训练时相同）
    vol = vol.astype(np.float32)
    vol_max = float(vol.max())
    vol_normalized = vol / vol_max
    print(f"Normalized volume range: [{vol_normalized.min():.6f}, {vol_normalized.max():.6f}]")
    
    # 创建输出目录
    out_dir = "original_slices"
    os.makedirs(out_dir, exist_ok=True)
    
    # 提取每个切片
    for z_idx in slice_indices:
        if z_idx >= vol.shape[0]:
            print(f"Warning: slice {z_idx} exceeds volume depth {vol.shape[0]}")
            continue
            
        # 提取原始切片（未归一化）
        original_slice = vol[z_idx, :, :].astype(np.float32)
        
        # 提取归一化切片
        normalized_slice = vol_normalized[z_idx, :, :].astype(np.float32)
        
        # 保存原始切片
        original_path = os.path.join(out_dir, f"original_z{z_idx:05d}.tif")
        tifffile.imwrite(original_path, original_slice)
        print(f"Saved {original_path} with range [{original_slice.min():.6f}, {original_slice.max():.6f}]")
        
        # 保存归一化切片
        normalized_path = os.path.join(out_dir, f"normalized_z{z_idx:05d}.tif")
        tifffile.imwrite(normalized_path, normalized_slice)
        print(f"Saved {normalized_path} with range [{normalized_slice.min():.6f}, {normalized_slice.max():.6f}]")
        
        # 打印一些统计信息
        print(f"  Slice {z_idx} stats:")
        print(f"    Original: mean={original_slice.mean():.3f}, std={original_slice.std():.3f}")
        print(f"    Normalized: mean={normalized_slice.mean():.6f}, std={normalized_slice.std():.6f}")
        print()

if __name__ == "__main__":
    extract_original_slices()
