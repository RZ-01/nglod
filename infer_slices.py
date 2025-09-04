import argparse
import os
import random
import sys
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile

# Add sdf-net to path to import OctreeSDF
sys.path.append(os.path.join(os.path.dirname(__file__), 'sdf-net'))
from lib.models.OctreeSDF import OctreeSDF

# Define Args class before loading checkpoint to avoid pickle issues
class Args:
    def __init__(self):
        self.net = 'OctreeSDF'
        self.pos_enc = False          
        self.feature_dim = 64       # 增加特征维度：32→128，提高表达能力
        self.feature_size = 4        
        self.num_layers = 2          # 增加层数：1→2，提高网络深度
        self.num_lods = 5           
        self.base_lod = 2            
        self.ff_dim = 0              
        self.ff_width = 16.0         # Fourier特征宽度
        self.hidden_dim = 256        # 增加隐藏层维度：128→512，提高网络容量
        self.pos_invariant = False   
        self.joint_decoder = False   
        self.feat_sum = False        
        
        # 基础参数
        self.input_dim = 3           
        self.interpolate = None      # LOD插值值（None表示不插值，或者设为0.0-1.0之间的浮点数）
        
        self.epochs = 10000          
        self.batch_size = 100000        # 保持大批量训练
        self.grow_every = 1000       # 增加LOD增长间隔：1000→1500，让每个LOD训练更充分
        self.growth_strategy = 'increase'  
        
        self.optimizer = 'adam'      
        self.lr = 1e-4           
        self.loss = ['l1_loss']    # 使用L1损失
        self.return_lst = True            


def read_volume_shape(tif_path: str):
    # Try to get shape without loading all data
    try:
        with tifffile.TiffFile(tif_path) as tf:
            shape = tf.series[0].shape
            if len(shape) != 3:
                raise ValueError(f"Expected 3D volume, got shape {shape}")
            return tuple(int(x) for x in shape)  # (z, y, x)
    except Exception:
        vol = tifffile.imread(tif_path)
        if vol.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {vol.shape}")
        return tuple(int(x) for x in vol.shape)


def build_plane_normalized_coords(z_index: int, height: int, width: int, full_dims, device: torch.device) -> torch.Tensor:
    vz, vy, vx = full_dims
    ys = torch.arange(0, height, device=device, dtype=torch.float32)
    xs = torch.arange(0, width, device=device, dtype=torch.float32)
    # Normalize to [-1, 1] using global dims
    ys_n = ys / (vy - 1) * 2.0 - 1.0
    xs_n = xs / (vx - 1) * 2.0 - 1.0
    z_n_val = (float(z_index) / (vz - 1)) * 2.0 - 1.0
    grid_y, grid_x = torch.meshgrid(ys_n, xs_n, indexing='ij')
    grid_z = torch.full((height, width), z_n_val, device=device)
    coords = torch.stack([grid_x, grid_y, grid_z], dim=-1)
    coords = coords.view(-1, 3)
    return coords


def predict_plane(coords_flat: torch.Tensor, model: OctreeSDF, lod_level: int = None) -> torch.Tensor:
    """直接推理整个平面，不使用micro batch"""
    if lod_level is not None:
        pred = model.sdf(coords_flat, lod=lod_level)
    else:
        pred = model.sdf(coords_flat)
    return pred


def choose_random_slices(dz: int, num_slices: int, seed: int) -> List[int]:
    rnd = random.Random(seed)
    start_z = 300
    end_z = dz - 200
    
    if start_z >= end_z:
        # If range is invalid, fall back to full range
        indices = list(range(dz))
    else:
        indices = list(range(start_z, end_z))
    
    if num_slices >= len(indices):
        return indices
    return rnd.sample(indices, num_slices)


def main():
    parser = argparse.ArgumentParser(description="Infer random z-slices from a trained NGLOD occupancy network and save as float32 TIF")
    parser.add_argument("--volume_tif", type=str, default="Mouse_Heart_Angle0_patch.tif", help="Path to the reference volume (for shape)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/nglod_angle_0.pth", help="Path to the trained checkpoint .pth")
    parser.add_argument("--out_dir", type=str, default="nglod_out_new", help="Directory to save inferred TIFs")
    parser.add_argument("--num_slices", type=int, default=5, help="How many random z-slices to infer")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for selecting slices")
    parser.add_argument("--lod_level", type=int, default=0, help="Specific LOD level to use (0=coarse, 4=fine). None=auto select highest")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print("Using device:", device)

    # Get volume shape (z, y, x)
    dz, dy, dx = read_volume_shape(args.volume_tif)
    print(f"Reference volume shape: (z, y, x) = {(dz, dy, dx)}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Reconstruct NGLOD model from saved args
    saved_args = ckpt.get('args', None)
    model = OctreeSDF(saved_args).to(device)
    
    # Load model weights
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    print(f"Loaded NGLOD model with {sum(p.numel() for p in model.parameters())} parameters")
    if hasattr(saved_args, 'num_lods'):
        print(f"Model has {saved_args.num_lods} LOD levels")
    
    # Get current training stage if available
    current_stage = ckpt.get('current_stage', saved_args.num_lods if hasattr(saved_args, 'num_lods') else 5)
    max_lod = (saved_args.num_lods - 1) if hasattr(saved_args, 'num_lods') else 4
    
    if args.lod_level is None:
        use_lod = max_lod
        print(f"Auto-selecting highest LOD level: {use_lod}")
    else:
        use_lod = min(args.lod_level, max_lod)
        print(f"Using specified LOD level: {use_lod}")

    # Choose random z indices
    z_indices = choose_random_slices(dz, args.num_slices, args.seed)

    with torch.no_grad():
        for z_idx in z_indices:
            coords_flat = build_plane_normalized_coords(z_idx, dy, dx, (dz, dy, dx), device)
            pred_flat = predict_plane(coords_flat, model, lod_level=use_lod)
            pred_plane = pred_flat.view(dy, dx)
            pred_plane = pred_plane.clamp(0.0, 1.0)

            # Save as float32 in [0,1]
            out_img = pred_plane.detach().cpu().numpy().astype(np.float32)

            out_path = os.path.join(args.out_dir, f"nglod_infer_z{z_idx:05d}_lod{use_lod}.tif")
            tifffile.imwrite(out_path, out_img)
            print(f"Saved {out_path} with range [{out_img.min():.6f}, {out_img.max():.6f}]")


if __name__ == "__main__":
    main()


