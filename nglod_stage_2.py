import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sys
import math

sys.path.append(os.path.join(os.path.dirname(__file__), 'sdf-net'))
from lib.models.OctreeSDF import OctreeSDF


class Args:
    """NGLOD原始参数类 - 用于加载nglod.py保存的检查点"""
    def __init__(self):
        # ===== 网络架构参数 =====
        self.net = 'OctreeSDF'
        self.feature_dim = 32     
        self.feature_size = 4        
        self.num_layers = 1         
        self.hidden_dim = 512      
        self.input_dim = 3           
        
        # ===== LOD相关参数 =====
        self.num_lods = 5           
        self.base_lod = 2            
        self.interpolate = None    
        self.growth_strategy = 'increase'  
        self.grow_every = -1       
        
        # ===== 位置编码参数 =====
        self.pos_enc = False          
        self.ff_dim = -1                    
        
        # ===== 训练参数 =====
        self.epochs = 10000          
        self.batch_size = 100000       
        self.optimizer = 'adam'      
        self.lr = 6e-4         
        self.loss = ['l1_loss']    
        
        # ===== 其他参数 =====
        self.pos_invariant = False   
        self.joint_decoder = False   
        self.feat_sum = False        
        self.return_lst = True


def load_nglod_model_from_checkpoint(checkpoint_path, device, freeze_mlp=True):
    """加载NGLOD模型并冻结MLP参数"""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'args' in ckpt:
        nglod_args = ckpt['args']
        print("Loaded nglod args from checkpoint")
    else:
        raise ValueError("No nglod args found in checkpoint")
    
    model = OctreeSDF(nglod_args).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    
    # 冻结所有MLP参数
    if freeze_mlp:
        for name, param in model.named_parameters():
            if 'louts' in name:  # MLP decoder参数
                param.requires_grad = False
                #print(f"Frozen: {name}")
        
        # 确保feature参数可训练
        for name, param in model.named_parameters():
            if 'features' in name and 'fm' in name:
                param.requires_grad = True
                #print(f"Trainable: {name}, shape: {param.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    #print(f"\nTotal params: {total_params:,}")
    #print(f"Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    #print(f"Frozen params: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
    
    return model, nglod_args


def compute_block_feature_mask(block_coords, block_shape, volume_shape, 
                               lod_idx, base_lod, pad_ratio=0.2):
    """
    计算当前block在指定LOD的feature grid中对应的voxel范围
    
    Args:
        block_coords: (z0, y0, x0) 块的起始坐标（在原始volume中）
        block_shape: (bz, by, bx) 块的形状
        volume_shape: (vz, vy, vx) 完整volume的形状
        lod_idx: 当前LOD层级索引
        base_lod: base LOD参数
        pad_ratio: 扩展比例，用于包含邻近的feature voxels
    
    Returns:
        mask: 布尔tensor，形状为feature grid的形状，True表示该voxel需要更新
    """
    z0, y0, x0 = block_coords
    bz, by, bx = block_shape
    vz, vy, vx = volume_shape
    
    # 计算该LOD的feature grid分辨率
    fsize = 2 ** (lod_idx + base_lod)
    
    # 将block坐标转换到归一化空间[-1, 1]
    z0_norm = (z0 / (vz - 1)) * 2.0 - 1.0
    y0_norm = (y0 / (vy - 1)) * 2.0 - 1.0
    x0_norm = (x0 / (vx - 1)) * 2.0 - 1.0
    
    z1_norm = ((z0 + bz - 1) / (vz - 1)) * 2.0 - 1.0
    y1_norm = ((y0 + by - 1) / (vy - 1)) * 2.0 - 1.0
    x1_norm = ((x0 + bx - 1) / (vx - 1)) * 2.0 - 1.0
    
    # 扩展边界，确保包含所有可能被采样的feature voxels
    pad_z = (z1_norm - z0_norm) * pad_ratio
    pad_y = (y1_norm - y0_norm) * pad_ratio
    pad_x = (x1_norm - x0_norm) * pad_ratio
    
    z0_norm -= pad_z
    y0_norm -= pad_y
    x0_norm -= pad_x
    z1_norm += pad_z
    y1_norm += pad_y
    x1_norm += pad_x
    
    # 将归一化坐标转换到feature grid坐标 [0, fsize]
    # grid_sample使用的坐标范围是[-1,1]对应到[0, fsize]
    z0_feat = (z0_norm + 1.0) / 2.0 * fsize
    y0_feat = (y0_norm + 1.0) / 2.0 * fsize
    x0_feat = (x0_norm + 1.0) / 2.0 * fsize
    
    z1_feat = (z1_norm + 1.0) / 2.0 * fsize
    y1_feat = (y1_norm + 1.0) / 2.0 * fsize
    x1_feat = (x1_norm + 1.0) / 2.0 * fsize
    
    # 转换为整数索引（向下取整和向上取整）
    z0_idx = max(0, int(np.floor(z0_feat)))
    y0_idx = max(0, int(np.floor(y0_feat)))
    x0_idx = max(0, int(np.floor(x0_feat)))
    
    z1_idx = min(fsize, int(np.ceil(z1_feat)) + 1)
    y1_idx = min(fsize, int(np.ceil(y1_feat)) + 1)
    x1_idx = min(fsize, int(np.ceil(x1_feat)) + 1)
    
    # 创建mask
    mask = torch.zeros((1, 1, fsize+1, fsize+1, fsize+1), dtype=torch.bool)
    mask[0, 0, z0_idx:z1_idx, y0_idx:y1_idx, x0_idx:x1_idx] = True
    
    return mask


def register_gradient_masks(model, block_coords, block_shape, volume_shape, base_lod, device):
    """
    为模型的feature parameters注册梯度mask，只允许与当前block相关的features更新
    
    Returns:
        hook_handles: hook句柄列表，用于后续移除
    """
    hook_handles = []
    
    for lod_idx, feature_module in enumerate(model.features):
        mask = compute_block_feature_mask(
            block_coords, block_shape, volume_shape, 
            lod_idx, base_lod, pad_ratio=0.3
        ).to(device)
        

        def create_hook(mask_tensor):
            def hook(grad):
                # 只保留mask为True的位置的梯度
                if grad is not None:
                    return grad * mask_tensor.float()
                return grad
            return hook
        
        handle = feature_module.fm.register_hook(create_hook(mask))
        hook_handles.append(handle)
    
    return hook_handles


def remove_gradient_hooks(hook_handles):
    for handle in hook_handles:
        handle.remove()

def idx_to_coord(start, end, stride, idx, steps):
    if idx == steps - 1:
        return end
    return start + idx * stride

def steps_ceil(start, end, stride):
    return math.ceil((end - start) / stride)

def sample_block_start_indices(volume_shape, block_shape, region_size, step=0):
    """
    Iterate through blocks in a fixed row-column order
    """
    vz, vy, vx = volume_shape
    bz, by, bx = block_shape
    
    z_min, z_max = 200, 430
    z0_low = z_min
    z0_high_inclusive = max(z_min, z_max - bz + 1)
    
    #region_size = region_size
    y_min = vy - region_size
    x_min = vx - region_size
    y_max = vy - by
    x_max = vx - bx

    sz = bz // 4
    sy = by // 4
    sx = bx // 4

    z_steps = steps_ceil(z0_low, z0_high_inclusive, sz)
    y_steps = steps_ceil(y_min, y_max, sy)
    x_steps = steps_ceil(x_min, x_max, sx)

    
    total_blocks = z_steps * y_steps * x_steps

    current_block = step % total_blocks
    
    z_idx = current_block // (y_steps * x_steps)
    remaining = current_block % (y_steps * x_steps)
    y_idx = remaining // x_steps
    x_idx = remaining % x_steps
    
    z0 = idx_to_coord(z0_low, z0_high_inclusive, sz, z_idx, z_steps)
    y0 = idx_to_coord(y_min,     y_max,          sy, y_idx, y_steps)
    x0 = idx_to_coord(x_min,     x_max,          sx, x_idx, x_steps)
    
    return z0, y0, x0




def psf_finetune_step(model: nn.Module, norm_volume_np: np.ndarray,
                      device: torch.device, writer: SummaryWriter, global_step: int,
                      psf_kernels: torch.Tensor, block_shape, block_coords=None) -> torch.Tensor:
    """PSF微调的单步训练"""
    model.train()

    vz, vy, vx = norm_volume_np.shape
    bz, by, bx = block_shape
    kH, kW = psf_kernels.shape[1], psf_kernels.shape[2]
    pad_h, pad_w = kH // 2, kW // 2

    psf_depth = psf_kernels.shape[0]
    psf_center = psf_depth // 2

    z0, y0, x0 = block_coords
    ext_h, ext_w = by + 2 * pad_h, bx + 2 * pad_w
    ext_y0, ext_x0 = y0 - pad_h, x0 - pad_w
    """
    
    extended_target_volume = torch.zeros((bz, ext_h, ext_w), device=device, dtype=torch.float32)
    
    # 计算全局有效范围
    y_start_global = max(0, ext_y0)
    y_end_global = min(vy, ext_y0 + ext_h)
    x_start_global = max(0, ext_x0)  
    x_end_global = min(vx, ext_x0 + ext_w)
    
    if y_start_global < y_end_global and x_start_global < x_end_global:
        ext_y_start = y_start_global - ext_y0
        ext_y_end = y_end_global - ext_y0
        ext_x_start = x_start_global - ext_x0
        ext_x_end = x_end_global - ext_x0
        
        for u in range(bz):
            z_idx = z0 + u
            if 0 <= z_idx < vz:
                target_slice = norm_volume_np[z_idx, y_start_global:y_end_global, x_start_global:x_end_global]
                extended_target_volume[u, ext_y_start:ext_y_end, ext_x_start:ext_x_end] = torch.from_numpy(target_slice).to(device)
    """
    
    # 预计算坐标网格
    ys_idx_all = torch.arange(ext_y0, ext_y0 + ext_h, device=device, dtype=torch.float32)
    xs_idx_all = torch.arange(ext_x0, ext_x0 + ext_w, device=device, dtype=torch.float32)
    valid_y_all = (ys_idx_all >= 0) & (ys_idx_all < vy)
    valid_x_all = (xs_idx_all >= 0) & (xs_idx_all < vx)
    mask_ext_all = valid_y_all[:, None] & valid_x_all[None, :]
    
    ys_n_all = ys_idx_all / (vy - 1) * 2.0 - 1.0
    xs_n_all = xs_idx_all / (vx - 1) * 2.0 - 1.0
    grid_y_all, grid_x_all = torch.meshgrid(ys_n_all, xs_n_all, indexing='ij')
    
    # 生成预测
    predicted_clear_extended_planes = []
    
    for u in range(bz):
        z_idx = z0 + u
        z_n_val = (float(z_idx) / (vz - 1)) * 2.0 - 1.0
        grid_z = torch.full((ext_h, ext_w), z_n_val, device=device)
        coords_flat = torch.stack([grid_x_all, grid_y_all, grid_z], dim=-1).view(-1, 3)
        
        # NGLOD推理
        pred_flat = model.sdf(coords_flat)
        pred_plane_ext = pred_flat.view(1, 1, ext_h, ext_w)
        pred_plane_ext = pred_plane_ext * mask_ext_all.to(pred_plane_ext.dtype).unsqueeze(0).unsqueeze(0)
        predicted_clear_extended_planes.append(pred_plane_ext)

    # PSF卷积模拟
    simulated_focal_planes = []
    for focal_z in range(bz):
        simulated_focal_plane = torch.zeros((1, 1, by, bx), device=device)
        for u in range(bz):
            z_distance = u - focal_z
            psf_idx = psf_center + z_distance
            
            if 0 <= psf_idx < psf_depth:
                kernel = psf_kernels[psf_idx].unsqueeze(0).unsqueeze(0)
                contribution = F.conv2d(predicted_clear_extended_planes[u], kernel, padding=0)
                simulated_focal_plane += contribution
                
        simulated_focal_planes.append(simulated_focal_plane)
    
    # 计算损失
    target_block_np = norm_volume_np[z0:z0+bz, y0:y0+by, x0:x0+bx]
    target_block = torch.from_numpy(target_block_np).to(device=device, dtype=torch.float32)
    simulated_block = torch.cat(simulated_focal_planes, dim=0).squeeze(1)
    loss = F.l1_loss(simulated_block, target_block)
    
    writer.add_scalar("PSF_FT_Selective/Loss", loss.item(), global_step)
    
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume_tif", type=str, default="../data/Mouse_Heart_Angle0_patch.tif")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/nglod_angle_0.pth")  
    parser.add_argument("--psf_npy", type=str, default="psf_t0_v0.npy")
    parser.add_argument("--steps", type=int, default=4800)
    parser.add_argument("--lr", type=float, default=6e-3)
    parser.add_argument("--block_z", type=int, default=115)
    parser.add_argument("--block_h", type=int, default=266)
    parser.add_argument("--block_w", type=int, default=266)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_path", type=str, default="checkpoints/nglod_psf_plain.pth")
    parser.add_argument("--logdir", type=str, default="runs/psf_finetune_plain")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vol = tifffile.imread(args.volume_tif).astype(np.float32)
    vol_max = float(vol.max())
    vol_norm = vol / vol_max
    
    # 加载模型并冻结MLP
    model, nglod_args = load_nglod_model_from_checkpoint(args.checkpoint, device, freeze_mlp=True)
    
    # 只优化feature parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    total_steps = args.steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    
    writer = SummaryWriter(log_dir=args.logdir)

    psf = np.load(args.psf_npy).astype(np.float32)
    psf = psf / (psf.sum() + 1e-12)
    psf_kernels = torch.from_numpy(psf).to(device)

    print(f"\nPSF shape: {psf_kernels.shape}")
    print(f"Block shape: ({args.block_z}, {args.block_h}, {args.block_w})")
    print(f"Volume shape: {vol_norm.shape}\n")
    
    block_shape = (args.block_z, args.block_h, args.block_w)
    
    # Debug block大小为训练块的1/4
   # debug_block_shape = (args.block_z // 32, args.block_h // 32, args.block_w // 32)  # 约(31, 33, 33)
   # xy_region_start = vol_norm.shape[1] - 266
    #debug_block_coords = (250, xy_region_start + 50, xy_region_start + 50)  
    #ebug_losses = []  
    
    #print(f"Debug monitoring block at coordinates: {debug_block_coords}")
    
    pbar = tqdm(
        total=total_steps,
        desc="Training ",
        dynamic_ncols=True,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
    )
    
    for step in range(total_steps):
        current_block_coords = sample_block_start_indices(
            vol_norm.shape, block_shape, 532, step
        )
        
        # 检查当前训练的block是否与debug block重叠
        """
        cz, cy, cx = current_block_coords
        dz, dy, dx = debug_block_coords
        dbz, dby, dbx = debug_block_shape
        bz, by, bx = block_shape
        
        # 扩展训练块的范围（考虑pad_ratio=0.3）
        pad_z = bz * 0.3
        pad_y = by * 0.3
        pad_x = bx * 0.3
        
        c_z_min, c_z_max = cz - pad_z, cz + bz + pad_z
        c_y_min, c_y_max = cy - pad_y, cy + by + pad_y
        c_x_min, c_x_max = cx - pad_x, cx + bx + pad_x
        
        d_z_min, d_z_max = dz, dz + dbz
        d_y_min, d_y_max = dy, dy + dby
        d_x_min, d_x_max = dx, dx + dbx
        
        # 检查是否有重叠（两个box相交）
        overlap_z = not (c_z_max <= d_z_min or c_z_min >= d_z_max)
        overlap_y = not (c_y_max <= d_y_min or c_y_min >= d_y_max)
        overlap_x = not (c_x_max <= d_x_min or c_x_min >= d_x_max)
        is_debug_block_affected = overlap_z and overlap_y and overlap_x
        """
        
        hook_handles = register_gradient_masks(
            model, current_block_coords, block_shape, 
            vol_norm.shape, nglod_args.base_lod, device
        )
        
        optimizer.zero_grad(set_to_none=True)
        
        loss = psf_finetune_step(
            model=model,
            norm_volume_np=vol_norm,
            device=device,
            writer=writer,
            global_step=step,
            psf_kernels=psf_kernels,
            block_shape=block_shape,
            block_coords=current_block_coords,
        )
        
        loss.backward()
        
        remove_gradient_hooks(hook_handles)
        
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=30.0)
        optimizer.step()
        scheduler.step()
        
        pbar.update(1)
        pbar.set_postfix({
            'loss': f'{loss.item():.6f}',
            'block': f'({current_block_coords[0]},{current_block_coords[1]},{current_block_coords[2]})'
        })
        """
        if step % 1 == 0: 
            with torch.no_grad():
                debug_loss = psf_finetune_step(
                    model=model,
                    norm_volume_np=vol_norm,
                    device=device,
                    writer=writer,
                    global_step=step,
                    psf_kernels=psf_kernels,
                    block_shape=debug_block_shape,
                    block_coords=debug_block_coords,
                    reg_lambda=reg,
                )
                debug_losses.append(debug_loss.item())
                writer.add_scalar("Debug/Fixed_Block_Loss", debug_loss.item(), step)
                writer.add_scalar("Debug/Is_Debug_Block_Affected", float(is_debug_block_affected), step)
                
                if len(debug_losses) >= 2:
                    loss_change = debug_losses[-1] - debug_losses[-2]
                    affected_str = "AFFECTED" if is_debug_block_affected else "NOT affected"
                    print(f"\n[DEBUG] Step {step}: Debug block {debug_block_coords} (size={debug_block_shape})")
                    print(f"  Training block {current_block_coords} (size={block_shape})")
                    print(f"  Loss = {debug_loss.item():.6f}, change = {loss_change:+.6f}, {affected_str}")
                """
        #if step < 800 and (step + 1) % 100 == 0:
        #    reg *= 0.7

    pbar.close()

    save_obj = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'nglod_args': nglod_args,
        'psf_finetune_config': {
            'lr': args.lr,
            'steps': args.steps,
            'block_z': args.block_z,
            'block_h': args.block_h,
            'block_w': args.block_w,
            'psf_npy': args.psf_npy,
            'volume_tif': args.volume_tif,
        },
        'finetune_type': 'psf_selective',
        'training_mode': 'selective_feature_update'
    }
    torch.save(save_obj, args.save_path)
    print(f"\nSaved final PSF-finetuned model (selective) to {args.save_path}")


if __name__ == "__main__":
    main()
