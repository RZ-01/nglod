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

def load_nglod_model_from_checkpoint(checkpoint_path, device, freeze_mlp=False):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'args' in ckpt:
        nglod_args = ckpt['args']
        print("load nglod args from checkpoint")
    else:
        raise ValueError("No nglod args")
    
    model = OctreeSDF(nglod_args).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    
    """
    if freeze_mlp:
        for name, param in model.named_parameters():
            if 'louts' in name:
                param.requires_grad = False
                
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        print(f"Freezed: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        
        trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    
    else:
        total_params = sum(p.numel() for p in model.parameters())
    """
    return model, nglod_args

def load_psf_finetuned_checkpoint(checkpoint_path, device, freeze_mlp=True):
    print(f"Loading PSF-finetuned checkpoint: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Check if this is a PSF-finetuned checkpoint
    if 'finetune_type' not in ckpt or ckpt['finetune_type'] != 'psf':
        raise ValueError(f"Checkpoint {checkpoint_path} is not a PSF-finetuned checkpoint")
    
    if 'nglod_args' in ckpt:
        nglod_args = ckpt['nglod_args']
        print("Loaded nglod args from PSF-finetuned checkpoint")
    else:
        print("No nglod args in checkpoint, using default")
        nglod_args = Args()
    
    model = OctreeSDF(nglod_args).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    
    if freeze_mlp:
        for name, param in model.named_parameters():
            if 'louts' in name:
                param.requires_grad = False
                
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        print(f"Freezed: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
    else:
        total_params = sum(p.numel() for p in model.parameters())
    
    return model, nglod_args, ckpt  


def sample_block_start_indices(volume_shape, block_shape, psf_shape=None):
    vz, vy, vx = volume_shape
    bz, by, bx = block_shape
    

    z_min, z_max = 200, 430
    z0_low = z_min
    z0_high_inclusive = max(z_min, z_max - bz + 1)
    z0 = int(np.random.randint(z0_low, z0_high_inclusive + 1))
    
    if psf_shape is not None:
        pad_h, pad_w = psf_shape[1] // 2, psf_shape[2] // 2
        ext_h, ext_w = by + 2 * pad_h, bx + 2 * pad_w
        y0 = max(0, vy - ext_h)  
        x0 = max(0, vx - ext_w)  
    else:
        y0 = max(0, vy - by)  # 最右边（y轴最大位置）
        x0 = max(0, vx - bx)  # 最下边（x轴最大位置）

    return z0, y0, x0

def build_plane_coords_and_mask(z_index: int, y_start: int, x_start: int, height: int, width: int,
                                full_dims, device: torch.device):
    vz, vy, vx = full_dims
    ys_idx = torch.arange(y_start, y_start + height, device=device, dtype=torch.long)
    xs_idx = torch.arange(x_start, x_start + width, device=device, dtype=torch.long)

    valid_y = (ys_idx >= 0) & (ys_idx < vy)
    valid_x = (xs_idx >= 0) & (xs_idx < vx)
    mask = valid_y[:, None] & valid_x[None, :]

    ys_f = ys_idx.to(torch.float32)
    xs_f = xs_idx.to(torch.float32)
    ys_n = ys_f / (vy - 1) * 2.0 - 1.0
    xs_n = xs_f / (vx - 1) * 2.0 - 1.0
    z_n_val = (float(z_index) / (vz - 1)) * 2.0 - 1.0
    grid_y, grid_x = torch.meshgrid(ys_n, xs_n, indexing='ij')
    grid_z = torch.full((height, width), z_n_val, device=device)
    coords = torch.stack([grid_x, grid_y, grid_z], dim=-1).view(-1, 3)
    return coords, mask

def psf_finetune_step(model: nn.Module, norm_volume_np: np.ndarray,
                      device: torch.device, writer: SummaryWriter, global_step: int,
                      psf_kernels: torch.Tensor, block_shape, block_coords=None, reg_lambda=0.01) -> torch.Tensor:
    model.train()

    vz, vy, vx = norm_volume_np.shape
    bz, by, bx = block_shape
    kH, kW = psf_kernels.shape[1], psf_kernels.shape[2]
    pad_h, pad_w = kH // 2, kW // 2

    # PSF的深度和中心索引
    psf_depth = psf_kernels.shape[0]
    psf_center = psf_depth // 2  # PSF的中心层索引

    z0, y0, x0 = block_coords
    ext_h, ext_w = by + 2 * pad_h, bx + 2 * pad_w
    ext_y0, ext_x0 = y0 - pad_h, x0 - pad_w
    
    extended_target_volume = torch.zeros((bz, ext_h, ext_w), device=device, dtype=torch.float32)
    
    # 计算全局有效范围，避免在循环中重复计算
    y_start_global = max(0, ext_y0)
    y_end_global = min(vy, ext_y0 + ext_h)
    x_start_global = max(0, ext_x0)  
    x_end_global = min(vx, ext_x0 + ext_w)
    
    if y_start_global < y_end_global and x_start_global < x_end_global:
        ext_y_start = y_start_global - ext_y0
        ext_y_end = y_end_global - ext_y0
        ext_x_start = x_start_global - ext_x0
        ext_x_end = x_end_global - ext_x0
        
        # 批量填充有效区域，减少循环和条件判断
        for u in range(bz):
            z_idx = z0 + u
            if 0 <= z_idx < vz:
                target_slice = norm_volume_np[z_idx, y_start_global:y_end_global, x_start_global:x_end_global]
                extended_target_volume[u, ext_y_start:ext_y_end, ext_x_start:ext_x_end] = torch.from_numpy(target_slice).to(device)
    
    # 预计算扩展区域的mask和坐标（所有z层通用，避免重复计算）
    ys_idx_all = torch.arange(ext_y0, ext_y0 + ext_h, device=device, dtype=torch.float32)
    xs_idx_all = torch.arange(ext_x0, ext_x0 + ext_w, device=device, dtype=torch.float32)
    valid_y_all = (ys_idx_all >= 0) & (ys_idx_all < vy)
    valid_x_all = (xs_idx_all >= 0) & (xs_idx_all < vx)
    mask_ext_all = valid_y_all[:, None] & valid_x_all[None, :]
    
    # 预计算归一化坐标网格（Y和X坐标对所有z层都相同）
    ys_n_all = ys_idx_all / (vy - 1) * 2.0 - 1.0
    xs_n_all = xs_idx_all / (vx - 1) * 2.0 - 1.0
    grid_y_all, grid_x_all = torch.meshgrid(ys_n_all, xs_n_all, indexing='ij')
    
    # 高效的单次循环：生成预测 + 计算正则化损失
    predicted_clear_extended_planes = []
    reg_loss = 0.0
    
    for u in range(bz):
        z_idx = z0 + u
        
        # 只需计算z坐标（Y和X坐标已预计算）
        z_n_val = (float(z_idx) / (vz - 1)) * 2.0 - 1.0
        grid_z = torch.full((ext_h, ext_w), z_n_val, device=device)
        coords_flat = torch.stack([grid_x_all, grid_y_all, grid_z], dim=-1).view(-1, 3)
        
        # NGLOD推理
        pred_flat = model.sdf(coords_flat)
        pred_plane_ext = pred_flat.view(1, 1, ext_h, ext_w)
        
        # 应用预计算的mask
        pred_plane_ext = pred_plane_ext * mask_ext_all.to(pred_plane_ext.dtype).unsqueeze(0).unsqueeze(0)
        predicted_clear_extended_planes.append(pred_plane_ext)
        
        # 正则化损失计算（使用预计算的mask）
        if mask_ext_all.sum() > 0:
            pred_2d = pred_plane_ext.squeeze(0).squeeze(0)  # [ext_h, ext_w]
            target_2d = extended_target_volume[u]  # [ext_h, ext_w] 
            reg_loss += F.l1_loss(pred_2d[mask_ext_all], target_2d[mask_ext_all])

    # PSF卷积模拟（保持原有逻辑，但减少条件判断）
    simulated_focal_planes = []
    for focal_z in range(bz):
        simulated_focal_plane = torch.zeros((1, 1, by, bx), device=device)
        for u in range(bz):
            z_distance = u - focal_z
            psf_idx = psf_center + z_distance
            
            # 减少分支：直接计算，越界的kernel会被预先设置为0或跳过
            if 0 <= psf_idx < psf_depth:
                kernel = psf_kernels[psf_idx].unsqueeze(0).unsqueeze(0)
                contribution = F.conv2d(predicted_clear_extended_planes[u], kernel, padding=0)
                simulated_focal_plane += contribution
                
        simulated_focal_planes.append(simulated_focal_plane)
    
    # 计算主损失
    target_block_np = norm_volume_np[z0:z0+bz, y0:y0+by, x0:x0+bx]
    target_block = torch.from_numpy(target_block_np).to(device=device, dtype=torch.float32)
    simulated_block = torch.cat(simulated_focal_planes, dim=0).squeeze(1)  # [bz, by, bx]
    main_loss = F.l1_loss(simulated_block, target_block)
    
    # 平均正则化损失
    reg_loss = reg_loss / bz if bz > 0 else reg_loss
    
    # 总损失
    total_loss = main_loss + reg_lambda * reg_loss
    
    writer.add_scalar("PSF_FT/Main_Loss_L1_3D", main_loss.item(), global_step)
    writer.add_scalar("PSF_FT/Reg_Loss_L1", reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss, global_step)
    writer.add_scalar("PSF_FT/Total_Loss_L1_3D", total_loss.item(), global_step)
    return total_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume_tif", type=str, default="../data/Mouse_Heart_Angle0_patch.tif")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/nglod_angle_0.pth")  
    parser.add_argument("--psf_npy", type=str, default="psf_t0_v0.npy")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--block_z", type=int, default=125)
    parser.add_argument("--block_h", type=int, default=133)
    parser.add_argument("--block_w", type=int, default=133)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_path", type=str, default="checkpoints/nglod_psf_finetuned_full.pth")
    parser.add_argument("--logdir", type=str, default="runs/psf_finetune_nglod_full")
    parser.add_argument("--freeze_mlp", action="store_true", default=False)
    parser.add_argument("--load_from_checkpoint", type=str, default=None)
    parser.add_argument("--reg_lambda", type=float, default=0.01, help="正则化损失权重")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vol = tifffile.imread(args.volume_tif).astype(np.float32)
    vol_max = float(vol.max())
    vol_norm = vol / vol_max
    
    if args.load_from_checkpoint:
        model, nglod_args, ckpt = load_psf_finetuned_checkpoint(args.load_from_checkpoint, device, args.freeze_mlp)
        print(f"Resuming from PSF-finetuned checkpoint at step {ckpt.get('epoch', 0)}")
    else:
        model, nglod_args = load_nglod_model_from_checkpoint(args.checkpoint, device, args.freeze_mlp)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    if args.load_from_checkpoint and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # 强制更新学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
        #print(f"Loaded optimizer state from checkpoint and updated lr to {args.lr}")

    total_steps = args.steps
    #hold_steps = 200
    #decay_step_size = 200
    #gamma = 0.7
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_step_size, gamma=gamma)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    
    writer = SummaryWriter(log_dir=args.logdir)

    psf = np.load(args.psf_npy).astype(np.float32)
    psf = psf / (psf.sum() + 1e-12)
    psf_kernels = torch.from_numpy(psf).to(device)

    print(f"PSF shape: {psf_kernels.shape}, Block shape: ({args.block_z}, {args.block_h}, {args.block_w})")
    block_shape = (args.block_z, args.block_h, args.block_w)
    
    # 每个block训练的epoch数
    epochs_per_block = 10
    
    pbar = tqdm(
        total=total_steps,
        desc="Training steps",
        dynamic_ncols=True,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
    )
    reg = args.reg_lambda
    
    # 初始化第一个block
    current_block_coords = sample_block_start_indices(vol_norm.shape, block_shape, psf_kernels.shape)
    
    for step in range(total_steps):
        # 每10个epoch切换到新的block
        if step % epochs_per_block == 0:
            current_block_coords = sample_block_start_indices(vol_norm.shape, block_shape, psf_kernels.shape)
           # print(f"\nStep {step}: Switching to new block at coordinates {current_block_coords}")
        
        optimizer.zero_grad(set_to_none=True)
        
        total_loss = psf_finetune_step(
            model=model,
            norm_volume_np=vol_norm,
            device=device,
            writer=writer,
            global_step=step,
            psf_kernels=psf_kernels,
            block_shape=block_shape,
            block_coords=current_block_coords,
            reg_lambda=reg,
        )
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=30.0)
        optimizer.step()
        
        # 前半段不调用 scheduler
       # if step >= hold_steps:
        scheduler.step()
        
        pbar.update(1)
        
        if step < 800 and (step + 1) % 100 == 0:
            reg *= 0.7

        if step >= 800:
            reg *= 0

        
        # 每隔200个epoch保存checkpoint
        if (step + 1) % 4000 == 0:
            checkpoint_name = f"checkpoint_step_{step + 1}_block.pth"
            checkpoint_path = os.path.join(os.path.dirname(args.save_path), checkpoint_name)
            
            save_obj = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                #'total_loss': float(total_loss.item()),
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
                #'epoch': step + 1,
                'finetune_type': 'psf',
            }
            torch.save(save_obj, checkpoint_path)
            print(f"\nSaved checkpoint at step {step + 1}: {checkpoint_path}")

    pbar.close()

    # Save new checkpoint
    save_obj = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        #'total_loss': float(total_loss.item()),
        'nglod_args': nglod_args,  # NGLOD的网络架构参数
        'psf_finetune_config': {   # PSF微调的训练配置
            'lr': args.lr,
            'steps': args.steps,
            'block_z': args.block_z,
            'block_h': args.block_h,
            'block_w': args.block_w,
            'psf_npy': args.psf_npy,
            'volume_tif': args.volume_tif,
        },
        #'epoch': args.steps,
        'finetune_type': 'psf',  # 标记这是PSF微调的检查点
    }
    torch.save(save_obj, args.save_path)
    print(f"Saved PSF-finetuned NGLOD model to {args.save_path}")


if __name__ == "__main__":
    main()
