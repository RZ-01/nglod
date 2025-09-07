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

torch.set_float32_matmul_precision('high')


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
    print(f"Loading NGLOD checkpoint: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'args' in ckpt:
        nglod_args = ckpt['args']
        print("load nglod args from checkpoint")
    else:
        print("no args in checkpoint, use default nglod args")
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
        
        trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    else:
        total_params = sum(p.numel() for p in model.parameters())
    
    return model, nglod_args  


def sample_block_start_indices(volume_shape, block_shape):
    vz, vy, vx = volume_shape
    bz, by, bx = block_shape
    z0 = np.random.randint(0, max(1, vz - bz + 1))
    y0 = np.random.randint(0, max(1, vy - by + 1))
    x0 = np.random.randint(0, max(1, vx - bx + 1))
    
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
                      psf_kernels: torch.Tensor, block_shape) -> torch.Tensor:
    model.train()

    vz, vy, vx = norm_volume_np.shape
    bz, by, bx = block_shape
    kH, kW = psf_kernels.shape[1], psf_kernels.shape[2]
    pad_h, pad_w = kH // 2, kW // 2

    if psf_kernels.shape[0] != bz:
        raise ValueError(f"PSF depth {psf_kernels.shape[0]} must equal block depth {bz}")

    z0, y0, x0 = sample_block_start_indices((vz, vy, vx), (bz, by, bx))

    ext_h, ext_w = by + 2 * pad_h, bx + 2 * pad_w
    ext_y0, ext_x0 = y0 - pad_h, x0 - pad_w
    
    predicted_clear_extended_planes = []
    for u in range(bz):
        z_idx = z0 + u
        coords_flat_ext, mask_ext = build_plane_coords_and_mask(
            z_index=z_idx, y_start=ext_y0, x_start=ext_x0, height=ext_h, width=ext_w,
            full_dims=(vz, vy, vx), device=device,
        )
        # 使用NGLOD进行推理
        pred_flat_ext = model.sdf(coords_flat_ext)
        pred_plane_ext = pred_flat_ext.view(1, 1, ext_h, ext_w)
        pred_plane_ext = pred_plane_ext * mask_ext.to(pred_plane_ext.dtype).unsqueeze(0).unsqueeze(0)
        predicted_clear_extended_planes.append(pred_plane_ext)

    # 对每个焦平面进行PSF卷积模拟
    simulated_focal_planes = []
    for focal_z in range(bz):  # 对每个z层都进行模拟
        simulated_focal_plane = torch.zeros((1, 1, by, bx), device=device)
        for u in range(bz):
            clear_extended_slice = predicted_clear_extended_planes[u]
            kernel = psf_kernels[abs(u - focal_z)]  # 使用距离对应的PSF核
            # Add dimensions to make kernel 4D: (out_channels, in_channels, height, width)
            kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, height, width)
            contribution = F.conv2d(clear_extended_slice, kernel, padding=0)
            simulated_focal_plane += contribution
        simulated_focal_planes.append(simulated_focal_plane)
    
    target_block_np = norm_volume_np[z0:z0+bz, y0:y0+by, x0:x0+bx]
    target_block = torch.from_numpy(target_block_np).to(device=device, dtype=torch.float32)
    
    simulated_block = torch.cat(simulated_focal_planes, dim=0)  # [bz, 1, by, bx]
    simulated_block = simulated_block.squeeze(1)  # [bz, by, bx]
    
    total_loss = F.mse_loss(simulated_block, target_block)
    
    writer.add_scalar("PSF_FT/Total_Loss_MSE_3D", total_loss.item(), global_step)
    return total_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume_tif", type=str, default="../data/Mouse_Heart_Angle0_patch.tif")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/nglod_angle_0.pth")  # 改为NGLOD检查点
    parser.add_argument("--psf_npy", type=str, default="psf_t0_v0.npy")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--block_z", type=int, default=25)
    parser.add_argument("--block_h", type=int, default=512)
    parser.add_argument("--block_w", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_path", type=str, default="checkpoints/nglod_psf_finetuned_decoder_only.pth")
    parser.add_argument("--logdir", type=str, default="runs/psf_finetune_nglod_decoder")
    parser.add_argument("--freeze_mlp", action="store_true", default=True)
    
    
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vol = tifffile.imread(args.volume_tif).astype(np.float32)
    vol_max = float(vol.max())
    vol_norm = vol / vol_max
    dz, dy, dx = vol_norm.shape
    print(f"Loaded volume {args.volume_tif} with shape (z,y,x) = {(dz, dy, dx)}. Normalized by max = {vol_max:.6f}")

    model, nglod_args = load_nglod_model_from_checkpoint(args.checkpoint, device, args.freeze_mlp)
    
    # 只对可训练参数创建优化器
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    print(f"优化器管理参数数量: {sum(p.numel() for p in trainable_params):,}")
    
    writer = SummaryWriter(log_dir=args.logdir)

    psf = np.load(args.psf_npy).astype(np.float32)
    if psf.ndim != 3:
        raise ValueError(f"PSF must be 3D, got shape {psf.shape}")
    psf /= np.sum(psf)
    psf_kernels = torch.from_numpy(psf).to(device)

    if psf_kernels.shape[0] != args.block_z:
        raise ValueError(f"PSF depth ({psf_kernels.shape[0]}) must match --block_z ({args.block_z})")

    block_shape = (args.block_z, args.block_h, args.block_w)
    pbar = tqdm(range(args.steps), desc="PSF fine-tune NGLOD")
    for step in pbar:
        optimizer.zero_grad(set_to_none=True)
        
        total_loss = psf_finetune_step(
            model=model,
            norm_volume_np=vol_norm,
            device=device,
            writer=writer,
            global_step=step,
            psf_kernels=psf_kernels,
            block_shape=block_shape,
        )
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)
        optimizer.step()
        
        if (step + 1) % 3 == 0:
            pbar.set_postfix(total_loss=f"{total_loss.item():.12f}")

    # Save new checkpoint
    save_obj = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'total_loss': float(total_loss.item()),
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
        'epoch': args.steps,
        'finetune_type': 'psf',  # 标记这是PSF微调的检查点
    }
    torch.save(save_obj, args.save_path)
    print(f"Saved PSF-finetuned NGLOD model to {args.save_path}")


if __name__ == "__main__":
    main()
