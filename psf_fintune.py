import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

torch.set_float32_matmul_precision('high')

class PosEncoding(nn.Module):
    def __init__(self, L: int = 6):
        super().__init__()
        self.L = L

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for i in range(self.L):
            for fn in (torch.sin, torch.cos):
                features.append(fn((2.0 ** i) * np.pi * x))
        return torch.cat(features, dim=-1)


class SimpleMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512, n_layers: int = 6):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.net(x))


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

def psf_finetune_step(model: nn.Module, pos_encoder: PosEncoding, norm_volume_np: np.ndarray,
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
    
    # 预测整个3D块的清晰体积（扩展版本）
    predicted_clear_extended_planes = []
    for u in range(bz):
        z_idx = z0 + u
        coords_flat_ext, mask_ext = build_plane_coords_and_mask(
            z_index=z_idx, y_start=ext_y0, x_start=ext_x0, height=ext_h, width=ext_w,
            full_dims=(vz, vy, vx), device=device,
        )
        # 内联推理逻辑
        enc_coords = pos_encoder(coords_flat_ext)
        pred_flat_ext = model(enc_coords)
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
    
    # 准备目标3D块
    target_block_np = norm_volume_np[z0:z0+bz, y0:y0+by, x0:x0+bx]
    target_block = torch.from_numpy(target_block_np).to(device=device, dtype=torch.float32)
    
    # 将模拟的焦平面堆叠成3D张量
    simulated_block = torch.cat(simulated_focal_planes, dim=0)  # [bz, 1, by, bx]
    simulated_block = simulated_block.squeeze(1)  # [bz, by, bx]
    
    # 计算3D块的MSE loss
    total_loss = F.mse_loss(simulated_block, target_block)
    
    writer.add_scalar("PSF_FT/Total_Loss_MSE_3D", total_loss.item(), global_step)
    return total_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume_tif", type=str, default="./data/Mouse_Heart_Angle0_patch.tif")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/new_model_1_angle_0.pth")
    parser.add_argument("--psf_npy", type=str, default="psf_t0_v0.npy")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--block_z", type=int, default=25)
    parser.add_argument("--block_h", type=int, default=256)
    parser.add_argument("--block_w", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_path", type=str, default="checkpoints/new_model_psf_finetuned.pth")
    parser.add_argument("--logdir", type=str, default="runs/psf_finetune")
    #parser.add_argument("--chunk_size", type=int, default=64, help="Chunk size for model inference to save memory.")
    
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vol = tifffile.imread(args.volume_tif).astype(np.float32)
    vol_max = float(vol.max())
    vol_norm = vol / vol_max
    dz, dy, dx = vol_norm.shape
    print(f"Loaded volume {args.volume_tif} with shape (z,y,x) = {(dz, dy, dx)}. Normalized by max = {vol_max:.6f}")

    pe = PosEncoding(L=6).to(device)
    in_dim = 3 + 2 * 6 * 3
    mlp = SimpleMLP(in_dim=in_dim).to(device)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    mlp.load_state_dict(ckpt['model_state_dict'])
    
    optimizer = torch.optim.Adam(mlp.parameters(), lr=args.lr)
    writer = SummaryWriter(log_dir=args.logdir)

    psf = np.load(args.psf_npy).astype(np.float32)
    if psf.ndim != 3:
        raise ValueError(f"PSF must be 3D, got shape {psf.shape}")
    psf /= np.sum(psf)
    psf_kernels = torch.from_numpy(psf).to(device)

    if psf_kernels.shape[0] != args.block_z:
        raise ValueError(f"PSF depth ({psf_kernels.shape[0]}) must match --block_z ({args.block_z})")

    # print(f"Loss weights: lambda_tv={args.lambda_tv}, lambda_hf={args.lambda_hf}")
    block_shape = (args.block_z, args.block_h, args.block_w)
    pbar = tqdm(range(args.steps), desc="PSF fine-tune")
    for step in pbar:
        optimizer.zero_grad(set_to_none=True)
        
        total_loss = psf_finetune_step(
            model=mlp,
            pos_encoder=pe,
            norm_volume_np=vol_norm,
            device=device,
            writer=writer,
            global_step=step,
            psf_kernels=psf_kernels,
            block_shape=block_shape,
            #chunk_size=args.chunk_size,
        )
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=50.0)
        optimizer.step()
        
        if (step + 1) % 10 == 0:
            pbar.set_postfix(total_loss=f"{total_loss.item():.6f}")

    # Save new checkpoint
    save_obj = {
        'model_state_dict': mlp.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'position_encoding': pe.state_dict(),
        'total_loss': float(total_loss.item())
    }
    torch.save(save_obj, args.save_path)
    print(f"Saved PSF-finetuned model to {args.save_path}")


if __name__ == "__main__":
    main()
