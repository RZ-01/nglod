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
    def __init__(self):
        # ===== Network Architecture Parameters =====
        self.net = 'OctreeSDF'
        self.feature_dim = 32     
        self.feature_size = 4        
        self.num_layers = 1         
        self.hidden_dim = 512      
        self.input_dim = 3           
        
        # ===== LOD-related Parameters =====
        self.num_lods = 5           
        self.base_lod = 2            
        self.interpolate = None    
        self.growth_strategy = 'increase'  
        self.grow_every = -1       
        
        # ===== Positional Encoding Parameters =====
        self.pos_enc = False          
        self.ff_dim = -1                    
        
        # ===== Training Parameters =====
        self.epochs = 10000          
        self.batch_size = 100000       
        self.optimizer = 'adam'      
        self.lr = 6e-4         
        self.loss = ['l1_loss']    
        
        # ===== Other Parameters =====
        self.pos_invariant = False   
        self.joint_decoder = False   
        self.feat_sum = False        
        self.return_lst = True


def load_nglod_model_from_checkpoint(checkpoint_path, device, freeze_mlp=True):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'args' in ckpt:
        nglod_args = ckpt['args']
    else:
        raise ValueError("No nglod args found in checkpoint")
    
    model = OctreeSDF(nglod_args).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    
    if freeze_mlp:
        for name, param in model.named_parameters():
            if 'louts' in name:
                param.requires_grad = False
    
    return model, nglod_args


def set_trainable_parameters(model, mode, verbose=True):
   # Set the trainable parts of the model based on the mode.
   #     verbose (bool): Whether to print status information.
    if verbose:
        print(f"\n---> Switching training mode to '{mode}' <---")
        
    for name, param in model.named_parameters():
        if 'louts' in name: # MLP parameters
            param.requires_grad = (mode == 'mlp')
        elif 'features' in name and 'fm' in name:  # Feature grid parameters
            param.requires_grad = (mode == 'feature')
        else:
            param.requires_grad = False  # Freeze anything else

    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    
    
    return trainable_params_list


def compute_block_feature_mask(block_coords, block_shape, volume_shape, 
                               lod_idx, base_lod, pad_ratio=0.2):

    #Calculate the corresponding voxel range for the current block in the specified LOD's feature grid

    z0, y0, x0 = block_coords
    bz, by, bx = block_shape
    vz, vy, vx = volume_shape
    
    fsize = 2 ** (lod_idx + base_lod)
    
    z0_norm = (z0 / (vz - 1)) * 2.0 - 1.0
    y0_norm = (y0 / (vy - 1)) * 2.0 - 1.0
    x0_norm = (x0 / (vx - 1)) * 2.0 - 1.0
    
    z1_norm = ((z0 + bz - 1) / (vz - 1)) * 2.0 - 1.0
    y1_norm = ((y0 + by - 1) / (vy - 1)) * 2.0 - 1.0
    x1_norm = ((x0 + bx - 1) / (vx - 1)) * 2.0 - 1.0
    
    pad_z = (z1_norm - z0_norm) * pad_ratio
    pad_y = (y1_norm - y0_norm) * pad_ratio
    pad_x = (x1_norm - x0_norm) * pad_ratio
    
    z0_norm -= pad_z
    y0_norm -= pad_y
    x0_norm -= pad_x
    z1_norm += pad_z
    y1_norm += pad_y
    x1_norm += pad_x
    
    z0_feat = (z0_norm + 1.0) / 2.0 * fsize
    y0_feat = (y0_norm + 1.0) / 2.0 * fsize
    x0_feat = (x0_norm + 1.0) / 2.0 * fsize
    
    z1_feat = (z1_norm + 1.0) / 2.0 * fsize
    y1_feat = (y1_norm + 1.0) / 2.0 * fsize
    x1_feat = (x1_norm + 1.0) / 2.0 * fsize
    
    z0_idx = max(0, int(np.floor(z0_feat)))
    y0_idx = max(0, int(np.floor(y0_feat)))
    x0_idx = max(0, int(np.floor(x0_feat)))
    
    z1_idx = min(fsize, int(np.ceil(z1_feat)) + 1)
    y1_idx = min(fsize, int(np.ceil(y1_feat)) + 1)
    x1_idx = min(fsize, int(np.ceil(x1_feat)) + 1)
    
    mask = torch.zeros((1, 1, fsize+1, fsize+1, fsize+1), dtype=torch.bool)
    mask[0, 0, z0_idx:z1_idx, y0_idx:y1_idx, x0_idx:x1_idx] = True
    
    return mask


def register_gradient_masks(model, block_coords, block_shape, volume_shape, base_lod, device):
   # Register gradient masks for the model's feature parameters, allowing only features related to the current block to be updated
    hook_handles = []
    
    for lod_idx, feature_module in enumerate(model.features):
        mask = compute_block_feature_mask(
            block_coords, block_shape, volume_shape, 
            lod_idx, base_lod, pad_ratio=0.3
        ).to(device)
        
        def create_hook(mask_tensor):
            def hook(grad):
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
  #  Iterate through blocks in a fixed row-column order
    vz, vy, vx = volume_shape
    bz, by, bx = block_shape
    
    z_min, z_max = 200, 430
    z0_low = z_min
    z0_high_inclusive = max(z_min, z_max - bz + 1)
    
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
    
    ys_idx_all = torch.arange(ext_y0, ext_y0 + ext_h, device=device, dtype=torch.float32)
    xs_idx_all = torch.arange(ext_x0, ext_x0 + ext_w, device=device, dtype=torch.float32)
    valid_y_all = (ys_idx_all >= 0) & (ys_idx_all < vy)
    valid_x_all = (xs_idx_all >= 0) & (xs_idx_all < vx)
    mask_ext_all = valid_y_all[:, None] & valid_x_all[None, :]
    
    ys_n_all = ys_idx_all / (vy - 1) * 2.0 - 1.0
    xs_n_all = xs_idx_all / (vx - 1) * 2.0 - 1.0
    grid_y_all, grid_x_all = torch.meshgrid(ys_n_all, xs_n_all, indexing='ij')
    
    predicted_clear_extended_planes = []
    
    for u in range(bz):
        z_idx = z0 + u
        z_n_val = (float(z_idx) / (vz - 1)) * 2.0 - 1.0
        grid_z = torch.full((ext_h, ext_w), z_n_val, device=device)
        coords_flat = torch.stack([grid_x_all, grid_y_all, grid_z], dim=-1).view(-1, 3)
        
        pred_flat = model.sdf(coords_flat)
        pred_plane_ext = pred_flat.view(1, 1, ext_h, ext_w)
        pred_plane_ext = pred_plane_ext * mask_ext_all.to(pred_plane_ext.dtype).unsqueeze(0).unsqueeze(0)
        predicted_clear_extended_planes.append(pred_plane_ext)

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
    
    target_block_np = norm_volume_np[z0:z0+bz, y0:y0+by, x0:x0+bx]
    target_block = torch.from_numpy(target_block_np).to(device=device, dtype=torch.float32)
    simulated_block = torch.cat(simulated_focal_planes, dim=0).squeeze(1)
    loss = F.l1_loss(simulated_block, target_block)
    
    writer.add_scalar("PSF_FT_Alternating/Loss", loss.item(), global_step)
    
    return loss


def get_training_schedule(total_steps, switch_every, feature_boost_ratio=2.0):
    schedule = []
    step = 0
    mode = 'feature'  # Start with feature
    
    transition_point = int(total_steps * 0.7)
    
    while step < total_steps:
        if step < transition_point:
            next_step = min(step + switch_every, transition_point)
            schedule.append((step, next_step, mode))
            step = next_step
            mode = 'mlp' if mode == 'feature' else 'feature'
        else:
            if mode == 'feature':
                duration = int(switch_every * feature_boost_ratio)
                next_step = min(step + duration, total_steps)
                schedule.append((step, next_step, 'feature'))
                step = next_step
                mode = 'mlp'
            else:
                duration = switch_every
                next_step = min(step + duration, total_steps)
                schedule.append((step, next_step, 'mlp'))
                step = next_step
                mode = 'feature'
    
    return schedule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume_tif", type=str, default="../data/Mouse_Heart_Angle0_patch.tif")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/nglod_angle_0.pth")  
    parser.add_argument("--psf_npy", type=str, default="psf_t0_v0.npy")
    parser.add_argument("--steps", type=int, default=4200)
    parser.add_argument("--lr", type=float, default=6e-3)
    parser.add_argument("--switch_every", type=int, default=200, help="Steps to alternate between training feature and MLP")
    parser.add_argument("--feature_boost_ratio", type=float, default=2.0)
    parser.add_argument("--block_z", type=int, default=115)
    parser.add_argument("--block_h", type=int, default=266)
    parser.add_argument("--block_w", type=int, default=266)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_path", type=str, default="checkpoints/nglod_psf_large_new.pth")
    parser.add_argument("--logdir", type=str, default="runs/psf_finetune_large_new")
    parser.add_argument("--region_size", type=int, default=532)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vol = tifffile.imread(args.volume_tif).astype(np.float32)
    vol_max = float(vol.max())
    vol_norm = vol / vol_max
    
    model, nglod_args = load_nglod_model_from_checkpoint(args.checkpoint, device, freeze_mlp=False)
    
    total_steps = args.steps
    writer = SummaryWriter(log_dir=args.logdir)

    psf = np.load(args.psf_npy).astype(np.float32)
    psf = psf / (psf.sum() + 1e-12)
    psf_kernels = torch.from_numpy(psf).to(device)

    print(f"\nPSF shape: {psf_kernels.shape}")
    print(f"Block shape: ({args.block_z}, {args.block_h}, {args.block_w})")
    print(f"Volume shape: {vol_norm.shape}\n")
    
    block_shape = (args.block_z, args.block_h, args.block_w)
    
    schedule = get_training_schedule(total_steps, args.switch_every, args.feature_boost_ratio)
    
    print("Training Schedule:")
    print(f"  Phase 1 (steps 0-{int(total_steps*0.7)}): Regular alternating (1:1 ratio)")
    print(f"  Phase 2 (steps {int(total_steps*0.7)}-{total_steps}): Feature-focused ({args.feature_boost_ratio}:1 ratio)")
    
    feature_steps = sum(end - start for start, end, mode in schedule if mode == 'feature')
    mlp_steps = sum(end - start for start, end, mode in schedule if mode == 'mlp')
    print(f"  Total feature steps: {feature_steps} ({feature_steps/total_steps*100:.1f}%)")
    print(f"  Total MLP steps: {mlp_steps} ({mlp_steps/total_steps*100:.1f}%)\n")
    
    current_mode = schedule[0][2]
    trainable_params = set_trainable_parameters(model, current_mode)
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    
    pbar = tqdm(
        total=total_steps,
        desc="Weighted Feature Training",
        dynamic_ncols=True,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
    )
    
    schedule_idx = 0
    for step in range(total_steps):
        if step >= schedule[schedule_idx][1]:
            schedule_idx += 1
            if schedule_idx < len(schedule):
                new_mode = schedule[schedule_idx][2]
                if new_mode != current_mode:
                    current_mode = new_mode
                    trainable_params = set_trainable_parameters(model, current_mode, verbose=False)
                    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

        eta_min = 1e-5
        current_lr = eta_min + (args.lr - eta_min) * 0.5 * (1 + np.cos(np.pi * step / total_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        current_block_coords = sample_block_start_indices(
            vol_norm.shape, block_shape, args.region_size, step
        )
        
        hook_handles = []
        if current_mode == 'feature':
            hook_handles = register_gradient_masks(
                model, current_block_coords, block_shape, 
                vol_norm.shape, nglod_args.base_lod, device
            )
        
        optimizer.zero_grad(set_to_none=True)
        
        loss = psf_finetune_step(
            model=model, norm_volume_np=vol_norm, device=device, writer=writer,
            global_step=step, psf_kernels=psf_kernels, block_shape=block_shape,
            block_coords=current_block_coords
        )
        
        loss.backward()
        
        if current_mode == 'feature':
            remove_gradient_hooks(hook_handles)
        
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=30.0)
        optimizer.step()
        
        pbar.update(1)
        pbar.set_postfix({
            'mode': current_mode,
            'loss': f'{loss.item():.6f}',
            'lr': f'{current_lr:.2e}',
            'block': f'({current_block_coords[0]},{current_block_coords[1]},{current_block_coords[2]})'
        })

    pbar.close()

    save_obj = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'nglod_args': nglod_args,
    }
    torch.save(save_obj, args.save_path)
    print(f"\nSaved final PSF-finetuned model (weighted feature training) to {args.save_path}")


if __name__ == "__main__":
    main()