#!/usr/bin/env python3
"""
NGLOD高级PSF微调：按Octree Node渐进式训练
"""
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
    """NGLOD原始参数类"""
    def __init__(self):
        self.net = 'OctreeSDF'
        self.feature_dim = 32     
        self.feature_size = 4        
        self.num_layers = 1         
        self.hidden_dim = 512      
        self.input_dim = 3           
        self.num_lods = 5           
        self.base_lod = 2            
        self.interpolate = None    
        self.growth_strategy = 'increase'  
        self.grow_every = -1       
        self.pos_enc = False          
        self.ff_dim = -1                    
        self.epochs = 10000          
        self.batch_size = 100000       
        self.optimizer = 'adam'      
        self.lr = 6e-4         
        self.loss = ['l1_loss']    
        self.pos_invariant = False   
        self.joint_decoder = False   
        self.feat_sum = False        
        self.return_lst = True


class OctreeNodeTrainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.lod_info = {}
        self._analyze_nglod_structure()
        node_sizes = [2, 3, 4, 6, 8]  # 对应LOD 0-4的节点大小
        self.all_nodes = self._generate_all_nodes(node_sizes)
        
        print(f" {len(self.all_nodes)} octree nodes in total")
        
        lod_counts = {}
        for node in self.all_nodes:
            lod = node['lod']
            lod_counts[lod] = lod_counts.get(lod, 0) + 1
        
        for lod, count in lod_counts.items():
            print(f"LOD {lod}: {count} nodes")
    
    def _analyze_nglod_structure(self):
        
        for lod_idx in range(self.model.num_lods):
            feature_volume = self.model.features[lod_idx]
            fm_shape = feature_volume.fm.shape  # [1, fdim, fsize+1, fsize+1, fsize+1]
            
            # 计算实际的空间维度
            # 根据OctreeSDF源码，每个LOD的大小是: 2^(i+base_lod)
            spatial_size = 2**(lod_idx + self.model.args.base_lod)
            spatial_dims = [spatial_size, spatial_size, spatial_size]  # [z, y, x]
            
            self.lod_info[lod_idx] = {
                'feature_volume': feature_volume,
                'fm_shape': fm_shape,
                'spatial_dims': spatial_dims,
                'spatial_size': spatial_size,
                'total_voxels': spatial_size ** 3,
                'total_params': feature_volume.fm.numel()
            }

            print(f"LOD {lod_idx}: spatial size {spatial_size}³, params {self.lod_info[lod_idx]['total_params']:,}")

    def _generate_all_nodes(self, node_sizes):
        all_nodes = []
        
        for lod_idx in range(self.model.num_lods):
            if lod_idx < len(node_sizes):
                node_size = node_sizes[lod_idx]
            else:
                node_size = node_sizes[-1]
            
            lod_data = self.lod_info[lod_idx]
            spatial_size = lod_data['spatial_size']
            
            nodes_per_dim = max(1, spatial_size // node_size)
            
            for z_idx in range(nodes_per_dim):
                for y_idx in range(nodes_per_dim):
                    for x_idx in range(nodes_per_dim):
                        # 计算节点的空间范围
                        z_start = z_idx * node_size
                        z_end = min((z_idx + 1) * node_size, spatial_size)
                        y_start = y_idx * node_size
                        y_end = min((y_idx + 1) * node_size, spatial_size)
                        x_start = x_idx * node_size
                        x_end = min((x_idx + 1) * node_size, spatial_size)
                        
                        node_info = {
                            'lod': lod_idx,
                            'node_idx': (z_idx, y_idx, x_idx),
                            'spatial_range': {
                                'z': (z_start, z_end),
                                'y': (y_start, y_end),
                                'x': (x_start, x_end)
                            },
                            'actual_size': (z_end - z_start, y_end - y_start, x_end - x_start)
                        }
                        all_nodes.append(node_info)
        
        return all_nodes
    
    def freeze_all_features(self):
        for name, param in self.model.named_parameters():
            if 'features' in name:
                param.requires_grad = False
    
    def activate_node(self, node_idx):
        self.freeze_all_features()
        
        if node_idx >= len(self.all_nodes):
            raise ValueError(f"Invalid node index {node_idx}")
        
        node_info = self.all_nodes[node_idx]
        lod_level = node_info['lod']
        
        # 训练整个LOD级别，但可以通过loss masking来聚焦特定区域
        feature_volume = self.model.features[lod_level]
        for param in feature_volume.parameters():
            param.requires_grad = True
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())

        print(f"Activate Node {node_idx} (LOD {lod_level})")
        print(f"  Spatial Range: {node_info['spatial_range']}")
        print(f"  Actual Size: {node_info['actual_size']}")
        print(f"  Trainable Params: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")

        return node_info
    
    def get_node_spatial_mask(self, node_info, coords):
        lod_level = node_info['lod']
        spatial_range = node_info['spatial_range']
        
        lod_data = self.lod_info[lod_level]
        spatial_size = lod_data['spatial_size']
        
        voxel_coords = (coords + 1) * 0.5  # 转换到[0, 1]
        voxel_coords[:, 0] *= (spatial_size - 1)  # x
        voxel_coords[:, 1] *= (spatial_size - 1)  # y  
        voxel_coords[:, 2] *= (spatial_size - 1)  # z
        
        x_in = (voxel_coords[:, 0] >= spatial_range['x'][0]) & (voxel_coords[:, 0] < spatial_range['x'][1])
        y_in = (voxel_coords[:, 1] >= spatial_range['y'][0]) & (voxel_coords[:, 1] < spatial_range['y'][1])
        z_in = (voxel_coords[:, 2] >= spatial_range['z'][0]) & (voxel_coords[:, 2] < spatial_range['z'][1])
        
        mask = x_in & y_in & z_in
        return mask


def load_nglod_model_from_checkpoint(checkpoint_path, device):
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


def psf_finetune_step_with_node_focus(model: nn.Module, node_trainer: OctreeNodeTrainer,
                                    current_node_info, norm_volume_np: np.ndarray,
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
    total_coords_count = 0
    
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
        
        # 记录node内的坐标数量（用于调试）
        total_coords_count += coords_flat_ext.shape[0]

    # PSF卷积模拟
    simulated_focal_planes = []
    for focal_z in range(bz):
        simulated_focal_plane = torch.zeros((1, 1, by, bx), device=device)
        for u in range(bz):
            clear_extended_slice = predicted_clear_extended_planes[u]
            kernel = psf_kernels[abs(u - focal_z)]
            kernel = kernel.unsqueeze(0).unsqueeze(0)
            contribution = F.conv2d(clear_extended_slice, kernel, padding=0)
            simulated_focal_plane += contribution
        simulated_focal_planes.append(simulated_focal_plane)
    
    target_block_np = norm_volume_np[z0:z0+bz, y0:y0+by, x0:x0+bx]
    target_block = torch.from_numpy(target_block_np).to(device=device, dtype=torch.float32)
    
    simulated_block = torch.cat(simulated_focal_planes, dim=0)
    simulated_block = simulated_block.squeeze(1)
    
    coords_for_loss_mask = []
    for u in range(bz):
        z_idx = z0+u
        coords_inner, _ = build_plane_coords_and_mask(
            z_index=z_idx, y_start=y0, x_start=x0, height=by, width=bx,
            full_dims=(vz, vy, vx), device=device,
        )
        coords_for_loss_mask.append(coords_inner)
        
    all_inner_coords = torch.cat(coords_for_loss_mask, dim=0)

    node_mask = node_trainer.get_node_spatial_mask(current_node_info, all_inner_coords)
    node_mask = node_mask.view(bz, by, bx) # 将1D掩码重塑为block的3D形状

    error = simulated_block - target_block
    masked_error = error * node_mask.float() # 应用掩码
        
    if node_mask.sum() > 0:
        total_loss = (masked_error**2).sum() / node_mask.sum()
    else:
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        
    num_masked_pixels = node_mask.sum().item()
    total_pixels = node_mask.numel()
    mask_ratio = num_masked_pixels / total_pixels
        
    writer.add_scalar("PSF_FT/Total_Loss_MSE_3D", total_loss.item(), global_step)
    writer.add_scalar(f"PSF_FT/Node_LOD{current_node_info['lod']}_Loss", total_loss.item(), global_step)
    writer.add_scalar(f"PSF_FT/Node_{current_node_info['lod']}_Mask_Ratio", mask_ratio, global_step)
        
    return total_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume_tif", type=str, default="../data/Mouse_Heart_Angle0_patch.tif")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/nglod_angle_0.pth")
    parser.add_argument("--psf_npy", type=str, default="psf_t0_v0.npy")
    parser.add_argument("--steps_per_node", type=int, default=100, help="epochs per node")
    parser.add_argument("--max_nodes", type=int, default=50, help="maximum number of nodes to train")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--block_z", type=int, default=25)
    parser.add_argument("--block_h", type=int, default=128)
    parser.add_argument("--block_w", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_path", type=str, default="checkpoints/nglod_psf_octree_finetuned.pth")
    parser.add_argument("--logdir", type=str, default="runs/psf_finetune_octree")
    
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    vol = tifffile.imread(args.volume_tif).astype(np.float32)
    vol_max = float(vol.max())
    vol_norm = vol / vol_max
    dz, dy, dx = vol_norm.shape
    print(f"Loaded volume {args.volume_tif} with shape (z,y,x) = {(dz, dy, dx)}. Normalized by max = {vol_max:.6f}")

    # 加载模型
    model, nglod_args = load_nglod_model_from_checkpoint(args.checkpoint, device)
    
    # 创建octree节点训练器
    node_trainer = OctreeNodeTrainer(model, device)
    
    # 限制训练的节点数量
    max_nodes = min(args.max_nodes, len(node_trainer.all_nodes))
    
    writer = SummaryWriter(log_dir=args.logdir)

    # 加载PSF
    psf = np.load(args.psf_npy).astype(np.float32)
    if psf.ndim != 3:
        raise ValueError(f"PSF must be 3D, got shape {psf.shape}")
    psf /= np.sum(psf)
    psf_kernels = torch.from_numpy(psf).to(device)

    if psf_kernels.shape[0] != args.block_z:
        raise ValueError(f"PSF depth ({psf_kernels.shape[0]}) must match --block_z ({args.block_z})")

    block_shape = (args.block_z, args.block_h, args.block_w)

    print(f"training {max_nodes} nodes, each one {args.steps_per_node} steps")

    global_step = 0
    
    # 逐个训练每个node
    for node_idx in range(max_nodes):
        print(f"\n--- training Node {node_idx}/{max_nodes-1} ---")
        
        current_node_info = node_trainer.activate_node(node_idx)
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
        
        node_pbar = tqdm(range(args.steps_per_node), 
                        desc=f"Node {node_idx} LOD{current_node_info['lod']}")
        
        for step in range(args.steps_per_node):
            optimizer.zero_grad(set_to_none=True)
            
            total_loss = psf_finetune_step_with_node_focus(
                model=model,
                node_trainer=node_trainer,
                current_node_info=current_node_info,
                norm_volume_np=vol_norm,
                device=device,
                writer=writer,
                global_step=global_step,
                psf_kernels=psf_kernels,
                block_shape=block_shape,
            )
            
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)
            optimizer.step()
            
            if step % 10 == 0:
                node_pbar.set_postfix({
                    'loss': f"{total_loss.item():.2e}",
                    'grad': f"{grad_norm:.2e}"
                })
            node_pbar.update(1)
            global_step += 1
        
        node_pbar.close()

    save_obj = {
        'model_state_dict': model.state_dict(),
        'nglod_args': nglod_args,
        'psf_finetune_config': {
            'method': 'octree_progressive',
            'lr': args.lr,
            'steps_per_node': args.steps_per_node,
            'max_nodes': max_nodes,
            'total_steps': global_step,
            'block_z': args.block_z,
            'block_h': args.block_h,
            'block_w': args.block_w,
            'psf_npy': args.psf_npy,
            'volume_tif': args.volume_tif,
        },
        'total_steps': global_step,
        'finetune_type': 'psf_octree_progressive',
    }
    torch.save(save_obj, args.save_path)
    
    print(f"saved to {args.save_path}")


if __name__ == "__main__":
    main()
