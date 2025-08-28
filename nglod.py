import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tifffile
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'sdf-net'))
from lib.models.OctreeSDF import OctreeSDF

writer = SummaryWriter(log_dir="runs/exp_new_1")  

class VolumeDataset(Dataset):
    def __init__(self, coords, volume, n_per_batch=10000, thresh=0.2):
        self.coords = coords.reshape(-1, 3).astype(np.float32)
        self.values = volume.reshape(-1).astype(np.float32)
        self.thresh = thresh
        self.n_per_batch = n_per_batch
        
        self.fg_idx = np.where(self.values >= self.thresh)[0]
        self.bg_idx = np.where(self.values < self.thresh)[0]
        print(f"foreground: {len(self.fg_idx)} points")
        print(f"background: {len(self.bg_idx)} points")

        self.n_fg = min(int(self.n_per_batch * 0.5), len(self.fg_idx))  
        self.n_bg = min(self.n_per_batch - self.n_fg, len(self.bg_idx))  

        if self.n_bg < (self.n_per_batch - self.n_fg):
            self.n_fg = min(self.n_per_batch - self.n_bg, len(self.fg_idx))
        
    def __len__(self):
        return 1
        
    def __getitem__(self, idx):
        if len(self.fg_idx) == 0 or len(self.bg_idx) == 0:
            idxs = np.random.choice(len(self.coords), self.n_per_batch, replace=False)
        else:
            idx_fg = np.random.choice(self.fg_idx, self.n_fg, replace=len(self.fg_idx) < self.n_fg)
            idx_bg = np.random.choice(self.bg_idx, self.n_bg, replace=len(self.bg_idx) < self.n_bg)
            idxs = np.concatenate([idx_fg, idx_bg])

        coords_batch = self.coords[idxs]
        values_batch = self.values[idxs]
        return coords_batch, values_batch

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
        self.interpolate = 'linear'      # LOD插值值
        
        self.epochs = 10000          
        self.batch_size = 100000        # 保持大批量训练
        self.grow_every = 1000       # 增加LOD增长间隔：1000→1500，让每个LOD训练更充分
        self.growth_strategy = 'increase'  
        
        self.optimizer = 'adam'      
        self.lr = 1e-2           
        self.loss = ['l1_loss']    # 使用L1损失
        self.return_lst = True      

def main():
    # 配置并实例化 NGLOD 模型
    args = Args()
    
    vol = tifffile.imread('Mouse_Heart_Angle0_patch.tif')
    print("Loaded volume")

    # Keep original max for de-normalization in PSF fine-tuning
    vol = vol.astype(np.float32)
    vol_max = float(vol.max())
    vol = vol / vol_max
    print("Normalized volume")
    dz, dy, dx = vol.shape
    n_samples = 600000  
    epochs_per_batch = 2000  
    threshold = 0.02

    T = np.array([dx-1, dy-1, dz-1], dtype=np.float32)  
    
    # SETUP
    device = torch.device('cuda')
    print('Using device:', device)
    
    model = OctreeSDF(args).to(device)
    
    # 设置训练模式
    model.train()
    
    # 初始化LOD训练策略 - 这是NGLOD的关键！
    current_stage = 1  # 从stage 1开始（对应LOD 0）
    max_stage = args.num_lods
    
    # TRAIN - 使用args中的参数
    opt = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-4)
    
    # 创建checkpoints目录
    os.makedirs('checkpoints', exist_ok=True)

    total_epochs = args.epochs  # 使用args中的epochs
    current_epoch = 0
    batch_idx = 0
    
    # LOD增长策略：使用args中的grow_every参数
    lod_grow_interval = args.grow_every
    last_grow_epoch = 0  # 记录上次增长LOD的epoch
    
    # 初始化loss_lods（官方实现的核心概念）
    def get_loss_lods(stage, strategy='increase'):
        if strategy == 'increase':
            return list(range(0, stage))
        elif strategy == 'onebyone':
            return [stage-1]
        else:
            return list(range(0, args.num_lods))

    while current_epoch < total_epochs:
        total_voxels = dz * dy * dx
        idxs = np.random.choice(total_voxels, n_samples, replace=False)
        zs, ys, xs = np.unravel_index(idxs, (dz, dy, dx))

        print(f"Batch {batch_idx + 1}: Randomly sampled {n_samples} points from entire volume")

        xs_n = xs / (dx - 1) * 2 - 1
        ys_n = ys / (dy - 1) * 2 - 1
        zs_n = zs / (dz - 1) * 2 - 1

        coords = np.stack([xs_n, ys_n, zs_n], axis=-1).astype(np.float32)
        values = vol[zs, ys, xs].astype(np.float32)
        
        print(f"Batch data: coords range [{coords.min():.3f}, {coords.max():.3f}], values range [{values.min():.6f}, {values.max():.6f}]")

        dataset = VolumeDataset(coords, values, n_per_batch=args.batch_size, thresh=threshold)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

        for epoch in tqdm(range(epochs_per_batch), desc=f"Training batch {batch_idx + 1}"):
            # 检查是否需要增长LOD级别 - 确保只在达到指定间隔时增长一次
            if (current_epoch + epoch >= last_grow_epoch + lod_grow_interval and 
                current_stage < max_stage and 
                current_epoch + epoch > 0):
                current_stage += 1
                last_grow_epoch = current_epoch + epoch
                print(f"\nEpoch {current_epoch + epoch}: Growing to stage {current_stage} (training LODs: {get_loss_lods(current_stage)})")
                opt = optim.Adam(model.parameters(), lr=args.lr)
                remaining_steps = total_epochs - (current_epoch + epoch)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=remaining_steps, eta_min=1e-4)
            
            # 获取当前要训练的LOD级别
            loss_lods = get_loss_lods(current_stage, args.growth_strategy)
            
            total_loss = 0
            for batch_coords, batch_val in loader:
                batch_coords = batch_coords.to(device)
                target = batch_val.to(device).unsqueeze(-1)
                
                # NGLOD官方方式：获取所有LOD级别的预测
                preds = []
                if args.return_lst:
                    # 方式1：一次性返回所有预测
                    all_preds = model.sdf(batch_coords, return_lst=True)
                    preds = [all_preds[i] for i in loss_lods]
                else:
                    # 方式2：分别计算每个LOD级别
                    for lod in loss_lods:
                        preds.append(model.sdf(batch_coords, lod=lod))
                
                # 计算损失（使用L1损失）
                loss = 0
                _l1_loss = 0
                for pred in preds:
                    _l1_loss = torch.abs(pred - target).mean()  # L1损失
                    loss += _l1_loss
                
                loss = loss / batch_coords.size(0)  # 除以batch_size

                opt.zero_grad()
                loss.backward()
                total_norm = clip_grad_norm_(model.parameters(), max_norm=10.0)

                opt.step()
                total_loss += loss.item() * batch_coords.size(0)   
                
            if (epoch + 1) % 10 == 0: 
                print(f"Epoch {current_epoch + epoch + 1}: l1 loss = {total_loss:.6f}, Stage = {current_stage}, LODs = {loss_lods}, Gradient norm: {total_norm:.8e}")
            writer.add_scalar("Loss/l1", total_loss, current_epoch + epoch)
            writer.add_scalar("Training/current_stage", current_stage, current_epoch + epoch)
            writer.add_scalar("Gradient/norm", total_norm, current_epoch + epoch)
            writer.add_scalar("Learning/rate", scheduler.get_last_lr()[0], current_epoch + epoch)
            scheduler.step()

        current_epoch += epochs_per_batch
        batch_idx += 1
        print(f"Completed batch {batch_idx}, total epochs: {current_epoch}")

    final_checkpoint = {
        'epoch': current_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'loss': total_loss,
        'args': args,
        'current_stage': current_stage,  # 保存当前训练阶段
        'transformation': {
            'T': T,
        }
    }
    torch.save(final_checkpoint, 'checkpoints/nglod_angle_0.pth')
    print(f"Saved final model at epoch {current_epoch} with stage {current_stage}")

if __name__ == "__main__":
    main()
    writer.close()