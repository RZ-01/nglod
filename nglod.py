import torch
import torch.optim as optim
import numpy as np
import tifffile
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import sys
import os
import time
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

def main():
    args = Args()
    
    vol = tifffile.imread('../data/Mouse_Heart_Angle0_patch.tif')
    print("Loaded volume")

    vol = vol.astype(np.float32)
    vol_max = float(vol.max())
    vol = vol / vol_max
    print("Normalized volume")
    dz, dy, dx = vol.shape
    n_samples = 600000  
    epochs_per_batch = 1000  
    threshold = 0.03

    T = np.array([dx-1, dy-1, dz-1], dtype=np.float32)  
    
    # SETUP
    device = torch.device('cuda')
    print('Using device:', device)
    
    model = OctreeSDF(args).to(device)
    
    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    feature_params = 0
    for i, feature_vol in enumerate(model.features):
        feature_params += sum(p.numel() for p in feature_vol.parameters())
        print(f"LOD {i} feature volume parameters: {sum(p.numel() for p in feature_vol.parameters()):,}")
    mlp_params = 0
    for i, decoder in enumerate(model.louts):
        mlp_params += sum(p.numel() for p in decoder.parameters())
        print(f"LOD {i} MLP decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")
    
    print(f"Total parameters: {total_params:,}")
    print(f"Octree feature parameters: {feature_params:,}")
    print(f"MLP decoder parameters: {mlp_params:,}")

    model.train()
    
    current_stage = 1 
    max_stage = args.num_lods
    
    opt = optim.Adam(model.parameters(), lr=args.lr)
    
    os.makedirs('checkpoints', exist_ok=True)

    total_epochs = args.epochs 
    current_epoch = 0
    batch_idx = 0
    
    lod_grow_interval = args.grow_every
    last_grow_epoch = 0 
    
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

        pbar = tqdm(total=epochs_per_batch, desc=f"Training batch {batch_idx + 1}", ncols=100, 
                    dynamic_ncols=True, leave=True, file=None, 
                    ascii=True, disable=False)
        start_time = time.time()

        for epoch in range(epochs_per_batch):
            if (current_epoch + epoch >= last_grow_epoch + lod_grow_interval and 
                current_stage < max_stage and 
                current_epoch + epoch > 0):
                current_stage += 1
                last_grow_epoch = current_epoch + epoch
                print(f"\nEpoch {current_epoch + epoch}: Growing to stage {current_stage} (training LODs: {get_loss_lods(current_stage)})")
            
            loss_lods = get_loss_lods(current_stage, args.growth_strategy)
            
            total_loss = 0
            for batch_coords, batch_val in loader:
                batch_coords = batch_coords.to(device)
                target = batch_val.to(device).unsqueeze(-1)
                
                preds = []
                if args.return_lst:
                    all_preds = model.sdf(batch_coords, return_lst=True)
                    preds = [all_preds[i] for i in loss_lods]
                else:
                    for lod in loss_lods:
                        preds.append(model.sdf(batch_coords, lod=lod))
                
                loss = 0
                _l1_loss = 0
                for pred in preds:
                    _l1_loss = torch.abs(pred - target).mean()  
                    loss += _l1_loss
                
                loss = loss / batch_coords.size(0)  

                opt.zero_grad()
                loss.backward()
                total_norm = clip_grad_norm_(model.parameters(), max_norm=50.0)

                opt.step()
                total_loss += loss.item() * batch_coords.size(0)   
            
                            
            elapsed_time = time.time() - start_time
            pbar.set_postfix({
                'Loss': f'{total_loss:.6f}',
                'Grad': f'{total_norm:.2e}',
                'Time': f'{elapsed_time:.1f}s'
            })
            pbar.update(1)

            writer.add_scalar("Loss/l1", total_loss, current_epoch + epoch)
            writer.add_scalar("Training/current_stage", current_stage, current_epoch + epoch)
            writer.add_scalar("Gradient/norm", total_norm, current_epoch + epoch)

        pbar.close()
        current_epoch += epochs_per_batch
        batch_idx += 1
        print(f"Completed batch {batch_idx}, total epochs: {current_epoch}")

    final_checkpoint = {
        'epoch': current_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'loss': total_loss,
        'args': args,
        'current_stage': current_stage,  
        'transformation': {
            'T': T,
        }
    }
    torch.save(final_checkpoint, 'checkpoints/nglod_angle_0.pth')
    print(f"Saved final model at epoch {current_epoch} with stage {current_stage}")

if __name__ == "__main__":
    main()
    writer.close()