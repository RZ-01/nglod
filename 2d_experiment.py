import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import convolve
import torch.nn.functional as F
from tqdm import tqdm
import time
import sys
import os
torch.set_float32_matmul_precision('high')

sys.path.append(os.path.join(os.path.dirname(__file__), 'sdf-net'))
from lib.models.OctreeSDF import OctreeSDF
torch.manual_seed(42)
np.random.seed(42)

def download_simple_image():
    img = Image.open('original.png')
    img = img.convert('L')
    return np.array(img) / 255.0

def create_psf(size=25, sigma=3.5):
    from scipy.special import j1  
    
    psf = np.zeros((size, size))
    center = size // 2
    
    k = 2 * np.pi / sigma
    
    for i in range(size):
        for j in range(size):
            r = np.sqrt((i - center)**2 + (j - center)**2)
            
            if r == 0:
                psf[i, j] = 1.0
            else:
                kr = k * r
                if kr > 0:
                    psf[i, j] = (2 * j1(kr) / kr) ** 2
                else:
                    psf[i, j] = 1.0
    
    psf = psf / psf.sum()
    return psf

def create_G_psf(size=15, sigma=2.0):
    psf = np.zeros((size, size))
    center = size // 2
    
    for i in range(size):
        for j in range(size):
            psf[i, j] = np.exp(-((i-center)**2 + (j-center)**2) / (2*sigma**2))
    
    psf = psf / psf.sum()
    return psf

class NGLODArgs:
    def __init__(self):
        # ===== 网络架构参数 =====
        self.net = 'OctreeSDF'
        self.input_dim = 3           # 输入维度
        self.feature_dim = 32        # 特征维度
        self.feature_size = 4        # 特征网格大小
        self.hidden_dim = 128        # 隐藏层维度
        self.num_layers = 2          # MLP层数
        
        # ===== LOD相关参数 =====
        self.num_lods = 7            # LOD级别数量
        self.base_lod = 2            # 基础LOD级别
        self.interpolate = None      # LOD插值（None或0.0-1.0）
        
        # ===== 位置编码参数 =====
        self.pos_enc = False         # 是否使用位置编码 (True时输入维度变为39)
        self.ff_dim = 0              # Fourier特征维度（0表示关闭，>0时优先于pos_enc）
        self.ff_width = 16.0         # Fourier特征宽度 (控制频率范围)
        
        # ===== 训练参数 =====
        self.epochs = 2000
        self.grow_every = -1        # 每隔多少epoch增长LOD
        self.growth_strategy = 'increase'  # LOD增长策略
        self.lr = 2e-4
        
        # ===== 其他参数 =====
        self.pos_invariant = False   # 位置不变性
        self.joint_decoder = False   # 联合解码器
        self.feat_sum = False        # 特征求和
        self.return_lst = True       # 返回所有LOD预测
        self.optimizer = 'adam'      # 优化器

def sample_random_patch(H: int, W: int, patch_h: int, patch_w: int):
    y0 = np.random.randint(0, max(1, H - patch_h + 1))
    x0 = np.random.randint(0, max(1, W - patch_w + 1))
    return y0, x0

def build_extended_patch_coords(y0_ext: int, x0_ext: int, ext_h: int, ext_w: int, H: int, W: int, device: torch.device) -> torch.Tensor:
    ys_ext = torch.linspace(-1.0 + 2.0 * y0_ext / (H - 1), -1.0 + 2.0 * (y0_ext + ext_h - 1) / (H - 1), steps=ext_h, device=device)
    xs_ext = torch.linspace(-1.0 + 2.0 * x0_ext / (W - 1), -1.0 + 2.0 * (x0_ext + ext_w - 1) / (W - 1), steps=ext_w, device=device)
    grid_y, grid_x = torch.meshgrid(ys_ext, xs_ext, indexing='ij')
    coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), torch.zeros_like(grid_x.reshape(-1))], dim=-1)
    return coords

def build_image_coords(height: int, width: int, device: torch.device) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, steps=height, device=device)
    xs = torch.linspace(-1.0, 1.0, steps=width, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), torch.zeros_like(grid_x.reshape(-1))], dim=-1)  # [H*W, 3]
    return coords

def apply_psf_torch(image, psf, device=None):
    if len(image.shape) == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif len(image.shape) == 3:
        image = image.unsqueeze(1)
    
    if device is None:
        device = image.device

    psf_tensor = torch.from_numpy(psf).float().unsqueeze(0).unsqueeze(0).to(device)
    
    image = image.to(device)
    
    pad = psf.shape[0] // 2
    blurred = F.conv2d(image, psf_tensor, padding=pad)
    
    return blurred

def train_deblur_nglod_patch(clear_img, blurred_img, psf):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    H, W = clear_img.shape
    
    blurred_tensor = torch.from_numpy(blurred_img).float().to(device)

    args = NGLODArgs()
    model = OctreeSDF(args).to(device)
    
    # === 打印参数信息 ===
    total_params = sum(p.numel() for p in model.parameters())
    feat_params = 0
    mlp_params = 0
    for i, f in enumerate(model.features):
        fp = sum(p.numel() for p in f.parameters())
        feat_params += fp
    for i, dec in enumerate(model.louts):
        dp = sum(p.numel() for p in dec.parameters())
        mlp_params += dp
    print(f"Total params: {total_params:,} | Feature: {feat_params:,} | Decoder: {mlp_params:,} | Feature%: {feat_params/total_params*100:.1f}%")
    # ================================
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, fused=True)

    psf_kernel = torch.from_numpy(psf).float().unsqueeze(0).unsqueeze(0).to(device)
    psf_pad = psf.shape[0] // 2
    
    patch_size = 128  
    all_losses = []
    
    pbar = tqdm(total=args.epochs, desc="Training NGLOD ", ncols=100, 
                dynamic_ncols=True, leave=True, file=None, 
                ascii=True, disable=False)
    start_time = time.time()

    for epoch in range(args.epochs):
        patch_h = patch_w = min(patch_size, min(H, W))
        y0, x0 = sample_random_patch(H, W, patch_h, patch_w)
        
        y0_ext = max(0, y0 - psf_pad)
        x0_ext = max(0, x0 - psf_pad)
        y1_ext = min(H, y0 + patch_h + psf_pad)
        x1_ext = min(W, x0 + patch_w + psf_pad)
        ext_h = y1_ext - y0_ext
        ext_w = x1_ext - x0_ext
        
        y_core0 = y0 - y0_ext
        x_core0 = x0 - x0_ext
        
        coords_ext = build_extended_patch_coords(y0_ext, x0_ext, ext_h, ext_w, H, W, device)
        
        target_blurred_patch = blurred_tensor[y0:y0+patch_h, x0:x0+patch_w].unsqueeze(0).unsqueeze(0)
        
        pred_flat = model.sdf(coords_ext, return_lst=False)  # [ext_h*ext_w, 1]
        predicted_clear_ext = pred_flat.view(1, 1, ext_h, ext_w)
        
        predicted_blurred_ext = F.conv2d(predicted_clear_ext, psf_kernel, padding=psf_pad)
        
        predicted_blurred_patch = predicted_blurred_ext[:, :, y_core0:y_core0+patch_h, x_core0:x_core0+patch_w]
        
        loss = nn.MSELoss()(predicted_blurred_patch, target_blurred_patch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        all_losses.append(current_loss)
        
        elapsed_time = time.time() - start_time
        pbar.set_postfix({
            'Loss': f'{current_loss:.6f}',
            'Patch': f'[{y0}:{y0+patch_h},{x0}:{x0+patch_w}]',
            'Time': f'{elapsed_time:.1f}s'
        })
        pbar.update(1)
    
    pbar.close()
    
    return model, all_losses

def train_deblur_nglod(clear_img, blurred_img, psf):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    H, W = clear_img.shape
    blurred_tensor = torch.from_numpy(blurred_img).float().unsqueeze(0).unsqueeze(0).to(device)

    args = NGLODArgs()

    model = OctreeSDF(args).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, fused=True)

    psf_kernel = torch.from_numpy(psf).float().unsqueeze(0).unsqueeze(0).to(device)
    psf_pad = psf.shape[0] // 2
    
    coords_full = build_image_coords(H, W, device)
    
    all_losses = []
    
    pbar = tqdm(total=args.epochs, desc="Training NGLOD", ncols=100, 
                dynamic_ncols=True, leave=True, file=None, 
                ascii=True, disable=False)
    start_time = time.time()

    for epoch in range(args.epochs):

        pred_flat = model.sdf(coords_full, return_lst=False)  # [H*W, 1]
       
        predicted_clear = pred_flat.view(1, 1, H, W)
        
        predicted_blurred = F.conv2d(predicted_clear, psf_kernel, padding=psf_pad)
        
        loss = nn.MSELoss()(predicted_blurred, blurred_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        all_losses.append(current_loss)
        
        elapsed_time = time.time() - start_time
        pbar.set_postfix({
            'Loss': f'{current_loss:.6f}',
            'Time': f'{elapsed_time:.1f}s'
        })
        pbar.update(1)
    
    pbar.close()
    
    return model, all_losses

def main():
    clear_img = download_simple_image()
    
    psf = create_G_psf(size=15, sigma=2.0)  
    print(f"PSF: Gaussian, shape: {psf.shape}, sigma: 2.0")

    blurred_img = convolve(clear_img, psf, mode='constant')
    
    blur_strength = np.std(clear_img) / np.std(blurred_img) if np.std(blurred_img) > 0 else 1
    

    model, losses = train_deblur_nglod_patch(clear_img, blurred_img, psf)
    
    device = next(model.parameters()).device
    H, W = clear_img.shape
    
    with torch.no_grad():
        coords = build_image_coords(H, W, device)   
        pred_flat = model.sdf(coords, return_lst=False)  # [H*W, 1]
             
        deblurred_tensor = pred_flat.view(1, 1, H, W)
        deblurred_img = deblurred_tensor.cpu().numpy().squeeze()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(clear_img, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Original Clear Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(blurred_img, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Blurred Image')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(deblurred_img, cmap='gray', vmin=0, vmax=1)
    model_type = "NGLOD (Patch-based)" 
    axes[0, 2].set_title(f'{model_type} Deblurred Result')
    axes[0, 2].axis('off')
    
    im = axes[1, 0].imshow(psf, cmap='hot')
    axes[1, 0].set_title(f'PSF (Gaussian)\nSize: {psf.shape[0]}x{psf.shape[0]}')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], shrink=0.6)
    
    axes[1, 1].plot(losses)
    axes[1, 1].set_title('Training Loss Curve (Patch-based)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MSE Loss')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    mse = np.mean((clear_img - deblurred_img) ** 2)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    
    axes[1, 2].text(0.1, 0.5, f'Model: {model_type}\nPSNR: {psnr:.2f} dB\nFinal Loss: {losses[-1]:.6f}\nTraining: Patch-based (128x128)', 
                    fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

    return model, clear_img, blurred_img, deblurred_img

if __name__ == "__main__":
    model, clear_img, blurred_img, deblurred_img = main()