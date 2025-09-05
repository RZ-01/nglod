import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'sdf-net'))
from lib.models.OctreeSDF import OctreeSDF

class Image2DDataset(Dataset):
    """简单的2D图像数据集"""
    def __init__(self, image_array, batch_size=10000):
        self.height, self.width = image_array.shape[:2]
        self.batch_size = batch_size
        
        # 创建归一化的坐标网格 [-1, 1]
        y_coords, x_coords = np.mgrid[0:self.height, 0:self.width]
        self.x_norm = (x_coords / (self.width - 1)) * 2 - 1    # [0, W-1] -> [-1, 1]
        self.y_norm = (y_coords / (self.height - 1)) * 2 - 1   # [0, H-1] -> [-1, 1]
        
        # 将坐标和像素值展平
        self.coords = np.stack([
            self.x_norm.flatten(),
            self.y_norm.flatten(),
            np.zeros(self.height * self.width)  # Z坐标设为0（2D转3D）
        ], axis=-1).astype(np.float32)
        
        if len(image_array.shape) == 3:  # RGB图像
            self.values = image_array.reshape(-1, 3).astype(np.float32) / 255.0
        else:  # 灰度图像
            self.values = image_array.flatten().astype(np.float32) / 255.0
            
        print(f"Dataset: {len(self.coords)} pixels, coords range [{self.coords.min():.3f}, {self.coords.max():.3f}]")
        print(f"Values range [{self.values.min():.3f}, {self.values.max():.3f}]")
    
    def __len__(self):
        return (len(self.coords) + self.batch_size - 1) // self.batch_size
    
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.coords))
        
        coords_batch = self.coords[start_idx:end_idx]
        values_batch = self.values[start_idx:end_idx]
        
        return coords_batch, values_batch

class Args2D:
    """NGLOD 2D图像拟合的参数配置"""
    def __init__(self):
        # ===== 网络架构参数 =====
        self.net = 'OctreeSDF'
        self.input_dim = 3           # 输入维度（x, y, z=0）
        self.feature_dim = 32        # 特征维度，控制表达能力
        self.feature_size = 4        # 特征网格大小
        self.hidden_dim = 128        # 隐藏层维度
        self.num_layers = 2          # MLP层数
        
        # ===== LOD相关参数 =====
        self.num_lods = 4            # LOD级别数量（从粗到细）
        self.base_lod = 2            # 基础LOD级别
        self.interpolate = None      # LOD插值（None或0.0-1.0）
        
        # ===== 位置编码参数 =====
        self.pos_enc = False         # 是否使用位置编码 (True时输入维度变为39)
        self.ff_dim = 0              # Fourier特征维度（0表示关闭，>0时优先于pos_enc）
        self.ff_width = 16.0         # Fourier特征宽度 (控制频率范围)
        
        # ===== 训练参数 =====
        self.epochs = 800           # 总训练轮数
        self.batch_size = 4096       # 批量大小
        self.lr = 2e-3              # 学习率
        self.grow_every = -1        # 每隔多少epoch增长LOD
        self.growth_strategy = 'increase'  # LOD增长策略
        
        # ===== 其他参数 =====
        self.pos_invariant = False   # 位置不变性
        self.joint_decoder = False   # 联合解码器
        self.feat_sum = False        # 特征求和
        self.return_lst = True       # 返回所有LOD预测
        self.loss = ['l1_loss']      # 损失函数
        self.optimizer = 'adam'      # 优化器

def visualize_reconstruction(model, original_image, device, epoch, save_path=None):
    """可视化重建结果"""
    model.eval()
    
    height, width = original_image.shape[:2]
    
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    x_norm = (x_coords / (width - 1)) * 2 - 1
    y_norm = (y_coords / (height - 1)) * 2 - 1
    
    coords = np.stack([
        x_norm.flatten(),
        y_norm.flatten(), 
        np.zeros(height * width)
    ], axis=-1).astype(np.float32)
    
    coords_tensor = torch.from_numpy(coords).to(device)
    
    # 一次性预测
    with torch.no_grad():
        reconstructed = model.sdf(coords_tensor, return_lst=False).cpu().numpy()
    
    # 重塑为图像形状
    if len(original_image.shape) == 3:  # RGB
        reconstructed_image = reconstructed.reshape(height, width, 3)
    else:  # 灰度
        reconstructed_image = reconstructed.reshape(height, width)
    
    # 可视化对比
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_image, cmap='gray' if len(original_image.shape) == 2 else None)
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(np.clip(reconstructed_image, 0, 1), cmap='gray' if len(original_image.shape) == 2 else None)
    plt.title(f'Reconstructed (Epoch {epoch})')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    diff = np.abs(original_image/255.0 - np.clip(reconstructed_image, 0, 1))
    plt.imshow(diff, cmap='hot')
    plt.title('Difference')
    plt.axis('off')
    plt.colorbar()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    model.train()

def main():
    print("=== NGLOD 2D Image Fitting ===")
    
    # 加载图像
    image_path = 'original.png'
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found!")
        return
    
    image = Image.open(image_path)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    elif image.mode not in ['RGB', 'L']:
        image = image.convert('L')
    
    image_array = np.array(image)
    print(f"Loaded image: {image_array.shape}, dtype: {image_array.dtype}")
    
    args = Args2D()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = Image2DDataset(image_array, batch_size=args.batch_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    model = OctreeSDF(args).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params} parameters")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    model.train()
    current_stage = 1  # 从LOD 0开始
    last_grow_epoch = 0
    
    print(f"\n=== Training Started ===")
    
    for epoch in range(args.epochs):
        # LOD增长逻辑
        if (epoch >= last_grow_epoch + args.grow_every and 
            current_stage < args.num_lods and epoch > 0):
            current_stage += 1
            last_grow_epoch = epoch
            print(f"\nEpoch {epoch}: Growing to stage {current_stage}")
        
        # 确定当前训练的LOD级别
        if args.growth_strategy == 'increase':
            loss_lods = list(range(0, current_stage))
        elif args.growth_strategy == 'onebyone':
            loss_lods = [current_stage - 1]
        else:
            loss_lods = list(range(0, args.num_lods))
        
        epoch_loss = 0
        num_batches = 0
        
        for batch_coords, batch_values in dataloader:
            batch_coords = batch_coords.squeeze(0).to(device)  # 移除batch维度
            batch_values = batch_values.squeeze(0).to(device)
            
            # 确保值的形状正确
            if len(batch_values.shape) == 1:
                batch_values = batch_values.unsqueeze(-1)
            
            # 前向传播
            if args.return_lst:
                all_preds = model.sdf(batch_coords, return_lst=True)
                preds = [all_preds[i] for i in loss_lods]
            else:
                preds = [model.sdf(batch_coords, lod=lod) for lod in loss_lods]
            
            # 计算损失
            total_loss = 0
            for pred in preds:
                l1_loss = torch.abs(pred - batch_values).mean()
                total_loss += l1_loss
            
            total_loss = total_loss / len(preds)  # 平均多个LOD的损失
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        # 输出训练信息
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1:4d}: Loss = {avg_loss:.6f}, Stage = {current_stage}, LODs = {loss_lods}")
        
    
    print("\nFinal reconstruction:")
    visualize_reconstruction(model, image_array, device, args.epochs, save_path='nglod_2d_result.png')

if __name__ == "__main__":
    main()
