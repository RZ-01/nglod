"""
可视化feature grid mask的分布
帮助理解选择性更新机制
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from nglod_psf_ft_selective import compute_block_feature_mask


def visualize_2d_mask_slice(mask, title, lod_idx, z_slice=None):
    """可视化mask的2D切片"""
    # 移除batch和channel维度
    mask_np = mask.squeeze().numpy()
    
    # 获取中间的z切片
    if z_slice is None:
        z_slice = mask_np.shape[0] // 2
    
    slice_2d = mask_np[z_slice, :, :]
    
    plt.figure(figsize=(8, 8))
    plt.imshow(slice_2d, cmap='RdYlGn', interpolation='nearest')
    plt.colorbar(label='Trainable (1=Yes, 0=No)')
    plt.title(f'{title} - LOD {lod_idx} (z-slice {z_slice})')
    plt.xlabel('X dimension')
    plt.ylabel('Y dimension')
    plt.grid(True, alpha=0.3)
    
    # 统计信息
    total_voxels = mask_np.size
    active_voxels = mask_np.sum()
    plt.text(0.02, 0.98, 
             f'Active: {int(active_voxels)}/{total_voxels}\n({active_voxels/total_voxels*100:.1f}%)',
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             verticalalignment='top',
             fontsize=10)
    
    return plt.gcf()


def visualize_multi_lod_masks(block_coords, block_shape, volume_shape, base_lod=2, num_lods=5):
    """可视化所有LOD层级的masks"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    print(f"\nVisualizing masks for block at {block_coords}")
    print(f"Block shape: {block_shape}")
    print(f"Volume shape: {volume_shape}\n")
    
    for lod_idx in range(num_lods):
        mask = compute_block_feature_mask(
            block_coords, block_shape, volume_shape,
            lod_idx, base_lod, pad_ratio=0.3
        )
        
        # 获取mask的numpy版本
        mask_np = mask.squeeze().numpy()
        fsize = 2 ** (lod_idx + base_lod)
        
        # 取中间切片
        z_slice = mask_np.shape[0] // 2
        slice_2d = mask_np[z_slice, :, :]
        
        # 绘制
        ax = axes[lod_idx]
        im = ax.imshow(slice_2d, cmap='RdYlGn', interpolation='nearest', vmin=0, vmax=1)
        ax.set_title(f'LOD {lod_idx} (res: {fsize}x{fsize}x{fsize})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        
        # 添加统计信息
        total = mask_np.size
        active = int(mask_np.sum())
        ax.text(0.02, 0.98, 
                f'{active}/{total}\n({active/total*100:.1f}%)',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                verticalalignment='top',
                fontsize=9)
        
        print(f"LOD {lod_idx}: {active}/{total} voxels active ({active/total*100:.2f}%)")
    
    # 隐藏多余的subplot
    if num_lods < len(axes):
        axes[-1].axis('off')
    
    # 添加总标题
    fig.suptitle(f'Feature Grid Masks - Block at {block_coords}', 
                 fontsize=16, fontweight='bold')
    
    # 添加颜色条
    fig.colorbar(im, ax=axes, orientation='horizontal', 
                 fraction=0.046, pad=0.04, label='Trainable (1=Yes, 0=No)')
    
    plt.tight_layout()
    return fig


def visualize_block_positions():
    """可视化不同block位置的mask覆盖情况"""
    volume_shape = (512, 512, 512)
    block_shape = (100, 100, 100)
    base_lod = 2
    
    # 测试3个不同位置的blocks
    blocks = [
        ((50, 50, 50), "Corner Block"),
        ((256, 256, 256), "Center Block"),
        ((400, 400, 400), "Near-boundary Block"),
    ]
    
    fig, axes = plt.subplots(len(blocks), 2, figsize=(12, 4*len(blocks)))
    
    for i, (block_coords, label) in enumerate(blocks):
        # 可视化LOD 2和LOD 4
        for j, lod_idx in enumerate([2, 4]):
            mask = compute_block_feature_mask(
                block_coords, block_shape, volume_shape,
                lod_idx, base_lod, pad_ratio=0.3
            )
            
            mask_np = mask.squeeze().numpy()
            fsize = 2 ** (lod_idx + base_lod)
            z_slice = mask_np.shape[0] // 2
            slice_2d = mask_np[z_slice, :, :]
            
            ax = axes[i, j]
            im = ax.imshow(slice_2d, cmap='RdYlGn', interpolation='nearest', vmin=0, vmax=1)
            ax.set_title(f'{label} - LOD {lod_idx} (res: {fsize})')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.grid(True, alpha=0.3)
            
            # 统计
            active = int(mask_np.sum())
            total = mask_np.size
            ax.text(0.02, 0.98, 
                    f'{active}/{total}\n({active/total*100:.1f}%)',
                    transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    verticalalignment='top',
                    fontsize=9)
    
    fig.suptitle('Mask Coverage for Different Block Positions', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def compare_pad_ratios():
    """比较不同pad_ratio的效果"""
    volume_shape = (512, 512, 512)
    block_coords = (256, 256, 256)
    block_shape = (100, 100, 100)
    base_lod = 2
    lod_idx = 3
    
    pad_ratios = [0.0, 0.1, 0.2, 0.3, 0.5]
    
    fig, axes = plt.subplots(1, len(pad_ratios), figsize=(20, 4))
    
    for i, pad_ratio in enumerate(pad_ratios):
        mask = compute_block_feature_mask(
            block_coords, block_shape, volume_shape,
            lod_idx, base_lod, pad_ratio=pad_ratio
        )
        
        mask_np = mask.squeeze().numpy()
        z_slice = mask_np.shape[0] // 2
        slice_2d = mask_np[z_slice, :, :]
        
        ax = axes[i]
        im = ax.imshow(slice_2d, cmap='RdYlGn', interpolation='nearest', vmin=0, vmax=1)
        ax.set_title(f'pad_ratio={pad_ratio}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
        
        active = int(mask_np.sum())
        total = mask_np.size
        ax.text(0.02, 0.98, 
                f'{active}/{total}\n({active/total*100:.1f}%)',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                verticalalignment='top',
                fontsize=9)
    
    fig.suptitle(f'Effect of pad_ratio on Mask Coverage (LOD {lod_idx})', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def create_schematic_diagram():
    """创建示意图：展示volume空间到feature空间的映射"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 左图：Volume空间
    ax1.set_xlim(0, 512)
    ax1.set_ylim(0, 512)
    ax1.set_aspect('equal')
    ax1.set_title('Volume Space (XY plane)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X coordinate', fontsize=12)
    ax1.set_ylabel('Y coordinate', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 绘制完整volume
    volume_rect = patches.Rectangle((0, 0), 512, 512, 
                                    linewidth=2, edgecolor='black', 
                                    facecolor='lightgray', alpha=0.3)
    ax1.add_patch(volume_rect)
    ax1.text(256, -30, 'Full Volume (512x512x512)', 
             ha='center', fontsize=11, fontweight='bold')
    
    # 绘制训练block
    block_x, block_y = 150, 200
    block_size = 100
    block_rect = patches.Rectangle((block_x, block_y), block_size, block_size,
                                   linewidth=3, edgecolor='red', 
                                   facecolor='red', alpha=0.5)
    ax1.add_patch(block_rect)
    ax1.text(block_x + block_size/2, block_y + block_size/2, 
             'Training\nBlock', ha='center', va='center',
             fontsize=11, fontweight='bold', color='darkred')
    ax1.annotate(f'({block_x}, {block_y})', 
                xy=(block_x, block_y), xytext=(block_x-50, block_y-50),
                fontsize=10, arrowprops=dict(arrowstyle='->', lw=1.5))
    
    # 右图：Feature Grid空间
    fsize = 32  # LOD 3的分辨率
    ax2.set_xlim(0, fsize)
    ax2.set_ylim(0, fsize)
    ax2.set_aspect('equal')
    ax2.set_title(f'Feature Grid Space (LOD 3: {fsize}x{fsize}x{fsize})', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Feature Grid X', fontsize=12)
    ax2.set_ylabel('Feature Grid Y', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 绘制feature grid
    grid_rect = patches.Rectangle((0, 0), fsize, fsize,
                                  linewidth=2, edgecolor='black',
                                  facecolor='lightblue', alpha=0.3)
    ax2.add_patch(grid_rect)
    
    # 计算block在feature grid中的位置
    # 模拟坐标转换
    block_x_norm = (block_x / 511) * 2.0 - 1.0
    block_y_norm = (block_y / 511) * 2.0 - 1.0
    block_x_feat = (block_x_norm + 1.0) / 2.0 * fsize
    block_y_feat = (block_y_norm + 1.0) / 2.0 * fsize
    
    block_size_norm = (block_size / 511) * 2.0
    block_size_feat = block_size_norm / 2.0 * fsize
    
    # 绘制对应的feature region (带padding)
    pad = block_size_feat * 0.3
    feat_rect = patches.Rectangle((block_x_feat - pad, block_y_feat - pad),
                                  block_size_feat + 2*pad, block_size_feat + 2*pad,
                                  linewidth=3, edgecolor='green',
                                  facecolor='green', alpha=0.5)
    ax2.add_patch(feat_rect)
    ax2.text(block_x_feat + block_size_feat/2, block_y_feat + block_size_feat/2,
             'Active\nFeatures', ha='center', va='center',
             fontsize=11, fontweight='bold', color='darkgreen')
    
    # 绘制箭头表示映射
    fig.text(0.48, 0.5, '→\nCoordinate\nTransform', 
             ha='center', va='center', fontsize=12, 
             fontweight='bold', color='blue')
    
    plt.tight_layout()
    return fig


def main():
    """生成所有可视化"""
    print("="*60)
    print("Visualizing Feature Grid Masks")
    print("="*60)
    
    # 1. 多LOD masks可视化
    print("\n[1] Generating multi-LOD mask visualization...")
    volume_shape = (512, 512, 512)
    block_coords = (200, 200, 200)
    block_shape = (100, 100, 100)
    
    fig1 = visualize_multi_lod_masks(block_coords, block_shape, volume_shape)
    fig1.savefig('mask_visualization_multi_lod.png', dpi=150, bbox_inches='tight')
    print("    Saved: mask_visualization_multi_lod.png")
    
    # 2. 不同位置的blocks
    print("\n[2] Generating block position comparison...")
    fig2 = visualize_block_positions()
    fig2.savefig('mask_visualization_positions.png', dpi=150, bbox_inches='tight')
    print("    Saved: mask_visualization_positions.png")
    
    # 3. 不同pad_ratio的效果
    print("\n[3] Generating pad_ratio comparison...")
    fig3 = compare_pad_ratios()
    fig3.savefig('mask_visualization_pad_ratios.png', dpi=150, bbox_inches='tight')
    print("    Saved: mask_visualization_pad_ratios.png")
    
    # 4. 示意图
    print("\n[4] Generating schematic diagram...")
    fig4 = create_schematic_diagram()
    fig4.savefig('mask_schematic_diagram.png', dpi=150, bbox_inches='tight')
    print("    Saved: mask_schematic_diagram.png")
    
    print("\n" + "="*60)
    print("All visualizations generated successfully!")
    print("="*60)
    
    # 显示图形（可选）
    # plt.show()


if __name__ == "__main__":
    main()
