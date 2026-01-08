"""
测试使用未对齐的图片训练Beta-VAE的效果
比较对齐图片和未对齐图片的训练结果
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm

from models.beta_vae import BetaVAE


class UnalignedCelebADataset(Dataset):
    """
    加载未对齐的CelebA图片
    模拟未经过人脸对齐预处理的情况
    """
    def __init__(self, root, split='train', transform=None, add_noise=False):
        """
        Args:
            root: CelebA数据集根目录
            split: 'train', 'valid', 或 'test'
            transform: 图像变换
            add_noise: 是否添加随机噪声模拟未对齐效果
        """
        self.root = root
        self.transform = transform
        self.add_noise = add_noise
        
        # 读取图片列表
        img_dir = os.path.join(root, 'img_align_celeba')
        self.img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')])
        
        # 读取划分文件
        split_file = os.path.join(root, 'list_eval_partition.txt')
        split_map = {'train': 0, 'valid': 1, 'test': 2}
        split_id = split_map[split]
        
        with open(split_file, 'r') as f:
            lines = f.readlines()
        
        # 筛选对应split的图片
        filtered_paths = []
        for line in lines:
            filename, partition = line.strip().split()
            if int(partition) == split_id:
                img_path = os.path.join(img_dir, filename)
                if os.path.exists(img_path):
                    filtered_paths.append(img_path)
        
        self.img_paths = filtered_paths
        print(f"Loaded {len(self.img_paths)} images for {split} split")
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # 模拟未对齐效果：随机裁剪、旋转、缩放
        if self.add_noise:
            image = self.simulate_misalignment(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0  # 返回image和dummy label
    
    def simulate_misalignment(self, image):
        """
        模拟未对齐的效果：
        1. 随机裁剪（模拟人脸位置偏移）
        2. 随机旋转（模拟头部角度变化）
        3. 随机缩放（模拟远近距离变化）
        """
        w, h = image.size
        
        # 随机旋转 -15到15度
        angle = np.random.uniform(-15, 15)
        image = image.rotate(angle, fillcolor=(0, 0, 0))
        
        # 随机裁剪和缩放（模拟人脸位置和大小变化）
        crop_ratio = np.random.uniform(0.7, 1.0)
        new_w, new_h = int(w * crop_ratio), int(h * crop_ratio)
        
        left = np.random.randint(0, w - new_w + 1)
        top = np.random.randint(0, h - new_h + 1)
        
        image = image.crop((left, top, left + new_w, top + new_h))
        image = image.resize((w, h), Image.BILINEAR)
        
        return image


def train_vae(model, dataloader, optimizer, device, num_epochs=10, experiment_name=''):
    """
    训练VAE模型
    """
    model.train()
    history = {
        'loss': [],
        'recon_loss': [],
        'kld_loss': []
    }
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_recon = 0
        epoch_kld = 0
        
        pbar = tqdm(dataloader, desc=f'[{experiment_name}] Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            results = model(data)
            
            # 计算损失
            loss = model.loss_function(*results, M_N=1.0/len(dataloader.dataset), is_val=False)
            
            # Backward pass
            loss['loss'].backward()
            optimizer.step()
            
            # 记录损失
            epoch_loss += loss['loss'].item()
            epoch_recon += loss['Reconstruction_Loss'].item()
            epoch_kld += loss['KLD_Loss'].item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss['loss'].item():.4f}",
                'recon': f"{loss['Reconstruction_Loss'].item():.4f}",
                'kld': f"{loss['KLD_Loss'].item():.4f}"
            })
            
            # 定期清理GPU内存
            if batch_idx % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 计算平均损失
        avg_loss = epoch_loss / len(dataloader)
        avg_recon = epoch_recon / len(dataloader)
        avg_kld = epoch_kld / len(dataloader)
        
        history['loss'].append(avg_loss)
        history['recon_loss'].append(avg_recon)
        history['kld_loss'].append(avg_kld)
        
        print(f'[{experiment_name}] Epoch {epoch+1}: Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, KLD={avg_kld:.4f}')
    
    return history


def evaluate_reconstruction(model, dataloader, device, num_samples=8):
    """
    评估重建质量
    """
    model.eval()
    
    with torch.no_grad():
        # 获取一批样本
        data, _ = next(iter(dataloader))
        data = data[:num_samples].to(device)
        
        # 重建
        results = model(data)
        recons = results[0]
        
        # 计算重建误差
        mse = torch.mean((data - recons) ** 2).item()
        
        return data.cpu(), recons.cpu(), mse


def plot_comparison(original, reconstructed, title, save_path):
    """
    可视化原始图片和重建图片的对比
    """
    n = len(original)
    fig, axes = plt.subplots(2, n, figsize=(n*2, 4))
    
    for i in range(n):
        # 原始图片
        img_orig = original[i].permute(1, 2, 0).numpy()
        img_orig = np.clip(img_orig, 0, 1)
        axes[0, i].imshow(img_orig)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)
        
        # 重建图片
        img_recon = reconstructed[i].permute(1, 2, 0).numpy()
        img_recon = np.clip(img_recon, 0, 1)
        axes[1, i].imshow(img_recon)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10)
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved comparison plot to {save_path}")


def plot_training_curves(history_aligned, history_unaligned, save_dir):
    """
    绘制训练曲线对比
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    metrics = ['loss', 'recon_loss', 'kld_loss']
    titles = ['Total Loss', 'Reconstruction Loss', 'KLD Loss']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        axes[idx].plot(history_aligned[metric], label='Aligned', linewidth=2)
        axes[idx].plot(history_unaligned[metric], label='Unaligned', linewidth=2)
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel('Loss')
        axes[idx].set_title(title)
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves_comparison.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved training curves to {save_path}")


def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备（训练优先用GPU；评估用CPU更省显存）
    train_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval_device = torch.device('cpu')
    print(f"Train device: {train_device} | Eval device: {eval_device}")
    
    # 超参数
    config = {
        'image_size': 64,
        'latent_dim': 128,
        'hidden_dims': [32, 64, 128, 256, 512],
        'in_channels': 3,
        'beta': 1,
        'gamma': 10,
        'max_capacity': 250,
        'loss_type': 'H',
        'batch_size': 64,  # 减小batch size以节省GPU内存
        'num_epochs': 15,  # 减少epoch数量
        'learning_rate': 1e-4,
        'enable_perceptual_loss': False,
        'lpips_weight': 0.0,
        'tvl_weight': 0.0,
        'num_train_samples': 5000  # 限制训练样本数量
    }
    
    # 数据集路径
    data_root = './data/celeba'
    if not os.path.exists(data_root):
        print(f"Error: Data directory {data_root} not found!")
        return
    
    # 创建结果目录
    result_dir = './results_unaligned_test'
    os.makedirs(result_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(result_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
    ])
    
    print("\n" + "="*60)
    print("实验1: 使用对齐的图片训练")
    print("="*60)
    
    # 实验1: 使用对齐的图片（原始CelebA）
    from torch.utils.data import Subset

    dataset_aligned_full = UnalignedCelebADataset(
        root=data_root,
        split='train',
        transform=transform,
        add_noise=False  # 使用对齐图像
    )

    # 固定同一批样本，保证两次训练可比
    num_train_samples = min(int(config['num_train_samples']), len(dataset_aligned_full))
    indices = np.random.choice(len(dataset_aligned_full), size=num_train_samples, replace=False)

    dataset_aligned = Subset(dataset_aligned_full, indices)
    
    dataloader_aligned = DataLoader(
        dataset_aligned,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,  # 减少worker数量
        pin_memory=True if torch.cuda.is_available() else False
    )

    print("\n" + "="*60)
    print("实验2: 使用未对齐的图片训练（随机扰动模拟未对齐）")
    print("="*60)

    dataset_unaligned_full = UnalignedCelebADataset(
        root=data_root,
        split='train',
        transform=transform,
        add_noise=True  # 添加噪声，模拟未对齐
    )
    dataset_unaligned = Subset(dataset_unaligned_full, indices)

    dataloader_unaligned = DataLoader(
        dataset_unaligned,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,  # 减少worker数量
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 创建模型（对齐图片）
    model_aligned = BetaVAE(
        in_channels=config['in_channels'],
        latent_dim=config['latent_dim'],
        hidden_dims=config['hidden_dims'],
        beta=config['beta'],
        gamma=config['gamma'],
        max_capacity=config['max_capacity'],
        loss_type=config['loss_type'],
        image_size=(config['image_size'], config['image_size']),
        enable_perceptual_loss=config['enable_perceptual_loss'],
        lpips_weight=config['lpips_weight'],
        tvl_weight=config['tvl_weight']
    ).to(train_device)
    
    optimizer_aligned = torch.optim.Adam(model_aligned.parameters(), lr=config['learning_rate'])
    
    # 训练（对齐图片）
    history_aligned = train_vae(
        model_aligned,
        dataloader_aligned,
        optimizer_aligned,
        train_device,
        num_epochs=config['num_epochs'],
        experiment_name='Aligned Images'
    )
    
    # 保存模型
    torch.save(model_aligned.state_dict(), os.path.join(result_dir, 'model_aligned.pth'))
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 将已训练模型挪到CPU，避免GPU上同时驻留两套权重
    model_aligned = model_aligned.to(eval_device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 创建模型（未对齐图片）
    model_unaligned = BetaVAE(
        in_channels=config['in_channels'],
        latent_dim=config['latent_dim'],
        hidden_dims=config['hidden_dims'],
        beta=config['beta'],
        gamma=config['gamma'],
        max_capacity=config['max_capacity'],
        loss_type=config['loss_type'],
        image_size=(config['image_size'], config['image_size']),
        enable_perceptual_loss=config['enable_perceptual_loss'],
        lpips_weight=config['lpips_weight'],
        tvl_weight=config['tvl_weight']
    ).to(train_device)
    
    optimizer_unaligned = torch.optim.Adam(model_unaligned.parameters(), lr=config['learning_rate'])
    
    # 训练（未对齐图片）
    history_unaligned = train_vae(
        model_unaligned,
        dataloader_unaligned,
        optimizer_unaligned,
        train_device,
        num_epochs=config['num_epochs'],
        experiment_name='Unaligned Images'
    )

    # 保存模型
    torch.save(model_unaligned.state_dict(), os.path.join(result_dir, 'model_unaligned.pth'))

    # 将模型挪到CPU便于评估与节省显存
    model_unaligned = model_unaligned.to(eval_device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("评估和可视化")
    print("="*60)
    
    # 评估重建质量
    orig_aligned, recon_aligned, mse_aligned = evaluate_reconstruction(
        model_aligned, dataloader_aligned, eval_device
    )
    orig_unaligned, recon_unaligned, mse_unaligned = evaluate_reconstruction(
        model_unaligned, dataloader_unaligned, eval_device
    )
    
    print(f"\n重建MSE对比:")
    print(f"  对齐图片: {mse_aligned:.6f}")
    print(f"  未对齐图片: {mse_unaligned:.6f}")
    print(f"  差异: {abs(mse_aligned - mse_unaligned):.6f} ({((mse_unaligned/mse_aligned - 1)*100):.2f}%)")
    
    # 可视化重建结果
    plot_comparison(
        orig_aligned, recon_aligned,
        'Aligned Images - Reconstruction',
        os.path.join(result_dir, 'reconstruction_aligned.png')
    )
    
    plot_comparison(
        orig_unaligned, recon_unaligned,
        'Unaligned Images - Reconstruction',
        os.path.join(result_dir, 'reconstruction_unaligned.png')
    )
    
    # 绘制训练曲线
    plot_training_curves(history_aligned, history_unaligned, result_dir)
    
    # 保存训练历史
    results = {
        'aligned': {
            'history': history_aligned,
            'final_mse': mse_aligned
        },
        'unaligned': {
            'history': history_unaligned,
            'final_mse': mse_unaligned
        },
        'config': config
    }
    
    with open(os.path.join(result_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n" + "="*60)
    print("实验总结")
    print("="*60)
    print(f"\n最终训练损失对比:")
    print(f"  对齐图片 - Total Loss: {history_aligned['loss'][-1]:.4f}")
    print(f"  未对齐图片 - Total Loss: {history_unaligned['loss'][-1]:.4f}")
    print(f"\n最终重建损失对比:")
    print(f"  对齐图片 - Recon Loss: {history_aligned['recon_loss'][-1]:.4f}")
    print(f"  未对齐图片 - Recon Loss: {history_unaligned['recon_loss'][-1]:.4f}")
    print(f"\n最终KLD损失对比:")
    print(f"  对齐图片 - KLD Loss: {history_aligned['kld_loss'][-1]:.4f}")
    print(f"  未对齐图片 - KLD Loss: {history_unaligned['kld_loss'][-1]:.4f}")
    
    print(f"\n所有结果已保存到: {result_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
