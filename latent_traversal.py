import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

def get_active_dim_indices(model, data_loader, device='cuda', top_k=10):
    """
    扫描验证集，找出KLD最高的 top_k 个维度索引。
    这能避免我们去遍历那些“死掉”的维度（也就是没学到东西的噪声维度）。
    """
    model.eval()
    total_kld_per_dim = 0
    count = 0
    
    print("正在扫描活跃维度...")
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader):
            data = data.to(device)
            # Encode 获取 mu 和 logvar
            # 注意：根据你的模型写法，这里可能返回 (mu, logvar) 或者其他
            # 假设你的 encode 返回 mu, logvar
            mu, log_var = model.encode(data)
            
            # 计算这一批次每一维的 KLD: -0.5 * (1 + logvar - mu^2 - exp(logvar))
            # 我们只关心维度的平均值 (dim=0)
            kld = -0.5 * torch.mean(1 + log_var - mu ** 2 - log_var.exp(), dim=0)
            
            total_kld_per_dim += kld
            count += 1
            
            if batch_idx >= 50: # 没必要跑完整个验证集，跑50个batch够了
                break
                
    avg_kld = total_kld_per_dim / count
    
    # 获取数值最大的 top_k 个维度的索引
    # values: KLD 数值, indices: 对应的维度下标 (0-255)
    values, indices = torch.topk(avg_kld, top_k)
    
    print(f"找到最活跃的前 {top_k} 个维度，KLD 范围: {values.min().item():.4f} - {values.max().item():.4f}")
    return indices.tolist()

def visualize_traversal(model, base_image, active_dims, device='cuda', min_val=-3, max_val=3, steps=10):
    """
    生成遍历图：每一行代表一个活跃维度的变化。
    """
    model.eval()
    base_image = base_image.to(device).unsqueeze(0) # [1, C, H, W]
    
    # 1. 拿到基准图片的潜在向量 (mu)
    with torch.no_grad():
        mu, _ = model.encode(base_image) # 我们用 mu 作为基准，忽略随机性
    
    traversal_images = []
    
    # 2. 遍历每一个活跃维度
    print(f"正在生成遍历图，遍历范围 [{min_val}, {max_val}]...")
    for dim_idx in active_dims:
        # 在 min_val 和 max_val 之间生成 steps 个插值
        # 例如: [-3, -2.3, ..., 0, ..., 2.3, 3]
        interpolation_range = torch.linspace(min_val, max_val, steps).to(device)
        
        row_images = []
        for val in interpolation_range:
            # 复制一份基准向量
            z = mu.clone()
            
            # 修改特定维度的值 (这就是“控制变量法”)
            z[0, dim_idx] = val
            
            # 解码生成图像
            with torch.no_grad():
                recon = model.decode(z)
            
            row_images.append(recon.cpu())
            
        # 将这一行的图片拼起来
        traversal_images.extend(row_images)
    
    # 3. 绘图
    # stack 之后 shape 是 [dims * steps, C, H, W]
    traversal_tensor = torch.stack(traversal_images).squeeze(1) 
    
    # 制作网格：每一行有 steps 张图
    grid_img = make_grid(traversal_tensor, nrow=steps, padding=2, pad_value=1)
    
    plt.figure(figsize=(20, 2 * len(active_dims))) # 高度根据行数自动调整
    # 转换维度以适应 matplotlib: (C, H, W) -> (H, W, C)
    plt.imshow(grid_img.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.title(f"Latent Space Traversal (Top {len(active_dims)} Active Dimensions)", fontsize=16)
    plt.show()

# ==========================================
# 如何使用这段代码
# ==========================================

# 1. 准备数据和模型
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.eval()
# 获取一张测试图片作为“基准”（种子）
# dataset_item = next(iter(test_loader))[0] # 获取一个batch
# base_img = dataset_item[0] # 取第一张图

# 2. 找出那 36 个活跃维度里最强的 10 个
# top_dims = get_active_dim_indices(model, test_loader, device=device, top_k=10)
# print("活跃维度索引:", top_dims)

# 3. 开始可视化
# visualize_traversal(model, base_img, top_dims, device=device)