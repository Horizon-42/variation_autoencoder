# 📘 Burgess Beta-VAE 与感知损失 (LPIPS) 深度复盘

**文档版本：** 1.0
**主题：** 从 MSE 模糊瓶颈到 LPIPS 对抗攻击修复
**核心原理：** 信息瓶颈理论、感知距离度量、梯度优化动力学

---

## 1. 核心动力学：Burgess Capacity 机制

在 Burgess $\beta$-VAE 模型中，KLD 随着训练不断增大是**预期行为**，而非异常。

### 1.1 数学原理：信息瓶颈 (Information Bottleneck)
普通 VAE 的目标是最大化 ELBO。但在 Burgess 架构中，引入了动态约束：
$$\mathcal{L} = \mathcal{L}_{recon} + \gamma | D_{KL}(q(z|x) || p(z)) - C |$$

* **$C$ (Capacity)：** 它是“信息通道的带宽”。
* **$D_{KL}$ (KL 散度)：** 它是潜变量 $z$ 中包含的关于输入 $x$ 的**信息量 (Information Content)**。

### 1.2 训练过程中的物理现象
1.  **初始阶段 ($C \approx 0$)：**
    * 带宽被掐死。模型被迫丢弃信息，后验分布 $q(z|x)$ 坍缩为先验 $p(z)$。
    * **现象：** KLD $\approx$ 0，重建图为数据集的“平均脸”（模糊）。
2.  **爬坡阶段 ($C$ 线性增长)：**
    * 带宽逐渐解禁。模型**按重要性**学习特征：轮廓 -> 光照 -> 纹理。
    * Encoder 必须让 $D_{KL}$ **主动增大**以匹配 $C$，否则会被 $\gamma$ 惩罚。
    * **现象：** KLD 曲线呈现完美的线性上升。
3.  **几何意义：**
    * KLD 增大意味着分布 $q(z|x)$ 变得极度“尖锐”（低方差）或“偏离原点”。这代表模型在潜空间中对特征进行了精确定位。

---

## 2. 突破瓶颈：从 MSE 到 LPIPS

### 2.1 MSE 的局限性
即使 KLD 达到 150+，图片依然模糊。
* **原理：** MSE (L2 Loss) 假设误差服从高斯分布。面对不确定性（如发丝位置），MSE 的最优解是**平均值**。
* **后果：** 高频细节被平滑，导致“磨皮感”。

### 2.2 LPIPS 感知原理
LPIPS (Learned Perceptual Image Patch Similarity) 比较的是**特征**而非像素。

$$d(x, x_0) = \sum_{l} \frac{1}{H_l W_l} \sum_{h,w} || w_l \odot (\hat{y}^l_{hw} - \hat{y}^l_{0,hw}) ||_2^2$$

* **特征提取 ($\hat{y}^l$)：** 利用预训练网络 (VGG/AlexNet) 提取多层特征。
* **通道校准 ($w_l$)：** 这是一个**习得的** $1 \times 1$ 卷积层，基于人类感知数据 (BAPS) 训练，用于加权重要特征。
* **作用：** 只有生成出能激活特定特征层的高频细节，Loss 才会下降。

---

## 3. 故障复盘：优化器作弊 (Adversarial Attack)

引入 LPIPS 时出现了 Loss 为负且图像变成波纹噪声的事故。这是**未冻结参数**导致的。

### 3.1 事故机理推演
当 `requires_grad=True` 时，LPIPS 参数与 VAE 参数同时被更新。

1.  **目标异化：** 优化器试图最小化总 Loss。
2.  **权重翻转：** LPIPS 内部权重 $w_l$ 被优化器更新为**负数**。
3.  **最大化差异：** 公式变为 $Loss \propto -1 \times (\text{Diff})^2$。为了让 Loss 更负，Decoder 生成与原图差异最大的**波纹噪声**。

### 3.2 结论
这是一次对抗攻击：VAE 生成对抗样本欺骗 LPIPS，同时优化器篡改了评分标准。

---

## 4. 最佳实践代码 (Best Practices)

### 4.1 修复后的初始化
必须强制冻结 LPIPS 参数。

```python
import lpips

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载模型
        self.lpips_model = lpips.LPIPS(net='alex').eval()
        
        # 🚨 关键：切断梯度，防止作弊
        for param in self.lpips_model.parameters():
            param.requires_grad = False
            
        self.lpips_weight = 0.5
```

## 4.2 鲁棒的 Loss 计算
注意输入范围必须是 [-1, 1]。

```Python

def loss_function(self, recons, input, ...):
    # 假设 recons 和 input 均来自 Tanh 或已归一化至 [-1, 1]
    
    recon_loss = F.mse_loss(recons, input)
    
    # 确保没有负号，确保输入范围正确
    perceptual_loss = self.lpips_model(recons, input).mean()
    
    # Burgess KLD 惩罚
    loss = recon_loss + (self.lpips_weight * perceptual_loss) + gamma * |kld_loss - C|
    
    return loss
```

## 5. 总结与展望
当前状态： 波纹噪声已消失，模型恢复正常学习，但细节尚模糊。

下一步： 提高 lpips_weight (至 0.5-1.0)，保持 Capacity 策略，耐心训练等待纹理浮现。