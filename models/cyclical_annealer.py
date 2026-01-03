import numpy as np
import matplotlib.pyplot as plt

class CyclicalAnnealer:
    def __init__(self, 
                 total_steps: int, 
                 n_cycles: int = 4, 
                 max_beta: float = 1.0, 
                 ratio: float = 0.5, 
                 mode: str = 'linear'):
        """
        循环退火调度器
        
        参数:
            total_steps (int): 总训练步数 (epochs * steps_per_epoch)
            n_cycles (int): 周期数量 (通常为 4 或 5)
            max_beta (float): Beta 的最大值 (通常为 1.0)
            ratio (float): 每个周期内 Beta 上升阶段所占的比例 (0.0 ~ 1.0)。
                           例如 0.5 表示前 50% 时间上升，后 50% 时间保持 max_beta。
            mode (str): 'linear' (线性) 或 'sigmoid' (S型曲线)
        """
        self.total_steps = total_steps
        self.n_cycles = n_cycles
        self.max_beta = max_beta
        self.ratio = ratio
        self.mode = mode
        
        # 计算每个周期的长度
        self.period = total_steps // n_cycles
        # 计算上升阶段的步数
        self.step_growth = int(self.period * ratio)

    def __call__(self, step: int) -> float:
        """
        根据当前 step 返回对应的 beta 值
        """
        # 1. 确定当前处于周期的第几步
        cycle_step = step % self.period
        
        # 2. 如果处于上升阶段 (Annealing Phase)
        if cycle_step < self.step_growth:
            if self.mode == 'linear':
                # 线性增长: y = x / total
                return self.max_beta * (cycle_step / self.step_growth)
            
            elif self.mode == 'sigmoid':
                # Sigmoid 增长: 更加平滑
                # 映射 x 从 [-6, 6] 以涵盖 sigmoid 的主要变化区间
                x = (cycle_step / self.step_growth) * 12.0 - 6.0
                return self.max_beta / (1.0 + np.exp(-x))
        
        # 3. 如果处于保持阶段 (Plateau Phase)
        else:
            return self.max_beta

# --- 测试与可视化代码 ---
if __name__ == "__main__":
    # 假设训练 100 个 Epoch，每个 Epoch 100 步
    TOTAL_STEPS = 10000 
    
    # 初始化调度器
    annealer = CyclicalAnnealer(total_steps=TOTAL_STEPS, 
                                n_cycles=4, 
                                ratio=0.5, 
                                max_beta=10,
                                mode='linear')

    # 模拟训练过程
    betas = []
    for i in range(TOTAL_STEPS):
        b = annealer(i)
        betas.append(b)

    # 画图
    plt.figure(figsize=(10, 4))
    plt.plot(betas, label='Beta Value')
    plt.title(f"Cyclical Annealing Schedule (Cycles=4, Ratio=0.5)")
    plt.xlabel("Step")
    plt.ylabel("Beta")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()