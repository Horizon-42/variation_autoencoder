import torch
import lpips
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt

def run_diagnosis():
    print("="*50)
    print("🕵️‍♂️ LPIPS 负值问题 - 深度诊断脚本")
    print("="*50)

    # 1. 环境准备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"1. 运行设备: {device}")

    try:
        # 初始化模型 (使用 alex 或 vgg 都可以，这里用 alex 速度快)
        loss_fn = lpips.LPIPS(net='alex').to(device).eval()
        print("2. LPIPS 模型加载成功 ✅")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 2. 模拟数据 (模拟你 VAE 的输出)
    # 生成两个随机的 [Batch, 3, 64, 64] 张量
    # 模拟 Tanh 的输出范围 [-1, 1]
    torch.manual_seed(42) # 固定种子保证结果可复现
    
    # 构造原图 (Input)
    img_input = (torch.rand(4, 3, 64, 64) * 2 - 1).to(device)
    
    # 构造重构图 (Recons) - 添加一些噪声模拟不完美的重构
    img_recons = img_input + 0.1 * torch.randn_like(img_input)
    img_recons = torch.clamp(img_recons, -1, 1).to(device)

    print("\n" + "-"*30)
    print("3. 数据范围检查 (关键步骤)")
    print(f"   Input Min: {img_input.min().item():.4f}, Max: {img_input.max().item():.4f}")
    print(f"   Recons Min: {img_recons.min().item():.4f}, Max: {img_recons.max().item():.4f}")
    
    if img_input.min() < -1.1 or img_input.max() > 1.1:
        print("   ⚠️ 警告: 输入数据似乎超出了 [-1, 1] 范围")
    else:
        print("   ✅ 数据范围看起来正常 (符合 LPIPS 要求)")

    # 3. 核心测试：直接计算 LPIPS
    print("\n" + "-"*30)
    print("4. 核心测试: LPIPS 原始输出")
    
    with torch.no_grad():
        # 注意：这里 normalize=False，因为我们已经手动把数据弄成 [-1, 1] 了
        raw_dist = loss_fn(img_recons, img_input, normalize=False)
        mean_dist = raw_dist.mean().item()

    print(f"   Raw Distance Tensor Shape: {raw_dist.shape}")
    print(f"   >>> 原始平均距离 (Mean Distance): {mean_dist:.6f}")

    if mean_dist >= 0:
        print("   ✅ 结果为正数: LPIPS 库本身工作正常！")
    else:
        print("   ❌ 结果为负数: LPIPS 库或输入数据有严重问题！")

    # 4. 模拟场景复现 (寻找凶手)
    print("\n" + "-"*30)
    print("5. 模拟你的训练代码 (寻找负值来源)")

    # 模拟场景 A: 正常的加法
    weight_positive = 0.5
    loss_a = weight_positive * mean_dist
    print(f"   [场景 A] Loss = +0.5 * dist  ->  {loss_a:.6f} (✅ 正常)")

    # 模拟场景 B: 权重为负 (嫌疑人 1)
    weight_negative = -0.5
    loss_b = weight_negative * mean_dist
    print(f"   [场景 B] Loss = -0.5 * dist  ->  {loss_b:.6f} (❌ 负值 - 可能是权重设错了)")

    # 模拟场景 C: 减法公式 (嫌疑人 2)
    # 假设 Recon Loss 是 0.2
    mse_dummy = 0.2
    loss_c = mse_dummy - mean_dist
    print(f"   [场景 C] Loss = MSE - dist   ->  {loss_c:.6f} (❌ 负值 - 可能是用了减法)")

    # 用实际数据测试
    import pickle
    with open("recons_input_debug.pkl", "rb") as f:
        recons_loaded, input_loaded = pickle.load(f)
    recons_loaded = recons_loaded.to(device)
    input_loaded = input_loaded.to(device)

    with torch.no_grad():
        loaded_dist = loss_fn(recons_loaded, input_loaded, normalize=False)
        loaded_mean_dist = loaded_dist.mean().item()
    print(f"\n   [实际数据测试] Loaded Mean Distance: {loaded_mean_dist:.6f}")
    print(f"\n [实际数据distsum] Loaded Distance Sum: {loaded_dist.sum().item():.6f}")

    inv_transform = transforms.Compose([
        # denormalize
        transforms.Lambda(lambda x: (x * 0.5) + 0.5),
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
    ])

    # show recons and input images
    print("\n" + "-"*30)
    print("6. 可视化检查重构图和原图")
    for i in range(recons_loaded.size(0)):
        recons_img = inv_transform(recons_loaded[i].cpu())
        input_img = inv_transform(input_loaded[i].cpu())

        plt.subplot(2, recons_loaded.size(0), i+1)
        plt.imshow(recons_img)
        plt.title("Recons")
        plt.axis('off')

        plt.subplot(2, recons_loaded.size(0), i+1+recons_loaded.size(0))
        plt.imshow(input_img)
        plt.title("Input")
        plt.axis('off')    
    plt.show()

    print("\n" + "="*50)
    print("🏁 诊断结论:")
    if mean_dist >= 0:
        print("LPIPS 算出来的是正数。")
        print("既然你的日志里显示负数，说明你在 loss_function 里")
        print("一定做了 【减法】 或者乘了 【负权重】。")
        print("请去检查 loss = ... 那一行！")
    else:
        print("LPIPS 竟然算出了负数... 这在数学上几乎不可能。")
        print("请检查 lpips 库的版本或 PyTorch 版本。")

if __name__ == "__main__":
    run_diagnosis()