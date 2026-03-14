"""
图像去噪方法比较：DnCNN vs BM3D vs TV(ADMM) vs Wavelet(ISTA/FISTA)
要求：在 denoise 环境中运行，并确保 pretrained/model.pth 存在。
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from skimage import data, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import denoise_tv_chambolle, denoise_wavelet
import bm3d
import os

# -------------------- 设备设置 --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -------------------- DnCNN 模型定义 --------------------
class DnCNN(nn.Module):
    """DnCNN网络结构（17层，残差学习）"""
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bn=True):
        super(DnCNN, self).__init__()
        layers = [nn.Conv2d(image_channels, n_channels, kernel_size=3, padding=1),
                  nn.ReLU(inplace=True)]
        for _ in range(depth-2):
            layers.append(nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1))
            if use_bn:
                layers.append(nn.BatchNorm2d(n_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(n_channels, image_channels, kernel_size=3, padding=1))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.dncnn(x)

# -------------------- 加载 DnCNN 预训练权重 --------------------
weights_path = 'pretrained/model.pth'  # 用户下载的权重文件
if os.path.exists(weights_path):
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)  # 允许加载完整模型
    if isinstance(checkpoint, dict):
        # 如果保存的是 state_dict
        model = DnCNN().to(device)
        model.load_state_dict(checkpoint)
    else:
        # 如果保存的是完整模型
        model = checkpoint.to(device)
    model.eval()
    print("DnCNN model loaded successfully.")
else:
    model = None
    print(f"Warning: DnCNN weights not found at {weights_path}. Skipping DnCNN method.")

# -------------------- 准备测试图像 --------------------
# 使用 skimage 内置的 camera 图像（512x512 灰度图）
image = img_as_float(data.camera())  # 取值范围 [0,1]
np.random.seed(0)  # 固定随机种子以便复现
sigma = 25 / 255.0  # 噪声标准差（对应图像范围 [0,1]）
noisy = image + sigma * np.random.randn(*image.shape)
noisy = np.clip(noisy, 0, 1)  # 保证像素值在合法范围内

# -------------------- DnCNN 去噪函数 --------------------
def dncnn_denoise(model, img_noisy, device):
    """输入噪声图像 (H,W) 返回去噪图像 (H,W)"""
    if model is None:
        return img_noisy
    with torch.no_grad():
        # 转换为 tensor，形状 (1,1,H,W)
        img_tensor = torch.from_numpy(img_noisy).float().view(1, 1, img_noisy.shape[0], img_noisy.shape[1]).to(device)
        residual = model(img_tensor)               # 预测残差
        denoised = img_tensor - residual           # 去噪图像 = 噪声图像 - 残差
    return denoised.cpu().numpy().squeeze()       # 转回 numpy 并去除多余维度

# -------------------- 执行去噪 --------------------
results = {'Noisy': noisy}

# DnCNN
if model is not None:
    results['DnCNN'] = dncnn_denoise(model, noisy, device)

# BM3D（注意：sigma 应与图像范围一致，noisy 是 [0,1]，sigma 应为 25/255）
results['BM3D'] = bm3d.bm3d(noisy, sigma)  # sigma = 25/255.0

# TV 去噪（ADMM 的代表算法）
results['TV(ADMM)'] = denoise_tv_chambolle(noisy, weight=0.1, channel_axis=None)

# 小波去噪（ISTA/FISTA 的代表算法：正交小波基下的软阈值）
results['Wavelet(ISTA/FISTA)'] = denoise_wavelet(noisy, method='BayesShrink', mode='soft', rescale_sigma=True)

# -------------------- 计算评价指标 --------------------
metrics = {}
for name, img in results.items():
    p = psnr(image, img, data_range=1.0)
    s = ssim(image, img, data_range=1.0)
    metrics[name] = (p, s)

# -------------------- 打印表格 --------------------
print("\nMethod\t\t\tPSNR\tSSIM")
print("----------------------------------------")
for name, (p, s) in metrics.items():
    # 对齐输出
    print(f"{name:<20}\t{p:.2f}\t{s:.3f}")

# -------------------- 可视化对比 --------------------
plt.figure(figsize=(15, 10))
titles = ['Original', 'Noisy'] + list(results.keys())[1:]
images = [image, noisy] + list(results.values())[1:]
for i, (img, title) in enumerate(zip(images, titles)):
    plt.subplot(2, 4, i+1)
    plt.imshow(img, cmap='gray')
    if title in metrics:
        plt.title(f'{title}\nPSNR={metrics[title][0]:.2f}, SSIM={metrics[title][1]:.3f}')
    else:
        plt.title(title)
    plt.axis('off')
plt.tight_layout()
plt.savefig('comparison.png', dpi=300)
plt.show()

print("\n对比图像已保存为 comparison.png")