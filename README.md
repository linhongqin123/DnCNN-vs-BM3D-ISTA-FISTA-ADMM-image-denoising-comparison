# 图像去噪方法比较：DnCNN vs BM3D / ISTA / FISTA / ADMM

本项目对五种经典的图像去噪方法进行了对比实验，包括深度学习方法（DnCNN）、传统滤波方法（BM3D）以及基于优化的方法（TV-ADMM 代表 ADMM，小波软阈值代表 ISTA/FISTA）。所有方法均在标准测试图像 `camera` 上进行了评估，噪声水平为高斯白噪声（σ=25）。实验结果以 PSNR 和 SSIM 指标呈现，并提供可视化对比图。

## 项目结构
```
.
├── denoise_comparison.py          # 主程序脚本
├── pretrained/
│   └── model.pth                   # DnCNN 预训练权重（σ=25）
├── comparison.png                   # 生成的结果对比图
├── README.md                        # 本说明文件
├── DnCNN.tex           # 论文 LaTeX 源码
└── DnCNN.pdf            # 编译后的论文 PDF
```

## 环境配置
推荐使用 Python 3.8 和以下依赖库：
- PyTorch / torchvision
- numpy, matplotlib, opencv-python
- scikit-image
- bm3d
- tqdm

可以使用 pip 安装所有依赖：
```bash
pip install numpy matplotlib opencv-python scikit-image tqdm
pip install bm3d
pip install torch torchvision
```
如果 `bm3d` 安装失败，可尝试使用 conda 安装：
```bash
conda install -c conda-forge bm3d
```

## 使用方法
1. 将预训练的 DnCNN 权重文件 `model.pth` 放入 `pretrained/` 文件夹。
2. 在终端中运行：
   ```bash
   python denoise_comparison.py
   ```
3. 程序将输出各方法的 PSNR/SSIM 表格，并在当前目录生成对比图像 `comparison.png`。

## 实验结果
### 定量指标
| 方法                     | PSNR (dB) | SSIM  |
|--------------------------|-----------|-------|
| 噪声图像 (Noisy)         | 20.61     | 0.301 |
| DnCNN                    | 30.01     | 0.816 |
| BM3D                     | 29.70     | 0.796 |
| TV (ADMM)                | 28.33     | 0.729 |
| 小波软阈值 (ISTA/FISTA)  | 26.96     | 0.647 |

### 可视化对比
从comparison.png和表可见，DnCNN 取得了最佳性能，BM3D 紧随其后，而传统优化方法（TV 和小波）效果略逊一筹，这与文献中报告的结论一致。

## 论文
本项目的实验分析整理为学术论文 `DnCNN.pdf`（LaTeX 源码为 `DnCNN.tex`），可在仓库中直接查看。

## 参考文献
- [DnCNN: Beyond a Gaussian Denoiser](https://ieeexplore.ieee.org/document/7839189)
- [BM3D: Image Denoising by Sparse 3D Transform-Domain Collaborative Filtering](https://ieeexplore.ieee.org/document/4271520)
- [Nonlinear Total Variation Based Noise Removal Algorithms](https://doi.org/10.1016/0167-2789(92)90242-F)
- [A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems](https://epubs.siam.org/doi/10.1137/080716542)

