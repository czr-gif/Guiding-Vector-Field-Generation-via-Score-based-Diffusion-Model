from abc import ABC, abstractmethod
from typing import Optional, List, Type, Tuple, Dict
import math

import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.distributions as D
from core.schedules import *
from core.paths import GaussianConditionalProbabilityPath, LinearConditionalProbabilityPath, UniformProbabilityPath
from core.base_distributions import GaussianMixture,Gaussian,  MoonsSampleable, CirclesSampleable, UniformBox, SingleCircleSampleable, InputdataSampleable,TwoPolygonsSampleable,TwoCirclesSampleable, SquareSampleable, Star5Sampleable
from utils.plotting import *
from core.dynamics import ConditionalVectorFieldODE, ConditionalVectorFieldSDE, LearnedVectorFieldODE
from core.simulators import EulerSimulator,EulerMaruyamaSimulator, record_every
from models.NNmodels import MLPScore , ConditionalScoreMatchingTrainer
import os
from utils.device import get_device
# device = get_device()

# -----------------------------
# 主绘图函数
# -----------------------------
def plot_circles_distribution(mode='2D', grid_size=200, num_samples=500, sigma=0.2, save_dir="./saved_figure/discussion", show=True):
    device = torch.device('cpu')
    circle_dist = CirclesSampleable(device=device, noise=0.05, scale=4.0)

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 网格生成
    x = torch.linspace(-5, 5, grid_size)
    y = torch.linspace(-5, 5, grid_size)
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    # 计算 log-density
    logp = circle_dist.log_density(grid_points, sigma=sigma)
    logp_grid = logp.reshape(grid_size, grid_size).cpu().numpy()

    # 采样一些点
    samples = circle_dist.sample(num_samples).cpu().numpy()

    # 替换 -inf 或极小值
    min_val = logp_grid[~np.isinf(logp_grid)].min()
    logp_grid[np.isinf(logp_grid)] = min_val - 1

    if mode == '2D':
        plt.figure(figsize=(6,6))
        plt.contourf(xx.numpy(), yy.numpy(), logp_grid, levels=100, cmap='viridis')
        plt.colorbar(label='log-density')
        plt.scatter(samples[:,0], samples[:,1], color='red', s=10, label='samples', alpha=0.6)
        # 关键：限制坐标轴范围，保证正方形显示
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('2D Log-Density of Concentric Circles with Samples')
        # plt.axis('equal')
        plt.legend()
        save_path = os.path.join(save_dir, "log_density_2D.pdf")
        plt.savefig(save_path, format = 'pdf', dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    elif mode == '3D':
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(xx.numpy(), yy.numpy(), logp_grid, cmap='viridis', alpha=0.8, edgecolor='none')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('log-density')
        ax.set_title('3D Log-Density of Concentric Circles with Samples')
        ax.view_init(elev=45, azim=45)
        save_path = os.path.join(save_dir, "log_density_3D.pdf")
        plt.savefig(save_path, format = 'pdf', dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
    else:
        raise ValueError("mode must be '2D' or '3D'")

    print(f"✅ Figure saved at: {save_path}")


# -----------------------------
# 调用示例
# -----------------------------
if __name__ == "__main__":
    plot_circles_distribution(mode='2D', show=False)  # 保存 2D
    plot_circles_distribution(mode='3D', show=False)  # 保存 3D