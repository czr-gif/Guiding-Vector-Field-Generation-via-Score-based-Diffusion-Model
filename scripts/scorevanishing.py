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

# 初始化
def plot_circles_distribution(mode='2D', grid_size=200, num_samples=500, sigma=0.2):
    device = torch.device('cpu')
    circle_dist = CirclesSampleable(device=device, noise=0.05, scale=4.0)

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

    # 替换 -inf 或极小值（安全起见）
    min_val = logp_grid[~np.isinf(logp_grid)].min()
    logp_grid[np.isinf(logp_grid)] = min_val - 1

    if mode == '2D':
        # -----------------------------
        # 2D 热力图
        # -----------------------------
        plt.figure(figsize=(6,6))
        plt.contourf(xx.numpy(), yy.numpy(), logp_grid, levels=100, cmap='viridis')
        plt.colorbar(label='log-density')
        plt.scatter(samples[:,0], samples[:,1], color='red', s=10, label='samples', alpha=0.6)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('2D Log-Density of Concentric Circles with Samples')
        plt.axis('equal')
        plt.legend()
        plt.show()

    elif mode == '3D':
        # -----------------------------
        # 3D 曲面
        # -----------------------------
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(xx.numpy(), yy.numpy(), logp_grid, cmap='viridis', alpha=0.8, edgecolor='none')
        # ax.scatter(samples[:,0], samples[:,1], np.full(samples.shape[0], logp_grid.min()-0.5),
        #            color='red', s=20, label='samples')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('log-density')
        ax.set_title('3D Log-Density of Concentric Circles with Samples')
        ax.view_init(elev=45, azim=45)
        ax.legend()
        plt.show()
    else:
        raise ValueError("mode must be '2D' or '3D'")

# -----------------------------
# 调用示例
# -----------------------------
if __name__ == "__main__":
    # 画二维热力图
    plot_circles_distribution(mode='2D')
    # 画三维曲面图
    plot_circles_distribution(mode='3D')