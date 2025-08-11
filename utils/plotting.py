from abc import ABC, abstractmethod
from typing import Optional, List, Type, Tuple, Dict
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.axes._axes import Axes
import torch
import seaborn as sns
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from core.base_distributions import Sampleable, Density

def hist2d_samples(samples, ax: Optional[Axes] = None, bins: int = 200, scale: float = 5.0, percentile: int = 99, **kwargs):
    H, xedges, yedges = np.histogram2d(samples[:, 0], samples[:, 1], bins=bins, range=[[-scale, scale], [-scale, scale]])
    
    # Determine color normalization based on the 99th percentile
    cmax = np.percentile(H, percentile)
    cmin = 0.0
    norm = cm.colors.Normalize(vmax=cmax, vmin=cmin)
    
    # Plot using imshow for more control
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax.imshow(H.T, extent=extent, origin='lower', norm=norm, **kwargs)

def hist2d_sampleable(sampleable: Sampleable, num_samples: int, ax: Optional[Axes] = None, bins=200, scale: float = 5.0, percentile: int = 99, **kwargs):
    assert sampleable.dim == 2
    if ax is None:
        ax = plt.gca()
    samples = sampleable.sample(num_samples).detach().cpu() # (ns, 2)
    hist2d_samples(samples, ax, bins, scale, percentile, **kwargs)

def scatter_sampleable(sampleable: Sampleable, num_samples: int, ax: Optional[Axes] = None, **kwargs):
    assert sampleable.dim == 2
    if ax is None:
        ax = plt.gca()
    samples = sampleable.sample(num_samples) # (ns, 2)
    ax.scatter(samples[:,0].cpu(), samples[:,1].cpu(), **kwargs)

def kdeplot_sampleable(sampleable: Sampleable, num_samples: int, ax: Optional[Axes] = None, **kwargs):
    assert sampleable.dim == 2
    if ax is None:
        ax = plt.gca()
    samples = sampleable.sample(num_samples) # (ns, 2)
    sns.kdeplot(x=samples[:,0].cpu(), y=samples[:,1].cpu(), ax=ax, **kwargs)

def imshow_density(density: Density, x_bounds: Tuple[float, float], y_bounds: Tuple[float, float], bins: int, ax: Optional[Axes] = None, x_offset: float = 0.0, **kwargs):
    if ax is None:
        ax = plt.gca()
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    x = torch.linspace(x_min, x_max, bins).to(device) + x_offset
    y = torch.linspace(y_min, y_max, bins).to(device)
    X, Y = torch.meshgrid(x, y)
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    density = density.log_density(xy).reshape(bins, bins).T
    im = ax.imshow(density.cpu(), extent=[x_min, x_max, y_min, y_max], origin='lower', **kwargs)

def contour_density(density: Density, bins: int, scale: float, ax: Optional[Axes] = None, x_offset:float = 0.0, **kwargs):
    if ax is None:
        ax = plt.gca()
    x = torch.linspace(-scale + x_offset, scale + x_offset, bins).to(device)
    y = torch.linspace(-scale, scale, bins).to(device)
    X, Y = torch.meshgrid(x, y)
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    density = density.log_density(xy).reshape(bins, bins).T
    im = ax.contour(density.cpu(), origin='lower', **kwargs)

# def plot_drift_vector_field(ode_model, x_bounds, y_bounds, device='cuda',
#                             t_value=0.5, num_grid=30, scale=1.5, ax: Optional[Axes] = None, title=None):
#     """
#     可视化 ODE 模型返回的标准化 drift vector field。

#     参数：
#     - ode_model: 一个具有 `.drift_coefficient(x, t)` 接口的模型（例如 LearnedVectorFieldODE 实例）
#     - x_bounds: tuple (xmin, xmax)，x 轴范围
#     - y_bounds: tuple (ymin, ymax)，y 轴范围
#     - device: 计算设备（默认 'cuda'）
#     - t_value: 固定时间 t，用于生成向量场
#     - num_grid: 网格分辨率
#     - scale: quiver 图中箭头缩放因子
#     - title: 图标题（默认自动生成）

#     返回：
#     - None（直接绘图）
#     """
    
#     # 1. 构造网格点
    
#     x_vals = torch.linspace(x_bounds[0], x_bounds[1], num_grid)
#     y_vals = torch.linspace(y_bounds[0], y_bounds[1], num_grid)
#     X, Y = torch.meshgrid(x_vals, y_vals, indexing='ij')
#     points = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1).to(device)  # (N, 2)

#     # 2. 时间张量
#     t_fixed = torch.full((points.shape[0], 1), t_value).to(device)

#     # 3. 获取并标准化向量
#     with torch.no_grad():
#         vectors = ode_model.drift_coefficient(points, t_fixed).detach().cpu()
#         norms = torch.norm(vectors, dim=1, keepdim=True) + 1e-8
#         sc = torch.tanh(0.2*norms)
#         vectors = vectors/norms * sc * 0.3 # 缩放

#     # 4. 转为画图格式
#     U = vectors[:, 0].reshape(num_grid, num_grid)
#     V = vectors[:, 1].reshape(num_grid, num_grid)

#     # 5. 可视化
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(6,6))
#     ax.set_xlim(*x_bounds)
#     ax.set_ylim(*y_bounds)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_title(title if title else f"Drift Vector Field at t={t_value:.2f}")

#     ax.quiver(X.cpu(), Y.cpu(), U, V, angles='xy', scale_units='xy',
#               scale=scale, width=0.003, alpha=0.7)
#     plt.tight_layout()
#     plt.show()

# def plot_tangent_vector_field(tangent_model, x_bounds, y_bounds, device='cuda',
#                               num_grid=30, scale=1.5, ax: Optional[Axes] = None, title=None):
#     """
#     可视化 TangentNet 输出的向量场（假设 forward 输入 (N,2) 输出 (N,2)）。

#     参数：
#     - tangent_model: 一个输入 (x, y) 输出 (vx, vy) 的模型
#     - x_bounds: tuple (xmin, xmax)，x 轴范围
#     - y_bounds: tuple (ymin, ymax)，y 轴范围
#     - device: 计算设备（默认 'cuda'）
#     - num_grid: 网格分辨率
#     - scale: quiver 图中箭头缩放因子
#     - title: 图标题（默认自动生成）
#     """

#     tangent_model.eval()
    
#     # 1. 构造网格点
#     x_vals = torch.linspace(x_bounds[0], x_bounds[1], num_grid)
#     y_vals = torch.linspace(y_bounds[0], y_bounds[1], num_grid)
#     X, Y = torch.meshgrid(x_vals, y_vals, indexing='ij')
#     points = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1).to(device)  # (N, 2)

#     # 2. 获取并标准化向量
#     with torch.no_grad():
#         vectors = tangent_model(points).detach().cpu()  # (N, 2)
#         vectors = vectors * 0.3
#     # 3. 转为画图格式
#     U = vectors[:, 0].reshape(num_grid, num_grid)
#     V = vectors[:, 1].reshape(num_grid, num_grid)

#     # 4. 可视化
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(6, 6))
#     ax.set_xlim(*x_bounds)
#     ax.set_ylim(*y_bounds)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_title(title if title else "Tangent Vector Field")

#     ax.quiver(X.cpu(), Y.cpu(), U, V, angles='xy', scale_units='xy',
#               scale=scale, width=0.003, alpha=0.7)
#     plt.tight_layout()
#     plt.show()

def plot_drift_vector_field(ode_model, x_bounds, y_bounds, device='cuda',
                            t_value=0.5, num_grid=30, scale=1.5, ax: Optional[Axes] = None, title=None):
    # 1. 构造网格点
    x_vals = torch.linspace(x_bounds[0], x_bounds[1], num_grid)
    y_vals = torch.linspace(y_bounds[0], y_bounds[1], num_grid)
    X, Y = torch.meshgrid(x_vals, y_vals, indexing='ij')
    points = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1).to(device)

    # 2. 时间张量
    t_fixed = torch.full((points.shape[0], 1), t_value).to(device)

    # 3. 获取并标准化向量
    with torch.no_grad():
        vectors = ode_model.drift_coefficient(points, t_fixed).detach().cpu()
        norms = torch.norm(vectors, dim=1, keepdim=True) + 1e-8
        sc = torch.tanh(0.2*norms)
        vectors = vectors / norms * sc * 0.3

    U = vectors[:, 0].reshape(num_grid, num_grid)
    V = vectors[:, 1].reshape(num_grid, num_grid)

    # 4. 绘制
    if ax is None:
        ax = plt.gca()
    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title if title else f"Drift Vector Field at t={t_value:.2f}")
    ax.quiver(X.cpu(), Y.cpu(), U, V, angles='xy', scale_units='xy',
              scale=scale, width=0.003, alpha=0.7)


def plot_tangent_vector_field(tangent_model, x_bounds, y_bounds, device='cuda',
                              num_grid=30, scale=1.5, ax: Optional[Axes] = None, title=None):
    tangent_model.eval()
    # 1. 网格
    x_vals = torch.linspace(x_bounds[0], x_bounds[1], num_grid)
    y_vals = torch.linspace(y_bounds[0], y_bounds[1], num_grid)
    X, Y = torch.meshgrid(x_vals, y_vals, indexing='ij')
    points = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1).to(device)

    # 2. 模型输出
    with torch.no_grad():
        vectors = tangent_model(points).detach().cpu()
        vectors = vectors * 0.3

    U = vectors[:, 0].reshape(num_grid, num_grid)
    V = vectors[:, 1].reshape(num_grid, num_grid)

    # 3. 绘制
    if ax is None:
        ax = plt.gca()
    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title if title else "Tangent Vector Field")
    ax.quiver(X.cpu(), Y.cpu(), U, V, angles='xy', scale_units='xy',
              scale=scale, width=0.003, alpha=0.7)

def plot_drift_and_tangent_fields(ode_model, tangent_model, 
                                  x_bounds, y_bounds, 
                                  device='cuda', 
                                  t_value=0.5, num_grid=30, scale=1.5):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    plot_drift_vector_field(ode_model, x_bounds, y_bounds, device,
                            t_value, num_grid, scale, ax=axes[0],
                            title=f"Drift Vector Field (t={t_value:.2f})")

    plot_tangent_vector_field(tangent_model, x_bounds, y_bounds, device,
                              num_grid, scale, ax=axes[1],
                              title="Tangent Vector Field")

    plt.tight_layout()
    plt.show()
