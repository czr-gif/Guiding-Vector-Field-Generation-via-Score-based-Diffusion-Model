import numpy as np
from matplotlib import pyplot as plt
import torch

from core.schedules import *
from utils.plotting import *
from models.NNmodels import MLPScore, TangentNet
from core.base_distributions import GaussianMixture,Gaussian,  MoonsSampleable, CirclesSampleable, UniformBox, SingleCircleSampleable, InputdataSampleable,TwoPolygonsSampleable,TwoCirclesSampleable, SquareSampleable, Star5Sampleable, HexagonSampleable, OctagonSampleable
from typing import Callable
from utils.device import get_device
device = get_device()

taskname =  'experiment1'
if taskname == 'Square':
    p_data = SquareSampleable(device=device)
if taskname == 'Hexagon':
    p_data = HexagonSampleable(device=device)
if taskname == 'Octagon':
    p_data = OctagonSampleable(device=device)
if taskname == 'experiment1':
    p_data = CirclesSampleable(device = device,scale=2)
if taskname == 'experiment2':
    p_data = TwoCirclesSampleable(device=device)

################################
# 初始化模型
################################
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PARAMS = {"scale": 3.0}
score_model = MLPScore(dim=2, hiddens=[64, 64, 64, 64])
state = torch.load('saved_model/' + taskname + '/score.pth', map_location=device)
score_model.load_state_dict(state)
score_model.to(device)
score_model.eval()

tangent_model = TangentNet.load('saved_model/' + taskname + '/tangent.pth', map_location=device)
tangent_model.to(device)

# print(tangent_model.forward(torch.tensor([[1.0,0.0]])))
################################
# 组合场控制仿真
################################

def integrate_fields(
    x0: torch.Tensor,
    score_model: Callable,
    tangent_model: Callable,
    lam: float,
    dt: float,
    steps: int,
    device: torch.device,
    momentum: float = 0.0 # 动量系数
) -> np.ndarray:
    """
    Integrate a trajectory using a mixture of score-based and tangent vector fields.

    Args:
        x0: Initial state, shape (N, 2)
        score_model: Callable (x, t) -> score field tensor
        tangent_model: Callable (x) -> tangent field tensor
        lam: blending coefficient between score and tangent fields
        dt: integration step size
        steps: number of integration steps
        device: torch device

    Returns:
        np.ndarray of shape (steps+1)*N x 2, trajectory over time
    """
    x = x0.clone().to(device)
    traj = [x.clone()]
    v_prev = torch.zeros_like(x)  # 初始化动量为零
    with torch.no_grad():
        for _ in range(steps):
            t = torch.ones((1, 1), device=device)

            # score 场
            s_theta = score_model(x, t)
            norms = torch.norm(s_theta, dim=1, keepdim=True) + 1e-8
            sc = torch.tanh(0.2 * norms)
            s_theta = s_theta / norms * sc

            # tangent 场
            tangent_vec = tangent_model(x)

            # 混合场
            u = lam * s_theta + (1 - lam) * tangent_vec
            u = u / torch.norm(u, dim=1, keepdim=True)

            # 更新
            # x = x + dt * u
            # 动量更新
            v_prev = momentum * v_prev + (1 - momentum) * u

            # 更新位置
            x = x + dt * v_prev 
            traj.append(x.clone())
    print(type(traj))
    traj = torch.stack([t.cpu() for t in traj]).numpy().reshape(-1, 2)
    return traj


x0_list = [torch.tensor([[0.0, 0.0]], dtype=torch.float32),
           torch.tensor([[2.5, 2.5]], dtype=torch.float32)]

traj_list = []

for x0 in x0_list:
    traj = integrate_fields(
        x0=x0,
        score_model=score_model,
        tangent_model=tangent_model,
        lam=0.5,
        dt=0.01,
        steps=5000,
        device=device
    )
    traj_list.append(traj)

# print(traj)
################################
# 绘制结果
################################
# 网格向量场（用于可视化组合场）
X, Y = np.meshgrid(np.linspace(-PARAMS["scale"], PARAMS["scale"], 30),
                   np.linspace(-PARAMS["scale"], PARAMS["scale"], 30))
grid = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=-1), dtype=torch.float32, device=device)
t_grid = torch.ones((grid.shape[0], 1), device=device)

score_grid = score_model(grid, t_grid)
norms = torch.norm(score_grid, dim=1, keepdim=True) + 1e-8
sc = torch.tanh(0.2 * norms)
score_grid = score_grid / norms * sc

tangent_grid = tangent_model(grid)
combo_grid = score_grid + tangent_grid
U = combo_grid[:, 0].cpu().detach().numpy()
V = combo_grid[:, 1].cpu().detach().numpy()

fig, ax = plt.subplots(figsize=(7,7))
scatter_sampleable(p_data, num_samples=100, ax=ax, color='black', alpha=0.8, label='$p_{data}$')
# 绘制矢量场
plt.quiver(X, Y, U.reshape(X.shape), V.reshape(Y.shape), color='gray', alpha=0.5)

colors = ['red', 'blue']
markers = ['o', 's']  # 起点不同标记
for i, traj in enumerate(traj_list):
    # 轨迹线
    plt.plot(traj[:, 0], traj[:, 1], color=colors[i], linewidth=2.5,alpha=0.8, label=f'Agent {i+1} Trajectory')
    # 起点标记
    plt.scatter(traj[0, 0], traj[0, 1], c=colors[i], edgecolors='black',
                s=120, marker=markers[i], label=f'Agent {i+1} Start')
    # 可选：终点标记
    plt.scatter(traj[-1, 0], traj[-1, 1], c='white', edgecolors=colors[i],
                s=80, marker=markers[i], label=f'Agent {i+1} End')

plt.axis('equal')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper left', fontsize=10)
plt.title("Robot Trajectories under Combined Field", fontsize=14)
plt.xlabel("X")
plt.ylabel("Y")
plt.tight_layout()
plt.savefig('saved_figure/' + taskname + '/Pathfollowing.pdf', format='pdf', dpi=300)
plt.show()