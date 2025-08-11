import numpy as np
from matplotlib import pyplot as plt
import torch

from core.schedules import *
from utils.plotting import *
from models.NNmodels import MLPScore, TangentNet

################################
# 初始化模型
################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PARAMS = {"scale": 5.0}
score_model = MLPScore(dim=2, hiddens=[64, 64, 64, 64])
state = torch.load('circlescore.pth', map_location=device)
score_model.load_state_dict(state)
score_model.to(device)
score_model.eval()

tangent_model = TangentNet.load('circletangent.pth', map_location=device)
tangent_model.to(device)

# print(tangent_model.forward(torch.tensor([[1.0,0.0]])))
################################
# 组合场控制仿真
################################

# 初始化
x = torch.tensor([[1.0, 0.0]], device=device)
dt = 0.01
steps = 5000
lam = 0.2

# 轨迹列表（先保存 GPU 张量）
traj = [x.clone()]
with torch.no_grad():
    # 你的积分循环

    for _ in range(steps):
        
        t = torch.ones((1, 1), device=device)

        # score 场
        s_theta = score_model(x, t)
        norms = torch.norm(s_theta, dim=1, keepdim=True) + 1e-8
        sc = torch.tanh(0.2 * norms)
        s_theta = s_theta / norms * sc

        # tangent 场
        tangent_vec = tangent_model(x)

        # 更新状态
        u = lam * s_theta + (1-lam)* tangent_vec
        u = u/torch.norm(u)
        x = x + dt * u

        traj.append(x.clone())

# 循环结束后一次性转 CPU
traj = torch.stack(traj).cpu().numpy()
traj = traj.reshape(-1, 2)
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

plt.figure(figsize=(6, 6))
plt.quiver(X, Y, U.reshape(X.shape), V.reshape(Y.shape), color='gray', alpha=0.5)
plt.plot(traj[:, 0], traj[:, 1], 'r-', linewidth=2, label='Trajectory')
plt.scatter(traj[0, 0], traj[0, 1], c='green', s=100, label='Start')
plt.scatter(traj[-1, 0], traj[-1, 1], c='blue', s=100, label='End')
plt.axis('equal')
plt.legend()
plt.title("Robot Trajectory under Combined Field")
plt.show()
