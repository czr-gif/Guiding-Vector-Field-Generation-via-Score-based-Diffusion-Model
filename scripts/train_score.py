from abc import ABC, abstractmethod
from typing import Optional, List, Type, Tuple, Dict
import math

import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.distributions as D
from core.schedules import *
from core.paths import GaussianConditionalProbabilityPath, LinearConditionalProbabilityPath, UniformProbabilityPath
from core.base_distributions import GaussianMixture,Gaussian,  MoonsSampleable, CirclesSampleable, UniformBox, SingleCircleSampleable, InputdataSampleable,TwoPolygonsSampleable,TwoCirclesSampleable, SquareSampleable, Star5Sampleable, HexagonSampleable, OctagonSampleable
from utils.plotting import *
from core.dynamics import ConditionalVectorFieldODE, ConditionalVectorFieldSDE, LearnedVectorFieldODE
from core.simulators import EulerSimulator,EulerMaruyamaSimulator, record_every
from models.NNmodels import MLPScore , ConditionalScoreMatchingTrainer
import os
from utils.device import get_device
device = get_device()

taskname = 'Octagon'

dirs_to_create = [
    os.path.join("saved_model", taskname),
    os.path.join("saved_figure", taskname)
]

for directory in dirs_to_create:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory created: {directory}")
    else:
        print(f"Directory already exists: {directory}")

PARAMS = {
    "scale": 5.0,
    "target_scale": 10.0,
    "target_std": 1.0,
}

# 初始化采样器
if taskname == 'circle':
    p_data = SingleCircleSampleable(device=device, radius=3.0)
    # p_data = sampler.sample(500)
if taskname == 'doublecircles':
    p_data = CirclesSampleable(device)
if taskname == 'westlake':
    np_points = np.load('utils/westlake_edges.npy')
    points_tensor = torch.tensor(np_points, dtype=torch.float32)
    p_data = InputdataSampleable(device=device, points=points_tensor, shuffle=True, bound=4.0)
if taskname == 'Polygon':
    p_data = TwoPolygonsSampleable(device=device)
if taskname == 'SperatedCircles':
    p_data = TwoCirclesSampleable(device=device)
if taskname == 'Square':
    p_data = SquareSampleable(device=device)
if taskname == 'Star5':
    p_data = Star5Sampleable(device=device)
if taskname == 'Hexagon':
    p_data = HexagonSampleable(device=device)
if taskname == 'Octagon':
    p_data = OctagonSampleable(device=device)

# Construct conditional probability path
path = UniformProbabilityPath(
    p_data = p_data, 
    alpha = LinearAlpha(),
    beta = SquareRootBeta()
).to(device)

# Construct learnable vector field
score_model = MLPScore(dim=2, hiddens=[64,64,64,64]).to(device)
# Construct trainer
trainer = ConditionalScoreMatchingTrainer(path, score_model)
losses = trainer.train(num_epochs=10000, device=device, lr=1e-3, batch_size=1000)
score_model.save('saved_model/' + taskname + '/score.pth')
# score_model.load('saved_model/' + taskname + '/score.pth')
plot_losses(losses, taskname , title="Score Network Training Loss", save=True, smooth=True, method="ema", alpha=0.95)
#######################
# Change these values #
#######################
num_samples = 1000
num_timesteps = 1000
num_marginals = 3

##############
# Setup Plot #
##############

scale = PARAMS["scale"]
x_bounds = [-scale,scale]
y_bounds = [-scale,scale]
legend_size=24
markerscale=1.8

###########################################
# Graph Samples from Learned Marginal ODE #
###########################################
fig, ax = plt.subplots(figsize=(7,7))

# 1. 画向量场（背景流场）
plot_drift_vector_field(
    ode_model=LearnedVectorFieldODE(score_model),
    x_bounds=x_bounds,
    y_bounds=y_bounds,
    t_value=1,
    num_grid=40,
    ax=ax,  # 将 ax 传进去以共享画布
    title=None  # 用统一标题
)

# 2. 画散点
scatter_sampleable(p_data, num_samples=1000, ax=ax, color='blue', alpha=0.5, label='$p_{data}$')

# 3. 整理图像格式
ax.set_title("Drift Vector Field + $p_{data}$ Samples (t=1)", fontsize=16)
ax.legend(fontsize=legend_size, markerscale=markerscale)
ax.set_xlim(x_bounds)
ax.set_ylim(y_bounds) # type: ignore
ax.grid(True)
plt.savefig('saved_figure/' + taskname + '/score.pdf', format = 'pdf', dpi=300)
plt.tight_layout()
plt.show()
