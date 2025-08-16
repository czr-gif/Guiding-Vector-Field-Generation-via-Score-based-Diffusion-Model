from abc import ABC, abstractmethod
from typing import Optional, List, Type, Tuple, Dict
import math

import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.distributions as D
from core.schedules import *
from core.paths import GaussianConditionalProbabilityPath, LinearConditionalProbabilityPath, UniformProbabilityPath
from core.base_distributions import GaussianMixture,Gaussian,  MoonsSampleable, CirclesSampleable, UniformBox, SingleCircleSampleable
from utils.plotting import *
from core.dynamics import ConditionalVectorFieldODE, ConditionalVectorFieldSDE, LearnedVectorFieldODE
from core.simulators import EulerSimulator,EulerMaruyamaSimulator, record_every
from models.NNmodels import MLPScore , ConditionalScoreMatchingTrainer
import os

taskname = 'doublecircles'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PARAMS = {
    "scale": 5.0,
    "target_scale": 10.0,
    "target_std": 1.0,
}

# 初始化采样器：圆半径为3，圆心在 (0,0)，采样点打乱
# sampler = SingleCircleSampleable(device=device, radius=3.0)
# p_data = sampler.sample(500)
p_data = CirclesSampleable(device)

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
losses = trainer.train(num_epochs=8000, device=device, lr=1e-3, batch_size=1000)
score_model.save('saved_model/' + taskname + '/score.pth')
# score_model.load('circlescore.pth')
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
ax.set_ylim(y_bounds)
ax.grid(True)
plt.savefig('saved_figure/' + taskname + '/score.pdf', format = 'pdf', dpi=300)
plt.tight_layout()
plt.show()
