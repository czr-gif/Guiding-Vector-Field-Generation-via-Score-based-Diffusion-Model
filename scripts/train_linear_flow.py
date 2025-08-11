from abc import ABC, abstractmethod
from typing import Optional, List, Type, Tuple, Dict
import math

import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.distributions as D
from core.schedules import *
from core.paths import GaussianConditionalProbabilityPath, LinearConditionalProbabilityPath
from core.base_distributions import GaussianMixture,Gaussian,  MoonsSampleable, CirclesSampleable, UniformBox
from utils.plotting import *
from core.dynamics import ConditionalVectorFieldODE, ConditionalVectorFieldSDE, LearnedVectorFieldODE
from core.simulators import EulerSimulator,EulerMaruyamaSimulator, record_every
from models.NNmodels import ConditionalFlowMatchingTrainer, MLPVectorField

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PARAMS = {
    "scale": 15.0,
    "target_scale": 10.0,
    "target_std": 1.0,
}

p_data = CirclesSampleable(device,noise=0)
p_simple = Gaussian.isotropic(p_data.dim, 6.0)
# Construct conditional probability path
# path = GaussianConditionalProbabilityPath(
#     p_data = p_data,
#     alpha = LinearAlpha(),
#     beta = SquareRootBeta(),
#     p_simple = Gaussian.isotropic(p_data.dim, 1.0)
# ).to(device)
path = LinearConditionalProbabilityPath(
    p_simple = p_simple ,
    p_data = p_data
).to(device)

# Construct learnable vector field
flow_model = MLPVectorField(dim=2, hiddens=[64,64,64,64])

# Construct trainer
trainer = ConditionalFlowMatchingTrainer(path, flow_model)
losses = trainer.train(num_epochs=10000, device=device, lr=1e-3, batch_size=1000)


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
    ode_model=LearnedVectorFieldODE(flow_model),
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
plt.tight_layout()
# plt.show()