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
from models.NNmodels import MLPScore , ConditionalScoreMatchingTrainer, TangentNet, TangentNetTrainer

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

tangent_model = TangentNet(hidden=128)
trainer = TangentNetTrainer(tangent_model, score_model, k=5, lr=1e-3, device=device)

################################
# 训练 tangent 网络
################################
num_steps = 5000
batch_size = 512
scale = PARAMS["scale"]

for step in range(1, num_steps + 1):
    x = torch.empty(batch_size, 2).uniform_(-scale, scale).to(device)
    loss, lu, ld = trainer.train_step(x)

    if step % 100 == 0:
        print(f"[{step}] loss={loss:.4f} unit={lu:.4f} dir={ld:.4f}")

    # # 定期保存
    # if step % 1000 == 0:
    #     torch.save({
    #         'step': step,
    #         'trainer_state': trainer.state_dict(),  # 如果 trainer 是 nn.Module
    #         'optimizer_state': trainer.optimizer.state_dict(),  # 保存优化器
    #     }, f"checkpoint_step_{step}.pth")

tangent_model.save('circletangent.pth')


################################
# 可视化tangent向量场
################################
plot_drift_and_tangent_fields(
    ode_model=LearnedVectorFieldODE(score_model),
    tangent_model=tangent_model,
    x_bounds=(-5, 5),
    y_bounds=(-5, 5),
    device=device,
    t_value=1.0,
    num_grid=40,
    scale=1.5
)
