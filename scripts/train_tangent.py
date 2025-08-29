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
from utils.device import get_device
device = get_device()

taskname = 'Square'

################################
# 初始化模型
################################
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PARAMS = {"scale": 5.0}
score_model = MLPScore(dim=2, hiddens=[64, 64, 64, 64])
state = torch.load('saved_model/' + taskname + '/score.pth', map_location=device)
score_model.load_state_dict(state)
score_model.to(device)
score_model.eval()

tangent_model = TangentNet(hidden=128)
trainer = TangentNetTrainer(tangent_model, score_model, k=5, lr=1e-3, device=device,
                 lambda_unit=1.0, lambda_orth=1.0, lambda_dir=1.0)

################################
# 训练 tangent 网络
################################
num_steps = 10000
batch_size = 512 
scale = PARAMS["scale"]
losses = []
for step in range(1, num_steps + 1):
    x = torch.empty(batch_size, 2).uniform_(-scale, scale).to(device)
    loss, lu, lo, ld = trainer.train_step(x)
    losses.append(loss)
    if step % 100 == 0:
        print(f"[{step}] loss={loss:.4f} unit={lu:.4f} orth={lo:.4f} dir={ld:.16f}")

    # # 定期保存
    # if step % 1000 == 0:
    #     torch.save({
    #         'step': step,
    #         'trainer_state': trainer.state_dict(),  # 如果 trainer 是 nn.Module
    #         'optimizer_state': trainer.optimizer.state_dict(),  # 保存优化器
    #     }, f"checkpoint_step_{step}.pth")

tangent_model.save('saved_model/' + taskname + '/tangent.pth')
plot_losses(losses, taskname , title="Tangent Network Training Loss", save=True, smooth=True, method="ema", alpha=0.95)

################################
# 可视化tangent向量场 
################################
x_bounds = [-scale,scale]
y_bounds = [-scale,scale]
legend_size=24
markerscale=1.8
fig, ax = plt.subplots(figsize=(7,7))
plot_tangent_vector_field(tangent_model, x_bounds=x_bounds, y_bounds=y_bounds, device='cuda',num_grid=30, scale=1.5, ax= ax, title=None)

# # 2. 画散点
# scatter_sampleable(p_data, num_samples=1000, ax=ax, color='blue', alpha=0.5, label='$p_{data}$')

# 3. 整理图像格式
ax.set_title("Trained Tangent Vector Field for " + taskname, fontsize=16)
# ax.legend(fontsize=legend_size, markerscale=markerscale)
ax.set_xlim(x_bounds)
ax.set_ylim(y_bounds)
ax.grid(True)
plt.savefig('saved_figure/' + taskname + '/tangent.pdf', format = 'pdf', dpi=300)
plt.tight_layout()
plt.show()
