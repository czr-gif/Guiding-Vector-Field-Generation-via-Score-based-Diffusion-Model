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

################################
# model init
################################
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PARAMS = {"scale": 5.0}
score_model = MLPScore(dim=2, hiddens=[64, 64, 64, 64])
state = torch.load('circlescore.pth', map_location=device)
score_model.load_state_dict(state)
score_model.to(device)
score_model.eval()

tangent_model = TangentNet.load('circletangent.pth', map_location=device)
tangent_model.to(device)

################################
# draw tangent vector field
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
