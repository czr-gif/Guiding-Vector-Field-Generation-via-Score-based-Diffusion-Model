import torch
import torch.nn.functional as F
import numpy as np
from models.NNmodels import MLPScore , ConditionalScoreMatchingTrainer, TangentNet, TangentNetTrainer
# from rknn.api import RKNN

device = 'cpu'
taskname = 'experiment1'
state_dims = []
action_dims = []

hidden_dim = 128
agents = []
# for i in range(3):
#     agents.append(MADDPG.DDPG(state_dims[i], action_dims[i], critic_input_dim,hidden_dim, actor_lr, critic_lr, "cuda")) 
score_model = MLPScore(dim=2, hiddens=[64, 64, 64, 64])
state = torch.load('saved_model/' + taskname + '/score.pth', map_location=device)
score_model.load_state_dict(state)
score_model.to(device)
score_model.eval()

tangent_model = TangentNet.load('saved_model/' + taskname + '/tangent.pth', map_location=device)
tangent_model.to(device)
tangent_model.eval()

dummy_x = torch.randn(1, 2)  # 平面 (dim=2)
dummy_t = torch.randn(1, 1)  # 时间变量
torch.onnx.export(
    score_model,
    (dummy_x, dummy_t),
    "trans_model/"+taskname+"/score.onnx",
    input_names=["x", "t"],
    output_names=["output"],
    opset_version=13,
    dynamic_axes={
        "x": {0: "batch_size"},
        "t": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)


torch.onnx.export(
    tangent_model,
    dummy_x,
    "trans_model/"+taskname+"/tangent.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=13,  # 推荐使用较高版本以支持更多算子
    dynamic_axes={
        "input": {0: "batch_size"},  # 支持动态 batch
        "output": {0: "batch_size"}
    }
)


# state = torch.tensor(np.array([1.5,1.5,0,0,]), dtype=torch.float, device="cpu")
# action = actor_0(state)
# action = action.unsqueeze(0)  
# print(action)