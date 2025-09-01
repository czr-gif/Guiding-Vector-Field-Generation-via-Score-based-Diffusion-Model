import numpy as np
import onnxruntime as ort
from matplotlib import pyplot as plt

# ONNX 模型路径
taskname = "doublecircles"
score_onnx_path = f"trans_model/{taskname}/score.onnx"
tangent_onnx_path = f"trans_model/{taskname}/tangent.onnx"

# 参数
PARAMS = {"scale": 5.0}

################################
# 加载 ONNX 模型
################################
score_sess = ort.InferenceSession(score_onnx_path)
tangent_sess = ort.InferenceSession(tangent_onnx_path)

# 获取输入名
score_input_names = [inp.name for inp in score_sess.get_inputs()]  # ['x','t']
tangent_input_names = [inp.name for inp in tangent_sess.get_inputs()]  # ['input']

################################
# 组合场仿真函数
################################
def integrate_fields_onnx(x0: np.ndarray, lam: float, dt: float, steps: int, momentum: float = 0.0):
    """
    x0: (N,2) numpy array
    """
    x = x0.astype(np.float32)
    traj = [x.copy()]
    v_prev = np.zeros_like(x)
    
    for _ in range(steps):
        # score 输入
        x_input = x.reshape(1, -1)
        t_input = np.ones((1, 1), dtype=np.float32)
        score_out = score_sess.run(None, {score_input_names[0]: x_input, score_input_names[1]: t_input})[0]
        s_theta = score_out
        norm = np.linalg.norm(s_theta, axis=1, keepdims=True) + 1e-8
        sc = np.tanh(0.2 * norm)
        s_theta = s_theta / norm * sc

        # tangent 输入
        tangent_out = tangent_sess.run(None, {tangent_input_names[0]: x_input})[0]

        # 混合场
        u = lam * s_theta + (1 - lam) * tangent_out
        u = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-8)

        # 动量更新
        v_prev = momentum * v_prev + (1 - momentum) * u
        x = x + dt * v_prev
        traj.append(x.copy())
    
    traj = np.vstack(traj)
    return traj

################################
# 初始化轨迹起点
################################
x0_list = [np.array([[1.0, -0.0]]), np.array([[-4.5, 0.0]])]
traj_list = [integrate_fields_onnx(x0, lam=0.5, dt=0.01, steps=4000) for x0 in x0_list]

################################
# 绘图
################################
X, Y = np.meshgrid(np.linspace(-PARAMS["scale"], PARAMS["scale"], 30),
                   np.linspace(-PARAMS["scale"], PARAMS["scale"], 30))

fig, ax = plt.subplots(figsize=(7,7))

colors = ['red', 'blue']
markers = ['o', 's']
for i, traj in enumerate(traj_list):
    plt.plot(traj[:,0], traj[:,1], color=colors[i], linewidth=2.5, label=f'Agent {i+1} Trajectory')
    plt.scatter(traj[0,0], traj[0,1], c=colors[i], edgecolors='black', s=120, marker=markers[i], label=f'Agent {i+1} Start')
    plt.scatter(traj[-1,0], traj[-1,1], c='white', edgecolors=colors[i], s=80, marker=markers[i], label=f'Agent {i+1} End')

plt.axis('equal')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper left', fontsize=10)
plt.title("Robot Trajectories under ONNX Combined Field", fontsize=14)
plt.xlabel("X")
plt.ylabel("Y")
plt.tight_layout()
plt.show()
