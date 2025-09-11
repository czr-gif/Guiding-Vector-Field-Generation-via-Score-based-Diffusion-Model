import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import torch
import numpy as np
from core.schedules import *
from utils.plotting import *
from models.NNmodels import MLPScore, TangentNet
from core.base_distributions import CirclesSampleable,TwoCirclesSampleable
from typing import Callable
from utils.device import get_device
device = get_device()

taskname = 'experiment2'

if taskname == 'experiment1':
    p_data = CirclesSampleable(device = device,scale=2)
if taskname == 'experiment2':
    p_data = TwoCirclesSampleable(device=device)

PARAMS = {"scale": 2.2}
score_model = MLPScore(dim=2, hiddens=[64, 64, 64, 64])
state = torch.load('saved_model/' + taskname + '/score.pth', map_location=device)
score_model.load_state_dict(state)
score_model.to(device)
score_model.eval()

tangent_model = TangentNet.load('saved_model/' + taskname + '/tangent.pth', map_location=device)
tangent_model.to(device)

# 读取数据，注意 sep="\t" 表示用 Tab 分隔
df = pd.read_csv("experiment_data/09-03_19-44-11_data.txt", sep="\t")

# 映射 RCchannels8 -> mode
def map_mode(val):
    if 1300 <= val <= 1600:
        return 1
    elif 1600 < val <= 2100:
        return 2
    else:
        return 0

df["mode"] = df["RCchannels8"].apply(map_mode)

# 转换为相对时间（秒）
df["time(s)"] = df["time(s)"] - df["time(s)"].iloc[0]

# 如果你想在终端里打印得更漂亮
# 选取论文中常用的核心字段
keep_cols = [
    # 时间
    "time(s)",

    # 位置
    "vrpn_pose_msg.position.x", "vrpn_pose_msg.position.y", "vrpn_pose_msg.position.z",
    "pos_kalman.x", "pos_kalman.y", "pos_kalman.z",

    # 速度
    "vrpn_twist_msg.linear.x", "vrpn_twist_msg.linear.y", "vrpn_twist_msg.linear.z",
    "vel_kalman.x", "vel_kalman.y", "vel_kalman.z",

    # 控制输入
    "throttle_ctl[0]", "motor_ctrl0", "motor_ctrl1", "motor_ctrl2", "motor_ctrl3",

    # 模式控制
    "mode",
]

# 有些列可能不存在（不同日志版本），这里做个安全过滤
keep_cols = [c for c in keep_cols if c in df.columns]

# 生成精简版 DataFrame
df_core = df[keep_cols]

motor_cols = ["motor_ctrl0", "motor_ctrl1", "motor_ctrl2", "motor_ctrl3"]

for col in motor_cols:
    df_core[col + "_N"] = df[col] * 0.005  # 新增列，单位 N


# 打印前几行看看
# print(tabulate(df_core.head(), headers="keys", tablefmt="psql"))

def split_by_mode(df, mode_col="mode", target_mode=2):
    """
    将 df 按照 target_mode 分段
    遇到其它 mode 时断开，返回一个 DataFrame 列表
    """
    segments = []
    current = []

    for _, row in df.iterrows():
        if row[mode_col] == target_mode:
            current.append(row)
        else:
            if current:  # 结束一个分段
                segments.append(pd.DataFrame(current))
                current = []
    if current:  # 最后一段
        segments.append(pd.DataFrame(current))
    return segments

# 分段
segments_mode2 = split_by_mode(df_core, target_mode=2)
segments_mode1 = split_by_mode(df_core, target_mode=1)

# Mode2 段的持续时间列表，从0开始
durations_mode2 = [0.0]  # 第一个元素为0
for seg in segments_mode2:
    duration = seg["time(s)"].iloc[-1] - seg["time(s)"].iloc[0]
    durations_mode2.append(duration)

# Mode1 段的持续时间列表，从0开始
durations_mode1 = [0.0]
for seg in segments_mode1:
    duration = seg["time(s)"].iloc[-1] - seg["time(s)"].iloc[0]
    durations_mode1.append(duration)



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
# 绘图

# 根据分支数量自动生成颜色
n_branches = len(segments_mode2)
colors = cm.get_cmap("Set1")

# 模式2：实线
for i, seg in enumerate(segments_mode2):
    plt.plot(seg["vrpn_pose_msg.position.x"],
             seg["vrpn_pose_msg.position.y"],
             label=f"SGVF-guided Flight",
             color=colors(i),   # 从 colormap 取颜色
             linestyle="-",
             linewidth=2)
    # 标记起始点，且只在图例里显示一次
    plt.scatter(seg["vrpn_pose_msg.position.x"].iloc[0],
                seg["vrpn_pose_msg.position.y"].iloc[0],
                color=colors(i),
                marker="o",
                s=100,
                edgecolor="black",
                zorder=5,
                label=f"Branch {i+1} Start")

# 模式1：虚线
for i, seg in enumerate(segments_mode1):
    if i == 1:
        plt.plot(seg["vrpn_pose_msg.position.x"],
                seg["vrpn_pose_msg.position.y"],
                label=f"Manual Control",
                color="slategray",
                linestyle="--",
                alpha=0.8)

plt.xlabel("X position [m]")
plt.ylabel("Y position [m]")
plt.title("FPV XY Trajectory under SGVF for Two Separated Circles")
plt.legend()
plt.axis("equal")
plt.grid(True)
plt.tight_layout()

save_dir = os.path.join("experiment_data")
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, taskname + "traj.pdf"), format='pdf', dpi=300)

plt.show()


# 配色
colors = cm.get_cmap("Set1")

# 创建上下两个子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

# 准备横轴刻度
xticks = []
xticklabels = []
cum_time = 0.0  # 累计时间，用于绘图

# 上图：XY速度
for i, seg in enumerate(segments_mode2):
    extra_gap = 3.0 if i > 0 else 0.0
    time_rel = seg["time(s)"] - seg["time(s)"].iloc[0]  # 段内时间从0开始
    time_plot = time_rel + cum_time + extra_gap

    # 绘图
    ax1.plot(time_plot, seg["vrpn_twist_msg.linear.x"], label="vel_x" if i==0 else "", color=colors(0))
    ax1.plot(time_plot, seg["vrpn_twist_msg.linear.y"], label="vel_y" if i==0 else "", color=colors(1))

    # 段首竖线
    if i > 0:
        ax1.axvline(time_plot.iloc[0], color="gray", linestyle=":", alpha=0.7)

    # 设置刻度：段首0，段尾显示段长度
    xticks.append(cum_time + extra_gap)
    xticklabels.append("0")
    xticks.append(time_plot.iloc[-1])
    xticklabels.append(f"{time_rel.iloc[-1]:.1f}")

    # 更新累计时间
    cum_time = time_plot.iloc[-1]

ax1.tick_params(bottom=False)   # 不显示刻度线
ax1.tick_params(labelbottom=False)  # 不显示刻度文字
ax1.set_ylabel("Velocity [m/s]")
ax1.set_title("FPV XY Velocity")
ax1.legend(loc='upper right', ncol=2, fontsize=10)
ax1.set_xlim(0, cum_time)
ax1.grid(axis="y")  # 只保留 y 方向网格

# 下图：四个控制输入
cum_time = 0.0  # 重新计算累计时间
for i, seg in enumerate(segments_mode2):
    extra_gap = 3.0 if i > 0 else 0.0
    time_rel = seg["time(s)"] - seg["time(s)"].iloc[0]
    time_plot = time_rel + cum_time + extra_gap

    # 绘图
    ax2.plot(time_plot, seg["motor_ctrl0_N"], label="motor0" if i==0 else "", color=colors(0))
    ax2.plot(time_plot, seg["motor_ctrl1_N"], label="motor1" if i==0 else "", color=colors(1))
    ax2.plot(time_plot, seg["motor_ctrl2_N"], label="motor2" if i==0 else "", color=colors(2))
    ax2.plot(time_plot, seg["motor_ctrl3_N"], label="motor3" if i==0 else "", color=colors(3))

    if i > 0:
        ax2.axvline(time_plot.iloc[0], color="gray", linestyle=":", alpha=0.7)

    cum_time = time_plot.iloc[-1]

ax2.set_xlabel("Time [s] (per segment)")
ax2.set_ylabel("Motor Control [N]")
ax2.set_title("Motor Inputs")
ax2.legend(loc='upper right', ncol=4, fontsize=10)
ax2.set_xlim(0, cum_time)
ax2.grid(axis="y")  # 只保留 y 方向网格

# 设置统一的 x 轴刻度
ax2.set_xticks(xticks)
ax2.set_xticklabels(xticklabels)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, taskname + "control.pdf"), format='pdf', dpi=300)
plt.show()