import cv2
import torch
import numpy as np
from typing import Optional
import os
import matplotlib.pyplot as plt

class ImageEdgeSampleable:
    """
    Samples points along edges of a binary/gray image,
    returning a torch.Tensor of shape (num_points, 2) on a given device.
    """
    def __init__(self, device: torch.device, img_path: str = None, shuffle: bool = True):
        self.device = device
        self.shuffle = shuffle
        self.img_path = img_path
        self.samples = None

        if img_path is not None:
            self._extract_edges(img_path)

    def _extract_edges(self, img_path: str):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)
        points = np.column_stack(np.where(edges > 0))[:, ::-1]  # (x, y)

        # 翻转 y 坐标，使上方向为正
        points[:, 1] = img.shape[0] - points[:, 1]

        self.samples = torch.tensor(points.copy(), dtype=torch.float32, device=self.device)
        if self.shuffle:
            indices = torch.randperm(self.samples.shape[0], device=self.device)
            self.samples = self.samples[indices]

    @property
    def dim(self) -> int:
        return 2

    def sample(self, num_samples: Optional[int] = None) -> torch.Tensor:
        if self.samples is None:
            raise ValueError("No samples available. Initialize with an image or load from file.")
        if num_samples is None or num_samples >= self.samples.shape[0]:
            return self.samples
        else:
            indices = torch.randperm(self.samples.shape[0], device=self.device)[:num_samples]
            return self.samples[indices]

    def save_samples(self, save_dir: str, filename: str = "edge_points.npy"):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        np.save(save_path, self.samples.cpu().numpy())
        print(f"Edge points saved to {save_path}")

    def load_samples(self, file_path: str):
        np_points = np.load(file_path)
        self.samples = torch.tensor(np_points, dtype=torch.float32, device=self.device)
        if self.shuffle:
            indices = torch.randperm(self.samples.shape[0], device=self.device)
            self.samples = self.samples[indices]
        print(f"Edge points loaded from {file_path}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img_path = 'utils/westlake.jpg'
save_dir = 'utils/'

# 1️⃣ 提取边缘并保存
img_sampler = ImageEdgeSampleable(device=device, img_path=img_path, shuffle=True)
img_sampler.save_samples(save_dir=save_dir, filename='westlake_edges.npy')

# 2️⃣ 加载边缘点（保持 tensor 格式）
sampler2 = ImageEdgeSampleable(device=device, shuffle=True)
sampler2.load_samples('utils/westlake_edges.npy')
edge_points_tensor = sampler2.sample()  # torch.Tensor
print(edge_points_tensor.shape, edge_points_tensor.device)

# 显示加载后的边缘点
points = sampler2.samples.cpu().numpy()  # (N, 2)
plt.figure(figsize=(6, 6))
# plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))  # 原图
plt.scatter(points[:, 0], points[:, 1], s=1, c='red')  # 边缘点
plt.axis('equal')  # 保持 x,y 比例
plt.title("Detected Edge Points (Loaded from File)")
plt.show()