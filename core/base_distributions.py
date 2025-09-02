from abc import ABC, abstractmethod
from typing import Optional, List, Type, Tuple, Dict
import numpy as np
import torch
import torch.distributions as D
from sklearn.datasets import make_moons, make_circles
from utils.device import get_device

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = get_device()

class Sampleable(ABC):
    """
    Distribution which can be sampled from
    """
    @property
    @abstractmethod
    def dim(self) -> int:
        """
        Returns:
            - Dimensionality of the distribution
        """
        pass
        
    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, dim)
        """
        pass

class Density(ABC):
    """
    Distribution with tractable density
    """
    @abstractmethod
    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the log density at x.
        Args:
            - x: shape (batch_size, dim)
        Returns:
            - log_density: shape (batch_size, 1)
        """
        pass

class UniformBox(torch.nn.Module, Sampleable, Density):
    """
    Uniform distribution on a d-dimensional hypercube [low, high]^d
    """
    def __init__(self, low: torch.Tensor, high: torch.Tensor):
        """
        low: shape (dim,)  — lower bounds for each dimension
        high: shape (dim,) — upper bounds for each dimension
        """
        super().__init__()
        assert low.shape == high.shape
        self.register_buffer("low", low)
        self.register_buffer("high", high)

    @property
    def dim(self) -> int:
        return self.low.shape[0]

    @property
    def distribution(self):
        return D.Uniform(self.low, self.high)

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.distribution.sample((num_samples,))  # (num_samples, dim)

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns log p(x), which is constant inside [low, high]^d and -inf outside
        """
        in_bounds = ((x >= self.low) & (x <= self.high)).all(dim=-1, keepdim=True)
        volume = torch.prod(self.high - self.low)
        log_prob = torch.full_like(in_bounds, fill_value=-float('inf'), dtype=torch.float)
        log_prob[in_bounds] = -torch.log(volume)
        return log_prob

    @classmethod
    def box(cls, dim: int, bound: float) -> "UniformBox":
        """
        Create uniform distribution over [-bound, bound]^d
        """
        print('dim:',type(dim), dim)
        low = torch.full((dim,), -bound)
        high = torch.full((dim,), bound)
        return cls(low, high)
    
class Gaussian(torch.nn.Module, Sampleable, Density):
    """
    Multivariate Gaussian distribution
    """
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor):
        """
        mean: shape (dim,)
        cov: shape (dim,dim)
        """
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("cov", cov)

    @property
    def dim(self) -> int:
        return self.mean.shape[0]

    @property
    def distribution(self):
        return D.MultivariateNormal(self.mean, self.cov, validate_args=False)

    def sample(self, num_samples) -> torch.Tensor:
        return self.distribution.sample((num_samples,))
        
    def log_density(self, x: torch.Tensor):
        return self.distribution.log_prob(x).view(-1, 1)

    @classmethod
    def isotropic(cls, dim: int, std: float) -> "Gaussian":
        mean = torch.zeros(dim)
        cov = torch.eye(dim) * std ** 2
        return cls(mean, cov)

class GaussianMixture(torch.nn.Module, Sampleable, Density):
    """
    Two-dimensional Gaussian mixture model, and is a Density and a Sampleable. Wrapper around torch.distributions.MixtureSameFamily.
    """
    def __init__(
        self,
        means: torch.Tensor,  # nmodes x data_dim
        covs: torch.Tensor,  # nmodes x data_dim x data_dim
        weights: torch.Tensor,  # nmodes
    ):
        """
        means: shape (nmodes, 2)
        covs: shape (nmodes, 2, 2)
        weights: shape (nmodes, 1)
        """
        super().__init__()
        self.nmodes = means.shape[0]
        self.register_buffer("means", means)
        self.register_buffer("covs", covs)
        self.register_buffer("weights", weights)

    @property
    def dim(self) -> int:
        return self.means.shape[1]

    @property
    def distribution(self):
        return D.MixtureSameFamily(
                mixture_distribution=D.Categorical(probs=self.weights, validate_args=False),
                component_distribution=D.MultivariateNormal(
                    loc=self.means,
                    covariance_matrix=self.covs,
                    validate_args=False,
                ),
                validate_args=False,
            )

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(x).view(-1, 1)

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.distribution.sample(torch.Size((num_samples,)))

    @classmethod
    def random_2D(
        cls, nmodes: int, std: float, scale: float = 10.0, x_offset: float = 0.0, seed = 0.0
    ) -> "GaussianMixture":
        torch.manual_seed(seed)
        means = (torch.rand(nmodes, 2) - 0.5) * scale + x_offset * torch.Tensor([1.0, 0.0])
        covs = torch.diag_embed(torch.ones(nmodes, 2)) * std ** 2
        weights = torch.ones(nmodes)
        return cls(means, covs, weights)

    @classmethod
    def symmetric_2D(
        cls, nmodes: int, std: float, scale: float = 10.0, x_offset: float = 0.0
    ) -> "GaussianMixture":
        angles = torch.linspace(0, 2 * np.pi, nmodes + 1)[:nmodes]
        means = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * scale + torch.Tensor([1.0, 0.0]) * x_offset
        covs = torch.diag_embed(torch.ones(nmodes, 2) * std ** 2)
        weights = torch.ones(nmodes) / nmodes
        return cls(means, covs, weights)
    
class MoonsSampleable(Sampleable):
    """
    Implementation of the Moons distribution using sklearn's make_moons
    """
    def __init__(self, device: torch.device, noise: float = 0.05, scale: float = 5.0, offset: Optional[torch.Tensor] = None):
        """
        Args:
            noise: Standard deviation of Gaussian noise added to the data
            scale: How much to scale the data
            offset: How much to shift the samples from the original distribution (2,)
        """
        self.noise = noise
        self.scale = scale
        self.device = device
        if offset is None:
            offset = torch.zeros(2)
        self.offset = offset.to(device)

    @property
    def dim(self) -> int:
        return 2

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Args:
            num_samples: Number of samples to generate
        Returns:
            torch.Tensor: Generated samples with shape (num_samples, 3)
        """
        samples, _ = make_moons(
            n_samples=num_samples,
            noise=self.noise,
            random_state=None  # Allow for random generation each time
        )
        return self.scale * torch.from_numpy(samples.astype(np.float32)).to(self.device) + self.offset

class CirclesSampleable(Sampleable):
    """
    Implementation of concentric circle distribution using sklearn's make_circles
    """
    def __init__(self, device: torch.device, noise: float = 0.05, scale=4.0, offset: Optional[torch.Tensor] = None):
        """
        Args:
            noise: standard deviation of Gaussian noise added to the data
        """
        self.noise = noise
        self.scale = scale
        self.device = device
        if offset is None:
            offset = torch.zeros(2)
        self.offset = offset.to(device)

    @property
    def dim(self) -> int:
        return 2

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Args:
            num_samples: number of samples to generate
        Returns:
            torch.Tensor: shape (num_samples, 3)
        """
        samples, _ = make_circles(
            n_samples=num_samples,
            noise=self.noise,
            factor=0.5,
            random_state=None
        )
        return self.scale * torch.from_numpy(samples.astype(np.float32)).to(self.device) + self.offset
    def log_density(self, x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
        """
        Smooth log-density of two concentric circles using Gaussian radial kernel.
        """
        r1 = self.scale * 0.5
        r2 = self.scale * 1.0
        dist = torch.norm(x - self.offset, dim=1)
        logp1 = -((dist - r1)**2) / (2*sigma**2)
        logp2 = -((dist - r2)**2) / (2*sigma**2)
        # logsumexp 保持连续性
        logp = torch.log(torch.exp(logp1) + torch.exp(logp2))
        return logp.unsqueeze(-1)  # 保持 (N,1) 形状
    
class SingleCircleSampleable(Sampleable):
    """
    Uniformly samples points on a single circle with a given radius.
    """
    def __init__(self, device: torch.device, radius: float = 1.0, offset: Optional[torch.Tensor] = None, shuffle: bool = True):
        """
        Args:
            device: computation device (e.g., cpu or cuda)
            radius: radius of the circle
            offset: center of the circle (2D vector)
            shuffle: whether to shuffle the sampled points
        """
        self.radius = radius
        self.device = device
        self.shuffle = shuffle
        if offset is None:
            offset = torch.zeros(2)
        self.offset = offset.to(device)

    @property
    def dim(self) -> int:
        return 2

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Uniformly sample points on a circle and optionally shuffle them.
        """
        # Avoid duplicated start and end point
        theta = torch.linspace(0, 2 * torch.pi * (1 - 1 / num_samples), num_samples).to(self.device)
        x = self.radius * torch.cos(theta)
        y = self.radius * torch.sin(theta)
        samples = torch.stack([x, y], dim=1) + self.offset  # shape: (num_samples, 2)

        if self.shuffle:
            indices = torch.randperm(num_samples, device=self.device)
            samples = samples[indices]

        return samples

class CheckerboardSampleable(Sampleable):
    """
    Checkboard-esque distribution
    """
    def __init__(self, device: torch.device, grid_size: int = 3, scale=5.0):
        """
        Args:
            noise: standard deviation of Gaussian noise added to the data
        """
        self.grid_size = grid_size
        self.scale = scale
        self.device = device

    @property
    def dim(self) -> int:
        return 2

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Args:
            num_samples: number of samples to generate
        Returns:
            torch.Tensor: shape (num_samples, 3)
        """
        grid_length = 2 * self.scale / self.grid_size
        samples = torch.zeros(0,2).to(device)
        while samples.shape[0] < num_samples:
            # Sample num_samples
            new_samples = (torch.rand(num_samples,2).to(self.device) - 0.5) * 2 * self.scale
            x_mask = torch.floor((new_samples[:,0] + self.scale) / grid_length) % 2 == 0 # (bs,)
            y_mask = torch.floor((new_samples[:,1] + self.scale) / grid_length) % 2 == 0 # (bs,)
            accept_mask = torch.logical_xor(~x_mask, y_mask)
            samples = torch.cat([samples, new_samples[accept_mask]], dim=0)
        return samples[:num_samples]
    
class InputdataSampleable(Sampleable):
    """
    Sampleable class for user-specified points (e.g., image edge points),
    linearly scaled to a square [-bound, bound]^2.
    """
    def __init__(self, device: torch.device, points: torch.Tensor, shuffle: bool = True, bound: float = 4.0):
        self.device = device
        self.shuffle = shuffle
        self.bound = bound

        # 转到设备并缩放到 [-bound, bound]^2
        points = points.to(device)
        points = self.scale_points_to_bound(points, self.bound)
        self.samples = points

    @staticmethod
    def scale_points_to_bound(points: torch.Tensor, bound: float) -> torch.Tensor:
        """
        Linearly scale and translate points to fit inside [-bound, bound]^2
        """
        min_vals = points.min(dim=0, keepdim=True)[0]
        max_vals = points.max(dim=0, keepdim=True)[0]

        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1.0  # 防止除以零

        scaled = (points - min_vals) / ranges * (2 * bound) - bound
        return scaled

    @property
    def dim(self) -> int:
        return self.samples.shape[1]

    def sample(self, num_samples: int) -> torch.Tensor:
        N = self.samples.shape[0]
        if num_samples >= N:
            sampled = self.samples.clone()
        else:
            indices = torch.randperm(N, device=self.device)[:num_samples]
            sampled = self.samples[indices]

        if self.shuffle:
            indices = torch.randperm(sampled.shape[0], device=self.device)
            sampled = sampled[indices]

        return sampled

class TwoPolygonsSampleable(Sampleable):
    """
    Uniform sampling along the edges of two polygons in 2D, with optional Gaussian noise.
    """
    def __init__(self, device: torch.device, shuffle: bool = True, noise_std: float = 0.05):
        self.device = device
        self.shuffle = shuffle
        self.noise_std = noise_std
        # 多边形顶点（首尾相接）
        self.poly1 = torch.tensor([[-4., 4.], [-1., 4.], [-1., 3.], [-3., 3.], [-3., -4.], [-4., -4.]], device=device)
        self.poly2 = torch.tensor([[4., 4.], [3., 4.], [3., -3.], [1., -3.], [1., -4.], [4., -4.]], device=device)

    @property
    def dim(self) -> int:
        return 2

    def sample_polygon_edges(self, poly: torch.Tensor, points_per_edge: int) -> torch.Tensor:
        """
        Sample points uniformly along the edges of a polygon.
        poly: (N, 2), closed polygon (首尾相接)
        """
        n = poly.shape[0]
        samples_list = []
        for i in range(n):
            p0, p1 = poly[i], poly[(i+1) % n]  # wrap around
            t = torch.linspace(0, 1, points_per_edge, device=self.device).unsqueeze(1)
            edge_samples = (1 - t) * p0 + t * p1
            samples_list.append(edge_samples)
        return torch.cat(samples_list, dim=0)

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Sample a total of `num_samples` points along all edges of the two polygons,
        with optional Gaussian noise.
        """
        # 每个多边形的边数
        num_edges_total = self.poly1.shape[0] + self.poly2.shape[0]
        points_per_edge = max(1, num_samples // num_edges_total)

        s1 = self.sample_polygon_edges(self.poly1, points_per_edge)
        s2 = self.sample_polygon_edges(self.poly2, points_per_edge)
        samples = torch.cat([s1, s2], dim=0)

        # 调整数量
        if samples.shape[0] > num_samples:
            indices = torch.randperm(samples.shape[0], device=self.device)[:num_samples]
            samples = samples[indices]
        elif samples.shape[0] < num_samples:
            repeat = (num_samples + samples.shape[0] - 1) // samples.shape[0]
            samples = samples.repeat(repeat, 1)[:num_samples]

        # 打乱
        if self.shuffle:
            indices = torch.randperm(samples.shape[0], device=self.device)
            samples = samples[indices]

        # 添加噪声
        if self.noise_std > 0:
            noise = torch.randn_like(samples) * self.noise_std
            samples = samples + noise

        return samples


class TwoCirclesSampleable(Sampleable):
    """
    Uniform sampling along the circumferences of two circles in 2D, with optional Gaussian noise.
    """
    def __init__(self, device: torch.device, shuffle: bool = True, noise_std: float = 0.05):
        self.device = device
        self.shuffle = shuffle
        self.noise_std = noise_std
        # 两个圆的圆心和半径
        self.centers = torch.tensor([[-1.0, 1.0], [1.0, -1.0]], device=device)
        self.radius = 1

    @property
    def dim(self) -> int:
        return 2

    def sample_circle(self, center: torch.Tensor, num_points: int) -> torch.Tensor:
        """
        Sample points uniformly along the circumference of a circle.
        """
        theta = torch.linspace(0, 2 * torch.pi, num_points + 1, device=self.device)[:-1]
        x = torch.cos(theta) * self.radius + center[0]
        y = torch.sin(theta) * self.radius + center[1]
        return torch.stack([x, y], dim=1)

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Sample a total of `num_samples` points along the circumferences of two circles.
        """
        # 每个圆分配的点数
        points_per_circle = max(1, num_samples // 2)

        s1 = self.sample_circle(self.centers[0], points_per_circle)
        s2 = self.sample_circle(self.centers[1], points_per_circle)
        samples = torch.cat([s1, s2], dim=0)

        # 调整数量
        if samples.shape[0] > num_samples:
            indices = torch.randperm(samples.shape[0], device=self.device)[:num_samples]
            samples = samples[indices]
        elif samples.shape[0] < num_samples:
            repeat = (num_samples + samples.shape[0] - 1) // samples.shape[0]
            samples = samples.repeat(repeat, 1)[:num_samples]

        # 打乱
        if self.shuffle:
            indices = torch.randperm(samples.shape[0], device=self.device)
            samples = samples[indices]

        # 添加噪声
        if self.noise_std > 0:
            noise = torch.randn_like(samples) * self.noise_std
            samples = samples + noise

        return samples
    
class SquareSampleable(Sampleable):
    """
    Uniform sampling along the edges of a square in 2D, without interior points.
    """
    def __init__(self, device: torch.device, shuffle: bool = True, noise_std: float = 0.0):
        self.device = device
        self.shuffle = shuffle
        self.noise_std = noise_std
        # 定义正方形顶点，顺时针或逆时针，首尾相接
        edge = 3.
        self.square = torch.tensor([
            [-edge, -edge],
            [-edge, edge],
            [edge, edge],
            [edge, -edge]
        ], device=device)

    @property
    def dim(self) -> int:
        return 2

    def sample_square_edges(self, points_per_edge: int) -> torch.Tensor:
        n = self.square.shape[0]
        samples_list = []
        for i in range(n):
            p0, p1 = self.square[i], self.square[(i+1) % n]
            t = torch.linspace(0, 1, points_per_edge, device=self.device).unsqueeze(1)
            edge_samples = (1 - t) * p0 + t * p1
            samples_list.append(edge_samples)
        return torch.cat(samples_list, dim=0)

    def sample(self, num_samples: int) -> torch.Tensor:
        points_per_edge = max(1, num_samples // 4)
        samples = self.sample_square_edges(points_per_edge)

        # 如果样本数量不足或过多，裁剪或重复
        if samples.shape[0] > num_samples:
            indices = torch.randperm(samples.shape[0], device=self.device)[:num_samples]
            samples = samples[indices]
        elif samples.shape[0] < num_samples:
            repeat = (num_samples + samples.shape[0] - 1) // samples.shape[0]
            samples = samples.repeat(repeat, 1)[:num_samples]

        if self.shuffle:
            indices = torch.randperm(samples.shape[0], device=self.device)
            samples = samples[indices]

        if self.noise_std > 0:
            noise = torch.randn_like(samples) * self.noise_std
            samples = samples + noise

        return samples
    
class HexagonSampleable(Sampleable):
    """
    Uniform sampling along the edges of a regular hexagon in 2D, without interior points.
    """
    def __init__(self, device: torch.device, shuffle: bool = True, noise_std: float = 0.0, radius: float = 3.0):
        """
        Args:
            device: torch device
            shuffle: whether to shuffle the sampled points
            noise_std: Gaussian noise standard deviation
            radius: distance from hexagon center to vertices
        """
        self.device = device
        self.shuffle = shuffle
        self.noise_std = noise_std
        self.radius = radius

        # 生成正六边形顶点，逆时针顺序
        n_vertices = 6
        angles = torch.arange(n_vertices, device=device) * (2 * torch.pi / n_vertices)

        self.hexagon = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1)

    @property
    def dim(self) -> int:
        return 2

    def sample_hexagon_edges(self, points_per_edge: int) -> torch.Tensor:
        n = self.hexagon.shape[0]
        samples_list = []
        for i in range(n):
            p0, p1 = self.hexagon[i], self.hexagon[(i + 1) % n]
            t = torch.linspace(0, 1, points_per_edge, device=self.device).unsqueeze(1)
            edge_samples = (1 - t) * p0 + t * p1
            samples_list.append(edge_samples)
        return torch.cat(samples_list, dim=0)

    def sample(self, num_samples: int) -> torch.Tensor:
        points_per_edge = max(1, num_samples // 6)
        samples = self.sample_hexagon_edges(points_per_edge)

        # 如果样本数量不足或过多，裁剪或重复
        if samples.shape[0] > num_samples:
            indices = torch.randperm(samples.shape[0], device=self.device)[:num_samples]
            samples = samples[indices]
        elif samples.shape[0] < num_samples:
            repeat = (num_samples + samples.shape[0] - 1) // samples.shape[0]
            samples = samples.repeat(repeat, 1)[:num_samples]

        if self.shuffle:
            indices = torch.randperm(samples.shape[0], device=self.device)
            samples = samples[indices]

        if self.noise_std > 0:
            noise = torch.randn_like(samples) * self.noise_std
            samples = samples + noise

        return samples
    
class OctagonSampleable(Sampleable):
    """
    Uniform sampling along the edges of a regular octagon in 2D, without interior points.
    """
    def __init__(self, device: torch.device, shuffle: bool = True, noise_std: float = 0.0, radius: float = 3.0):
        """
        Args:
            device: torch device
            shuffle: whether to shuffle the sampled points
            noise_std: Gaussian noise standard deviation
            radius: distance from octagon center to vertices
        """
        self.device = device
        self.shuffle = shuffle
        self.noise_std = noise_std
        self.radius = radius

        # 生成正八边形顶点，逆时针顺序
        n_vertices = 8
        angles = torch.arange(n_vertices, device=device) * (2 * torch.pi / n_vertices)
        self.octagon = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1)

    @property
    def dim(self) -> int:
        return 2

    def sample_octagon_edges(self, points_per_edge: int) -> torch.Tensor:
        n = self.octagon.shape[0]
        samples_list = []
        for i in range(n):
            p0, p1 = self.octagon[i], self.octagon[(i + 1) % n]
            t = torch.linspace(0, 1, points_per_edge, device=self.device).unsqueeze(1)
            edge_samples = (1 - t) * p0 + t * p1
            samples_list.append(edge_samples)
        return torch.cat(samples_list, dim=0)

    def sample(self, num_samples: int) -> torch.Tensor:
        points_per_edge = max(1, num_samples // 8)
        samples = self.sample_octagon_edges(points_per_edge)

        # 样本数量不足或过多，裁剪或重复
        if samples.shape[0] > num_samples:
            indices = torch.randperm(samples.shape[0], device=self.device)[:num_samples]
            samples = samples[indices]
        elif samples.shape[0] < num_samples:
            repeat = (num_samples + samples.shape[0] - 1) // samples.shape[0]
            samples = samples.repeat(repeat, 1)[:num_samples]

        if self.shuffle:
            indices = torch.randperm(samples.shape[0], device=self.device)
            samples = samples[indices]

        if self.noise_std > 0:
            noise = torch.randn_like(samples) * self.noise_std
            samples = samples + noise

        return samples



class Star5Sampleable(Sampleable):
    """
    Uniform sampling along the edges of a 5-pointed star in 2D, without interior points.
    """
    def __init__(self, device: torch.device, shuffle: bool = True, noise_std: float = 0.0):
        self.device = device
        self.shuffle = shuffle
        self.noise_std = noise_std
        
        # 五角星顶点，逆时针顺序连接形成闭合边
        R = 4.0  # 外接圆半径
        r = R * 0.382  # 内接圆半径（黄金比例）
        angles_outer = torch.linspace(0, 2 * torch.pi, 6)[:-1]  # 5个外顶点角度
        angles_inner = angles_outer + torch.pi / 5  # 5个内顶点角度

        outer_pts = torch.stack([R * torch.cos(angles_outer), R * torch.sin(angles_outer)], dim=1)
        inner_pts = torch.stack([r * torch.cos(angles_inner), r * torch.sin(angles_inner)], dim=1)

        # 交替排列外顶点和内顶点
        pts = torch.empty((10, 2), device=device)
        pts[0::2] = outer_pts
        pts[1::2] = inner_pts
        self.star = pts

    @property
    def dim(self) -> int:
        return 2

    def sample_star_edges(self, points_per_edge: int) -> torch.Tensor:
        n = self.star.shape[0]
        samples_list = []
        for i in range(n):
            p0, p1 = self.star[i], self.star[(i+1) % n]  # wrap-around
            t = torch.linspace(0, 1, points_per_edge, device=self.device).unsqueeze(1)
            edge_samples = (1 - t) * p0 + t * p1
            samples_list.append(edge_samples)
        return torch.cat(samples_list, dim=0)

    def sample(self, num_samples: int) -> torch.Tensor:
        points_per_edge = max(1, num_samples // 10)  # 10条边
        samples = self.sample_star_edges(points_per_edge)

        # 调整样本数量
        if samples.shape[0] > num_samples:
            indices = torch.randperm(samples.shape[0], device=self.device)[:num_samples]
            samples = samples[indices]
        elif samples.shape[0] < num_samples:
            repeat = (num_samples + samples.shape[0] - 1) // samples.shape[0]
            samples = samples.repeat(repeat, 1)[:num_samples]

        if self.shuffle:
            indices = torch.randperm(samples.shape[0], device=self.device)
            samples = samples[indices]

        if self.noise_std > 0:
            noise = torch.randn_like(samples) * self.noise_std
            samples = samples + noise

        return samples
