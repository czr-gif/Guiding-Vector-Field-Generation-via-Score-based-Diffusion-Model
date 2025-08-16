from abc import ABC, abstractmethod
from typing import Optional, List, Type, Tuple, Dict
from core.schedules import Alpha, Beta
import torch
import torch.distributions as D
from tqdm import tqdm
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def build_mlp(dims: List[int], activation: Type[torch.nn.Module] = torch.nn.SiLU):
        mlp = []
        for idx in range(len(dims) - 1):
            mlp.append(torch.nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                mlp.append(activation())
        return torch.nn.Sequential(*mlp)

class MLPVectorField(torch.nn.Module):
    """
    MLP-parameterization of the learned vector field u_t^theta(x)
    """
    def __init__(self, dim: int, hiddens: List[int]):
        super().__init__()
        self.dim = dim
        self.net = build_mlp([dim + 1] + hiddens + [dim])

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Args:
        - x: (bs, dim)
        Returns:
        - u_t^theta(x): (bs, dim)
        """
        
        xt = torch.cat([x,t], dim=-1)
        return self.net(xt)        
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
    
class MLPScore(torch.nn.Module):
    """
    MLP-parameterization of the learned score field
    """
    def __init__(self, dim: int, hiddens: List[int]):
        super().__init__()
        self.dim = dim
        self.net = build_mlp([dim + 1] + hiddens + [dim])

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Args:
        - x: (bs, dim)
        Returns:
        - s_t^theta(x): (bs, dim)
        """
        xt = torch.cat([x,t], dim=-1)
        return self.net(xt)        
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
    
class ScoreFromVectorField(torch.nn.Module):
    """
    Parameterization of score via learned vector field (for the special case of a Gaussian conditional probability path)
    """
    def __init__(self, vector_field: MLPVectorField, alpha: Alpha, beta: Beta):
        super().__init__()
        self.vector_field = vector_field
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Args:
        - x: (bs, dim)
        Returns:
        - score: (bs, dim)
        """
        alpha_t = self.alpha(t)               # (bs, 1)
        alpha_dt = self.alpha.dt(t)           # (bs, 1)
        beta_t = self.beta(t)                 # (bs, 1)
        beta_dt = self.beta.dt(t)             # (bs, 1)

        numerator = alpha_t * self.vector_field(x, t) - alpha_dt * x  # (bs, dim)
        denominator = beta_t ** 2 * alpha_dt - alpha_t * beta_t * beta_dt  # (bs, 1)
        denominator = torch.clamp(denominator, min=1e-8)

        score = numerator / denominator  # (bs, dim), broadcasting along dim
        return score

class TangentNet(torch.nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.hidden = hidden
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 2)  # 直接输出 (vx, vy)
        )

    def forward(self, x):
        """
        x: (N, 2) 输入位置
        输出: (N, 2) 切向向量 v_phi(x)
        """
        return self.net(x)

    def save(self, path: str):
        torch.save({
            'state_dict': self.state_dict(),
            'hidden': self.hidden
        }, path)

    @classmethod
    def load(cls, path: str, map_location=None):
        checkpoint = torch.load(path, map_location=map_location)
        model = cls(hidden=checkpoint['hidden'])
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model

    
################### Trainer ################
from core.paths import ConditionalProbabilityPath

class Trainer(ABC):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass

    def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, num_epochs: int, device: torch.device, lr: float = 1e-3, **kwargs) -> torch.Tensor:
        # Start
        self.model.to(device)
        opt = self.get_optimizer(lr)
        self.model.train()
        losses = []
        # Train loop
        pbar = tqdm(enumerate(range(num_epochs)))
        for idx, epoch in pbar:
            opt.zero_grad()
            loss = self.get_train_loss(**kwargs)
            loss.backward()
            opt.step()
            pbar.set_description(f'Epoch {idx}, loss: {loss.item()}')
            losses.append(loss.item())
        # Finish
        self.model.eval()
        return losses

class ConditionalFlowMatchingTrainer(Trainer):
    def __init__(self, path: ConditionalProbabilityPath, model: MLPVectorField, **kwargs):
        super().__init__(model, **kwargs)
        self.path = path

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        z = self.path.p_data.sample(batch_size).to('cuda')
        t = torch.rand(batch_size,1).to('cuda')
        x = self.path.sample_conditional_path(z,t).to('cuda')
        u = self.model.forward(x,t)
        ur = self.path.conditional_vector_field(x,z,t)
        loss = torch.mean(torch.sum((u-ur)**2,dim=1))
        return loss
    
class ConditionalScoreMatchingTrainer(Trainer):
    def __init__(self, path: ConditionalProbabilityPath, model: MLPScore, **kwargs):
        super().__init__(model, **kwargs)
        self.path = path

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        z = self.path.p_data.sample(batch_size).to('cuda')
        t = torch.rand(batch_size,1).to('cuda')
        x = self.path.sample_conditional_path(z,t).to('cuda')
        u = self.model.forward(x,t)
        ur = self.path.conditional_score(x,z,t)
        loss = torch.mean(torch.sum((u-ur)**2,dim=1))
        return loss

class TangentNetTrainer:
    def __init__(self, tangent_net, score_model, k=5, lr=1e-3, device='cuda'):
        self.tangent_net = tangent_net.to(device)
        self.score_model = score_model.to(device)
        self.opt = torch.optim.Adam(self.tangent_net.parameters(), lr=lr)
        self.device = device
        self.k = k

    def preprocess_score_field(self, s_theta):
        """
        按可视化时的方式缩放 score 场向量
        """
        norms = torch.norm(s_theta, dim=1, keepdim=True) + 1e-8
        sc = torch.tanh(0.2 * norms)          # 限幅
        return s_theta / norms * sc     # 缩放到合适大小

    def compute_directional_consistency_loss(self, points: torch.Tensor, vectors: torch.Tensor):
        """
        基于k近邻的方向一致性损失，鼓励邻居的向量方向相似
        points: (N, 2) 位置
        vectors: (N, 2) 混合向量（未归一化也没关系，函数内会归一化）
        """
        unit_vecs = F.normalize(vectors, dim=1)  # 归一化方向向量
        dists = torch.cdist(points, points)      # 欧式距离矩阵 (N, N)
        knn_indices = dists.topk(k=self.k + 1, largest=False).indices[:, 1:]  # 排除自己
        neighbors = unit_vecs[knn_indices]       # (N, k, 2)
        center = unit_vecs.unsqueeze(1)          # (N, 1, 2)
        cos_sim = F.cosine_similarity(center, neighbors, dim=-1)  # (N, k)
        loss = 1 - cos_sim.mean()
        return loss

    def compute_loss(self, x, t):
        # 1. 获取 score 场并做可视化时的缩放处理
        s_theta = self.score_model(x, t)         # (N, 2)
        s_theta = -self.preprocess_score_field(s_theta)

        # 2. 获取 tangent 场
        v_phi = self.tangent_net(x)              # (N, 2)

        # 3. 混合向量场
        m_field = s_theta + v_phi

        # 4. 单位长度约束
        m_norm = torch.norm(m_field, dim=1) + 1e-8
        loss_unit = torch.mean((m_norm - 1.0) ** 2)

        # 5. 方向一致性约束（基于邻居）
        loss_dir = self.compute_directional_consistency_loss(x, m_field)

        return loss_unit + loss_dir, loss_unit.item(), loss_dir.item()

    def train_step(self, x):
        # 固定 t = 1
        t = torch.ones(x.size(0), 1, device=self.device)
        self.opt.zero_grad()
        loss, lu, ld = self.compute_loss(x, t)
        loss.backward()
        self.opt.step()
        return loss.item(), lu, ld




# class TangentNetTrainer(Trainer):
#     """
#     训练 TangentNet 的 Trainer，通过引导其学习局部向量场中的方向一致性结构。
#     """
#     def __init__(
#         self,
#         score_model: MLPScore,
#         tangent_model: TangentNet,
#         k: int = 5,
#         scale: float = 3.0,
#         lambda_unit: float = 1.0,
#         lambda_orth: float = 1.0,
#         lambda_dir: float = 1.0,
#         **kwargs
#     ):
#         """
#         参数说明：
#         - score_model: 预训练的score网络（固定不动）
#         - tangent_model: 待训练的 TangentNet 网络
#         - k: 方向一致性中最近邻数量
#         - scale: 用于归一化混合向量的可视化尺度
#         - lambda_*: 损失权重
#         """
#         super().__init__(tangent_model, **kwargs)
#         self.score_model = score_model.eval()  # 固定 score 网络
#         self.k = k
#         self.scale = scale
#         self.lambda_unit = lambda_unit
#         self.lambda_orth = lambda_orth
#         self.lambda_dir = lambda_dir

#     @torch.no_grad()
#     def compute_mixed_vectors(self, points: torch.Tensor, t_tensor: torch.Tensor):
#         """
#         根据 score 模型与 Tangent_net 生成混合向量。
#         返回：
#         - mixed_vecs: (N, 2)
#         - s_vecs: (N, 2)
#         - v_vecs: (N, 2)
#         """
#         s_vecs = self.score_model(points, t_tensor)  # score 场
#         v_vecs = self.model(points, t_tensor)        # tangent 场

#         # 混合向量： m = s + v
#         mixed_vecs = s_vecs + v_vecs
#         return mixed_vecs, s_vecs, v_vecs

#     def get_train_loss(self, batch_size: int) -> torch.Tensor:
#         device = next(self.model.parameters()).device

#         # 采样一批点
#         z = self.path.p_data.sample(batch_size).to(device)
#         t = torch.rand(batch_size, 1).to(device)
#         x = self.path.sample_conditional_path(z, t).to(device)

#         # 计算 m, s, v
#         mixed_vecs, s_vecs, v_vecs = self.compute_mixed_vectors(x, t)

#         # ---- L_unit ----
#         m_norm = torch.norm(mixed_vecs, dim=1)
#         L_unit = ((m_norm - 1) ** 2).mean()

#         # ---- L_orth ----
#         s_unit = F.normalize(s_vecs, dim=1)
#         v_unit = F.normalize(v_vecs, dim=1)
#         L_orth = ( (s_unit * v_unit).sum(dim=1) ** 2 ).mean()

#         # ---- L_dir ----
#         L_dir = self.compute_directional_consistency_loss(x, mixed_vecs, k=self.k)

#         # 总损失
#         L_total = (
#             self.lambda_unit * L_unit +
#             self.lambda_orth * L_orth +
#             self.lambda_dir * L_dir
#         )

#         return L_total

#     def compute_directional_consistency_loss(self, points: torch.Tensor, mixed_vectors: torch.Tensor, k: int = 5):
#         """
#         方向一致性损失函数，鼓励相邻点的混合向量方向保持一致。
#         """
#         unit_vecs = F.normalize(mixed_vectors, dim=1)
#         dists = torch.cdist(points, points)  # (N, N)
#         knn_indices = dists.topk(k=k + 1, largest=False).indices[:, 1:]  # 排除自己
#         neighbors = unit_vecs[knn_indices]  # (N, k, 2)
#         center = unit_vecs.unsqueeze(1)     # (N, 1, 2)
#         cos_sim = F.cosine_similarity(center, neighbors, dim=-1)  # (N, k)
#         return 1 - cos_sim.mean()


