from manifolds.base import Manifold
from utils import tanh, artanh
import numpy as np
import torch

class PoincareBall(Manifold):
    def __init__(self, dimension, radius = 1.0):
        self._dimension = dimension
        self._radius = radius
        self.min_norm = 1e-6
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}

    def dimension(self):
        return self._dimension
    
    def _curvature(self):
        return 1.0 / np.power(self._radius, 2)
    
    def _lambda(self, x):
        c = self._curvature()
        x_norm_squared = torch.sum(torch.pow(x, 2), axis=-1, keepdims=True)
        return 2.0 / (1.0 - c * x_norm_squared).clamp_min(self.min_norm)
    
    def project(self, point):
        c = self._curvature()
        x_norm = point.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        maxnorm = (1 - self.eps[point.dtype]) / (c ** 0.5)
        cond = x_norm > maxnorm
        projected = point / x_norm * maxnorm
        return torch.where(cond, projected, point)

    def distance(self, point1, point2):
        return super().distance(point1, point2)
    
    def exp_map(self, point, tangent_vector):
        c = self._curvature()
        v_norm = tangent_vector.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        addend = (tanh(c ** 0.5 * self._lambda(point) * v_norm / 2)) * tangent_vector / (c ** 0.5 * v_norm)
        return self.mobius_add(point, addend)
    
    def log_map(self, point1, point2):
        c = self._curvature()
        sub = self.mobius_add(-point1, point2)
        sub_norm = sub.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        return 2 / c ** 0.5 / self._lambda(point1) * artanh(c ** 0.5 * sub_norm) * sub / sub_norm
    
    def exp_map0(self, tangent_vector):
        c = self._curvature()
        v_norm = tangent_vector.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        return tanh(c ** 0.5 * v_norm ) * tangent_vector / (c ** 0.5 * v_norm)
    
    def log_map0(self, point):
        c = self._curvature()
        p_norm = point.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        return 1 / c ** 0.5 * artanh(c ** 0.5 * p_norm) * point / p_norm
    
    def mobius_add(self, x, y):
        c = self._curvature()
        x_norm_squared = torch.sum(torch.pow(x, 2), axis=-1, keepdims=True)
        y_norm_squared = torch.sum(torch.pow(y, 2), axis=-1, keepdims=True)
        xy = torch.sum(x * y, axis=-1, keepdims=True)

        num = (1 + 2 * c * xy + c * y_norm_squared) * x + (1 - c * x_norm_squared) * y
        denom = 1 + 2 * c * xy + c ** 2 * x_norm_squared * y_norm_squared
        return num / denom.clamp_min(self.min_norm)
    
    def mobius_matrix_vector_mul(self, M, x):
        c = self._curvature()
        Mx = x @ M
        x_norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        Mx_norm = Mx.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        res = tanh(Mx_norm / x_norm * artanh(c ** 0.5 * x_norm)) * Mx / (c ** 0.5 * Mx_norm)

        condition_mask = (Mx == 0).prod(-1, keepdim=True).to(torch.bool)
        res_0 = torch.zeros(1, dtype=res.dtype, device=res.device)
        res = torch.where(condition_mask, res_0, res)
        return res
    
    def parallel_transport(self, x, y, vector):
        return super().parallel_transport(x, y, vector)
    
    def parallel_transport0(self, x, vector):
        return 2 * vector / self._lambda(x).clamp_min(self.min_norm)
    
