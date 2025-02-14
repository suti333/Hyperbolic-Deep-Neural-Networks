from manifolds.base import Manifold
import numpy as np

class Euclidean(Manifold):
    def __init__(self, dimension):
        self._dimension = dimension

    def dimension(self):
        return self._dimension

    def project(self, point):
        return point
    
    def distance(self, point1, point2):
        return np.linalg.norm(point2 - point1)
    
    def exp_map(self, point, tangent_vector):
        return point + tangent_vector
    
    def log_map(self, point1, point2):
        return point2 - point1
    
    def exp_map0(self, tangent_vector):
        return tangent_vector
    
    def log_map0(self, point):
        return point
    
    def mobius_add(self, x, y):
        return x + y
    
    def mobius_matrix_vector_mul(self, M, x):
        return x @ M
    