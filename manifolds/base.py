class Manifold:
    """Base class for implementing operations on a manifold."""

    def dimension(self):
        """Returns the dimension of the manifold."""
        raise NotImplementedError
    
    def project(self, point):
        """Projects a point onto the manifold."""
        raise NotImplementedError
    
    def distance(self, point1, point2):
        """Computes the distance between two points on the manifold."""
        raise NotImplementedError
    
    def exp_map(self, point, tangent_vector):
        """Computes the exponential map of a tangent vector at a given point."""
        raise NotImplementedError
    
    def log_map(self, point1, point2):
        """Computes the logarithmic map of point1 at point2."""
        raise NotImplementedError
    
    def exp_map0(self, tangent_vector):
        """Computes the exponential map of a tangent vector at the origin."""
        raise NotImplementedError
    
    def log_map0(self, point):
        """Computes the logarithmic map of a given point at the origin."""
        raise NotImplementedError
    
    def mobius_add(self, x, y):
        """Mobius addition."""
        raise NotImplementedError
    
    def mobius_matrix_vector_mul(self, M, x):
        """Mobius matrix-vector multiplication"""
        raise NotImplementedError
    
    def parallel_transport(self, x, y, vector):
        """Computes the parallel transport of a given vector from tangent space of x to that of y."""
        raise NotImplementedError
    
    def parallel_transport0(self, x, vector):
        """Computes the parallel transport of a given vector from tangent space of origin to that of x. """
        raise NotImplementedError
