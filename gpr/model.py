import sys

from cached_property import cached_property

import numpy as np
from numpy.linalg import *
from sklearn.gaussian_process.kernels import *
from scipy.integrate import *

from autologging import traced

from joblib import Parallel, delayed

import logging

from gpr.utils import *

class GPR:
    def __init__(self, X, Y, kernel, tol=1e-6, quad_samples=10, epsilon=1e-16):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.X = X.copy()
        self.Y = np.apply_along_axis(lambda t: np.array([np.cos(t), np.sin(t)]).T, 0, Y)
        self.logger.info("self.Y.shape: %r", self.Y.shape)
        self.kernel = kernel
        self.tol = tol
        self.quad_samples = quad_samples
        self.epsilon = epsilon
        self._K = self.kernel(self.X)
        self.K = diagonally_partitioned_matrix(self._K, self._K)

        self.train()

    def h(self, k):
        return I_1(k) / (I_0(k) * k)

    def h_prime(self, k):
        frac = I_1(k) / I_0(k)
        return 1 - (1 / k + 1 / k ** 2) * frac - frac ** 2

    def w_diagonal(self, f=None):
        if f is None:
            if not hasattr(self, "f_hat"):
                raise AttributeError("f_hat has not been calculated")
            f = self.f_hat
        
        norm_f = norm(f, axis=1)
        return self.flatten(np.apply_along_axis(self.h_prime, 0, norm_f)[:, np.newaxis] * f + np.apply_along_axis(self.h, 0, norm_f)[:, np.newaxis])

    def w(self, f=None):
        return np.diag(self.w_diagonal(f))

    def w_inversed(self, f=None):
        return np.diag(1. / (self.w_diagonal(f) + self.epsilon))

    def flatten(self, M):
        return M.flatten(order="F")

    def grad_log_P_Y(self, f):
        return - self.flatten(np.apply_along_axis(self.h, -1, norm(f, axis=1))[:, np.newaxis] * f + self.Y)

    @cached_property
    def grad_log_P_Y_f_hat(self):
        return self.grad_log_P_Y(self.f_hat)

    def k_star(self, x_star):
        k_vec = self.kernel(self.X, np.expand_dims(x_star, axis=0))
        return diagonally_partitioned_matrix(k_vec, k_vec)

    def k_double_star(self, x_star):
        k_dstar = self.kernel(np.expand_dims(x_star, axis=0))
        return np.array([[k_dstar, 0], [0, k_dstar]])

    def init_f_t(self):
        return np.random.rand(*self.X.shape) * 2 - 1

    def norm_squared(self, x):
        return np.inner(x, x)
        
    @traced
    def learn_latent_posterior(self):
        f = self.init_f_t()
        f_new = None

        self.logger.info("Learning Latent Posterior")

        while True:
            inv_w = self.w_inversed(f)
            grad = self.grad_log_P_Y(f)

            f_new = self.get_inv_K_w_inv(f, self._K) @ (self.flatten(f) + inv_w @ grad)
            f_new = f_new.reshape(self.X.shape)
            
            error = norm(self.flatten(f_new - f))
            self.logger.debug("norm(f_new - f): %f", error)
            if error <= self.tol:
                break

            assert not np.any(np.isnan(f_new))
            f = f_new

        # f_new becomes f_hat when it converges
        return f_new

    @cached_property
    def A(self):
        if not hasattr(self, "f_hat"):
            raise AttributeError("f_hat has not yet been calculated")

        return self.w + inv(self.K)

    def get_inv_K_w_inv(self, f=None, Mat_mul_left = None):
        inv_w_diagonal = 1. / (self.w_diagonal(f) + self.epsilon)
        dim = len(self._K)
        inv_w_1 = np.diag(inv_w_diagonal[:dim])
        inv_w_2 = np.diag(inv_w_diagonal[dim:])
        M11 = inv(self._K + inv_w_1)
        M22 = inv(self._K + inv_w_2)

        if Mat_mul_left is not None:
            M11 = Mat_mul_left @ M11
            M22 = Mat_mul_left @ M22

        return diagonally_partitioned_matrix(M11, M22)

    @cached_property
    def inv_K_w_inv(self):
        return self.get_inv_K_w_inv()

    @cached_property    
    def A_inversed(self):
        return inv(self.A)

    def mu_star(self, x_star):
        r"""
        :math:`\mu^\ast = (\mathbf{k^\ast})^T \grad \log \Prob{Y}{\hat{f}}`
        """
        return self.k_star(x_star).T @ self.grad_log_P_Y_f_hat

    def mu_stars(self, k_stars):
        return np.transpose(k_stars, (0, 2, 1)) @ self.grad_log_P_Y_f_hat

    def Sigma_stars(self, k_stars, k_dstars):
        def f(k_star):
            k_star = np.array(k_star).reshape(-1, 2)
            return k_star.T @ self.inv_K_w_inv @ k_star
        return k_dstars - np.vectorize(f, signature="(m,n)->(n,n)")(k_stars)

    def Sigma_star(self, x_star):
        r"""
        :math:`\Sigma^\ast = \mathbf{k^{\ast\ast}} - (\mathbf{k^\ast})^T (\mathbf{K} + w^{-1})^{-1} \mathbf{k^\ast}`
        """
        k_star = self.k_star(x_star)
        return (self.k_double_star(x_star) - k_star.T @ self.inv_K_w_inv @ k_star) # .astype("float")

    @traced
    def train(self):
        # Part 1 (a): Latent Posterior P(f | X, Y)
        # Laplace Approximation: q(f | X, Y) = N(f; \hat{f}, A^{-1})
        self.f_hat = self.learn_latent_posterior()
        

    def latent_posterior(self, f):
        return mn.pdf(f, mean = self.f_hat, cov = self.A_inversed)

    def predictive_latent(self, f_star):
        return mn.pdf(f_star, self.mu_star(f_star), self.Sigma_star(f_star))

    def observation_likelihood(self, y, f):
        return (1. / (2 * np.pi * I_0(norm(f)))) * np.exp(np.dot(y.T, f))

    def log_observation_likelihood(self, y, f):
        return np.log(self.observation_likelihood(y, f))


    def predict(self, X):
        k_stars = np.apply_along_axis(self.k_star, 1, X).reshape(len(X), -1, 2)
        k_dstars = np.apply_along_axis(self.k_double_star, 1, X).reshape(len(X), 2, 2)

        mu_star = f_star = self.mu_stars(k_stars)
        Sigma_star = self.Sigma_stars(k_stars, k_dstars)

        ts = np.linspace(0, 1, 10) * (2 * np.pi)
        y_star_domain = np.apply_along_axis(polar_to_cartesian, 0, ts)

        def get_y_star(y_ast):
            def f(mu, Sigma):
                return MultivariateGaussHermiteQuad.predict(2, mu, Sigma, self.quad_samples, self.log_observation_likelihood, y_ast.reshape(-1, 1), mean_only = True)

            return np.vectorize(f, signature="(m),(m,m)->(m)")(mu_star, Sigma_star)
       
        y_stars = np.vectorize(get_y_star, signature="(m)->(n,m)")(y_star_domain.T)
        y_stars_mean = np.mean(y_stars, axis = 0)
        y_stars_std = np.std(y_stars, axis = 0)


        return tuple(map(compose(normalize_radians, cartesian_to_polar), (f_star, y_stars_mean, y_stars_std)))

