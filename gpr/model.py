import sys

from cached_property import cached_property

import numpy as np
from numpy.linalg import *
from sklearn.gaussian_process.kernels import *
from scipy.integrate import *

from autologging import traced

import logging

from gpr.utils import *

class GPR:
    def __init__(self, X, Y, kernel, tol=1e-6, quad_samples=10):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.X = X.copy()
        self.Y = np.apply_along_axis(lambda t: np.array([np.cos(t), np.sin(t)]).T, 0, Y)
        self.logger.info("self.Y.shape: %r", self.Y.shape)
        self.kernel = kernel
        self.tol = tol
        self.quad_samples = quad_samples
        # self.K = self.K()


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
        return (np.apply_along_axis(self.h_prime, 0, norm(f, axis=1))[:, np.newaxis] * f + np.apply_along_axis(self.h, 0, norm(f, axis=1))[:, np.newaxis]).flatten(order="F")

    def w(self, f=None):
        return np.diag(self.w_diagonal(f))

    def w_inversed(self, f=None):
        return np.diag(1. / self.w_diagonal(f))

    def grad_log_P_Y(self, f):
        return - (np.apply_along_axis(self.h, -1, norm(f, axis=1))[:, np.newaxis] * f + self.Y).flatten(order="F")

    @cached_property
    def grad_log_P_Y_f_hat(self):
        return self.grad_log_P_Y(self.f_hat)

    @cached_property
    def K(self):
        K = self.kernel(self.X)
        zero = np.zeros(K.shape)
        return np.concatenate((np.concatenate((K, zero)), np.concatenate((zero, K))), axis=1)

    # @np_cache
    def k_star(self, x_star):
        k_vec = self.kernel(self.X, np.expand_dims(x_star, axis=0))
        zero = np.zeros(k_vec.shape)
        return np.concatenate((np.concatenate((k_vec, zero)), np.concatenate((zero, k_vec))), axis=1)

    # @np_cache
    def k_double_star(self, x_star):
        k_dstar = self.kernel(np.expand_dims(x_star, axis=0))
        return np.array([[k_dstar, 0], [0, k_dstar]])

    def init_f_t(self):
        return np.random.rand(*self.X.shape) * 2 - 1
        
    @traced
    def learn_latent_posterior(self):
        f = self.init_f_t()
        f_new = None

        self.logger.info("Learning Latent Posterior")

        while True:
            # self.logger.debug("inv_w = inv(self.w(f))")
            inv_w = self.w_inversed(f)
            # self.logger.debug("grad = self.grad_log_P_Y(f)")
            grad = self.grad_log_P_Y(f)

            # tmp = self.K + inv_w
            # self.logger.debug("(self.K + inv_w).shape: %r", tmp.shape)
            # self.logger.debug("f_new = self.K @ inv(self.K + inv_w) @ (f_t + inv_w @ grad)")
            f_new = self.K @ inv(self.K + inv_w) @ (f.flatten(order="F") + inv_w @ grad)

            f_new = f_new.reshape(self.X.shape)
            
            error = norm((f_new - f).flatten(order="F")) # , axis = 1)
            self.logger.info("norm(f_new - f): %f", error)
            if error <= self.tol:
                break

            f = f_new

        # f_new becomes f_hat when it converges
        return f_new

    @cached_property
    def A(self):
        if not hasattr(self, "f_hat"):
            raise AttributeError("f_hat has not yet been calculated")

        return self.w + inv(self.K)

    @cached_property    
    def A_inversed(self):
        return inv(self.A)

    # @np_cache
    def mu_star(self, x_star):
        r"""
        :math:`\mu^\ast = (\mathbf{k^\ast})^T \grad \log \Prob{Y}{\hat{f}}`
        """
        return self.k_star(x_star).T @ self.grad_log_P_Y_f_hat

    # @np_cache
    def Sigma_star(self, x_star):
        r"""
        :math:`\Sigma^\ast = \mathbf{k^{\ast\ast}} - (\mathbf{k^\ast})^T (\mathbf{K} + w^{-1})^{-1} \mathbf{k^\ast}`
        """
        k_star = self.k_star(x_star)
        return (self.k_double_star(x_star) - k_star.T @ inv(self.K + self.w_inversed()) @ k_star).astype("float")

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
        return (1. / (2 * np.pi * I_0(norm(f)))) * np.exp(y.T @ f)

    def integrate2d(self, q1, q2, f):
        total = 0
        for (x_1, w_1) in zip(*q1):
            for (x_2, w_2) in zip(*q2):
                total += f(w_1, x_1, w_2, x_2) # w_1 * w_2 * self.observation_likelihood(y_star, f_star(x_1, x_2))
        return total


    def predictive_output_distribution(self, x_star, y_star, mu_star = None, Gamma = None):
        """ Gauss-Hermite Quadrature """
        X_1, W_1 = q1 = np.polynomial.hermite.hermgauss(self.quad_samples)
        X_2, W_2 = q2 = np.polynomial.hermite.hermgauss(self.quad_samples)

        if mu_star is None:
            mu_star = self.mu_star(x_star)

        if Gamma is None:
            Sigma_star = self.Sigma_star(x_star)
            Gamma = cholesky(Sigma_star)

        u = lambda x, y: np.array([[x, y]]).T
        f_star = lambda x1, x2: np.sqrt(2) * Gamma @ u(x1, x2) + mu_star

        # print (W_1, X_1)
        # print (W_2, X_2)

        # total = 0
        # for (w_1, x_1) in zip(W_1, X_1):
        #     for (w_2, x_2) in zip(W_2, X_2):
        #         total += w_1 * w_2 * self.observation_likelihood(y_star, f_star(x_1, x_2))

        return (1. / np.pi) * self.integrate2d(q1, q2, lambda w1, x1, w2, x2: w1 * w2 * self.observation_likelihood(y_star, f_star(x1, x2)))

    # @traced
    def predict_output(self, x_star):
        y_star = lambda x_1, x_2: np.array([x_1, x_2])
        q1 = np.polynomial.legendre.leggauss(self.quad_samples)
        q2 = np.polynomial.legendre.leggauss(self.quad_samples)

        mu_star = self.mu_star(x_star)
        Sigma_star = self.Sigma_star(x_star)
        Gamma = cholesky(Sigma_star)

        return self.integrate2d(q1, q2, lambda w1, x1, w2, x2: w1 * w2 * y_star(x1, x2) * self.predictive_output_distribution(x_star, y_star(x1, x2), mu_star, Gamma))


    @traced
    def predict(self, X):
        f_star = np.apply_along_axis(self.mu_star, 1, X)

        y_star = np.apply_along_axis(self.predict_output, 1, X)
        return f_star, y_star
