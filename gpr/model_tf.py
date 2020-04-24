import sys
import os

from cached_property import cached_property

import numpy as np
# from numpy.linalg import *

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.math import *

from tensorflow.linalg import *
from sklearn.gaussian_process.kernels import *
from scipy.integrate import *

from sklearn.utils.extmath import cartesian

from autologging import traced

import logging

from gpr.utils import *

class GPR:
    def __init__(self, X, Y, kernel, tol=1e-6, quad_samples=10):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.kernel = kernel
        self.X_values = X.copy()
        self.X = tf.constant(self.X_values, name="X", dtype=tf.float64)
        self.X_shape = X.shape
        self.Y = tf.constant(np.apply_along_axis(lambda t: np.array([np.cos(t), np.sin(t)]).T, 0, Y), name="Y", dtype=tf.float64)
        self.K = self.K(X)
        self.logger.info("self.Y.shape: %r", self.Y.shape)
        self.tol = tol
        self.quad_samples = quad_samples
        # self.f_init = tf.Variable(tf.random.uniform(self.X_shape, -1, 1, dtype=tf.float64), shape=self.X_shape, dtype=tf.float64)
        self.f_init = tf.Variable(self.init_f_t, shape=self.X_shape, dtype=tf.float64)
        # self.K = self.K()
        self.model_path = "./model/"
        self.f_hat = tf.get_variable("f_hat", dtype=tf.float64, shape=self.X_shape)

        self.config = tf.ConfigProto()
        # self.sess = tf.Session(config=self.config)
        self.saver = tf.train.Saver(max_to_keep=1)
        self.f_star_quad_weight, self.f_star_quad_vectors = self.f_star_quad()
        self.y_star_quad_weight, self.y_star_quad_vectors = self.y_star_quad()

        self.init_op = tf.global_variables_initializer()


        if not os.path.exists(self.model_path):
            self.train()

    def f_star_quad(self):
        X_1, W_1 = q1 = np.polynomial.hermite.hermgauss(self.quad_samples)
        X_2, W_2 = q2 = np.polynomial.hermite.hermgauss(self.quad_samples)
        return tf.constant(np.outer(W_1, W_2), name="f_star_quad_weight"), tf.constant(cartesian([X_1, X_2]), name="f_star_quad_vectors")

    def y_star_quad(self):
        X_1, W_1 = q1 = np.polynomial.legendre.leggauss(self.quad_samples)
        X_2, W_2 = q2 = np.polynomial.legendre.leggauss(self.quad_samples)
        return tf.constant(np.outer(W_1, W_2), name="y_star_quad_weight"), tf.constant(cartesian([X_1, X_2]), name="y_star_quad_vectors")

    def h(self, k):
        return bessel_i1(k) / (bessel_i0(k) * k)

    def h_prime(self, k):
        frac = bessel_i1(k) / bessel_i0(k)
        return 1 - (1 / k + 1 / k ** 2) * frac - frac ** 2

    def w_diagonal(self, f=None):
        if f is None:
            if not hasattr(self, "f_hat"):
                raise AttributeError("f_hat has not been calculated")
            f = self.f_hat

        h_prime = tf.map_fn(self.h_prime, tf.norm(f, axis=1))
        h = tf.map_fn(self.h, tf.norm(f, axis=1))
        return self.flatten(tf.multiply(tf.expand_dims(h_prime, -1), f) + tf.expand_dims(h, -1))

    def w(self, f=None):
        return diag(self.w_diagonal(f))

    def w_inversed(self, f=None):
        return diag(tf.reciprocal(self.w_diagonal(f)))

    def flatten(self, m, matrix_form=False):
        shape = [-1]
        if matrix_form:
            shape = [-1, 1]
        if len(m.shape) > 1:
            m = transpose(m)
        return tf.reshape(m, shape=shape)

    def grad_log_P_Y(self, f):
        h = tf.map_fn(self.h, norm(f, axis=1))
        return - self.flatten(tf.multiply(tf.expand_dims(h, -1), f) + self.Y, matrix_form=True)

    @cached_property
    def grad_log_P_Y_f_hat(self):
        return self.grad_log_P_Y(self.f_hat)

    def K(self, X):
        K = self.kernel(X)
        zero = np.zeros(K.shape)
        return tf.constant(np.concatenate((np.concatenate((K, zero)), np.concatenate((zero, K))), axis=1), name="K", dtype=tf.float64)

    # @np_cache
    def k_star(self, x_star):
        k_vec = self.kernel(self.X_values, np.expand_dims(x_star, axis=0))
        zero = np.zeros(k_vec.shape, dtype=np.float64)
        return np.concatenate((np.concatenate((k_vec, zero), axis=0), np.concatenate((zero, k_vec), axis=0)), axis=1)

    # @np_cache
    def k_double_star(self, x_star):
        k_dstar = self.kernel(np.expand_dims(x_star, axis=0))
        return np.array([[k_dstar, 0], [0, k_dstar]], dtype=np.float64) # np.concatenate((np.concatenate((k_dstar, 0), axis=0), np.concatenate((0, k_dstar), axis=0)), axis=1)

    def init_f_t(self):
        return np.random.rand(*self.X_shape) * 2 - 1
        
    @traced
    def learn_latent_posterior(self):
        f = self.f_init  # self.init_f_t()
        f_new = None

        self.logger.info("Learning Latent Posterior")

        while True:
            # self.logger.debug("inv_w = inv(self.w(f))")
            inv_w = self.w_inversed(f)
            # self.logger.debug("grad = self.grad_log_P_Y(f)")
            grad = self.grad_log_P_Y(f)

            # self.logger.debug("(self.K + inv_w).shape: %r", tmp.shape)
            # self.logger.debug("f_new = self.K @ inv(self.K + inv_w) @ (f_t + inv_w @ grad)")
            # f_new = matmul(self.K, matmul(inv(self.K + inv_w), (self.flatten(f, matrix_form=True) + matmul(inv_w, grad))))
            f_new = self.K @ inv(self.K + inv_w) @ (self.flatten(f, matrix_form=True) + inv_w @ grad)

            f_new = tf.reshape(f_new, shape = self.X_shape)
            
            error = norm(self.flatten(f_new - f)).eval()
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
    def mu_star(self, k_star):
        r"""
        :math:`\mu^\ast = (\mathbf{k^\ast})^T \grad \log \Prob{Y}{\hat{f}}`
        """
        return tf.transpose(k_star) @ self.grad_log_P_Y_f_hat

    # @np_cache
    def Sigma_star(self, k_star, k_double_star):
        r"""
        :math:`\Sigma^\ast = \mathbf{k^{\ast\ast}} - (\mathbf{k^\ast})^T (\mathbf{K} + w^{-1})^{-1} \mathbf{k^\ast}`
        """
        return k_double_star - tf.transpose(k_star) @ inv(self.K + self.w_inversed()) @ k_star

    @traced
    def train(self):
        # Part 1 (a): Latent Posterior P(f | X, Y)
        # Laplace Approximation: q(f | X, Y) = N(f; \hat{f}, A^{-1})
        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)
            # Train our model
            self.f_hat = tf.assign(self.f_hat, self.learn_latent_posterior())

            # saving the model
            saved_path = self.saver.save(sess, self.model_path, global_step=1)
            self.logger.info("saved_path: %s", saved_path)
        

    def latent_posterior(self, f):
        return mn.pdf(f, mean = self.f_hat, cov = self.A_inversed)

    def predictive_latent(self, f_star):
        return mn.pdf(f_star, self.mu_star(f_star), self.Sigma_star(f_star))

    def observation_likelihood(self, y, f):
        return (1. / (2 * np.pi * bessel_i0(norm(f)))) * tf.exp(transpose(self.flatten(y, matrix_form=True)) @ self.flatten(f, matrix_form=True))

    def integrate2d(self, q1, q2, f):
        total = 0
        for (x_1, w_1) in zip(*q1):
            for (x_2, w_2) in zip(*q2):
                total += f(w_1, x_1, w_2, x_2) # w_1 * w_2 * self.observation_likelihood(y_star, f_star(x_1, x_2))
        return total

    def predictive_output_distribution(self, y_star, mu_star, Gamma):
        """ Gauss-Hermite Quadrature """

        f_star = lambda u: np.sqrt(2) * Gamma @ tf.expand_dims(u, -1) + mu_star
        return (1. / np.pi) * tf.reduce_sum(self.f_star_quad_weight * tf.reshape(tf.map_fn(compose(partial(self.observation_likelihood, y_star), f_star), self.f_star_quad_vectors), (self.quad_samples, self.quad_samples)))

    # @traced
    def predict_output(self, mu_star, Gamma):
        weights = self.y_star_quad_weight * tf.reshape(tf.map_fn(partial(self.predictive_output_distribution, mu_star=mu_star, Gamma=Gamma), self.y_star_quad_vectors), (self.quad_samples, self.quad_samples))
        return self.flatten(tf.reduce_sum(tf.expand_dims(weights, -1) * tf.reshape(self.y_star_quad_vectors, (self.quad_samples, self.quad_samples, 2)), axis=0))

    @traced
    def predict(self, X):
        X_tensor = tf.placeholder(dtype=tf.float64, shape=(None, *self.X_shape[1:]))
        k_star = tf.placeholder(dtype=tf.float64, shape=(None, self.X_shape[0] * 2, 2))
        k_double_star = tf.placeholder(dtype=tf.float64, shape=(None, 2, 2))
        def _predict():
            # This is effectively f_star
            f_star = mu_star = tf.map_fn(self.mu_star, k_star)
            Sigma_star = tf.map_fn(lambda tup: self.Sigma_star(*tup), (k_star, k_double_star), dtype=tf.float64)
            Gamma = tf.map_fn(cholesky, Sigma_star)

            y_star = tf.map_fn(lambda tup: self.predict_output(*tup), (mu_star, Gamma), dtype=tf.float64)

            return f_star, y_star

        with tf.Session(config=self.config) as sess:
            self.saver.restore(sess, self.model_path)
            self.f_hat = tf.train.load_variable(self.model_path, "f_hat")
            return sess.run(_predict(), feed_dict={X_tensor: X, k_star: np.apply_along_axis(self.k_star, -1, X), k_double_star: np.apply_along_axis(self.k_double_star, -1, X)})

