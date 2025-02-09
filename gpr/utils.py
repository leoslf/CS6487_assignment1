import sys
import os

import weakref

import numpy as np
from numpy.linalg import *

from functools import *
from itertools import *

import logging

from contextlib import contextmanager

import hashlib

from sklearn.gaussian_process.kernels import *

from scipy.stats import multivariate_normal as mn
from scipy.special import iv


logger = logging.getLogger(__name__)

def I_0(z):
    assert np.all(z >= 0)
    return iv(0, z)

def I_1(z):
    assert np.all(z >= 0)
    return iv(1, z)

def compose(*functions):
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

def cartesian_product(S, n):
    r""" Cartesian Product of a sequence :math:`S \in \mathbb{R}^k`

    Returns:
        Cartesian Product of :math:`S`: :math:`S^n \in \mathbb{R}^{k^{n - 1} \times n}
    """
    assert len(S.shape) == 1
    return np.fromiter(chain(*product(S, repeat=n)), dtype=S.dtype).reshape(n, -1)

def covariance_matrix(weights, xs, mean):
    # logger.info("xs.shape: %r, mean.shape: %r", xs.shape, mean.shape)
    deviations = (xs - mean.reshape(-1, 1)).flatten()
    outers = np.apply_along_axis(lambda deviation: deviation @ deviation.T, 0, deviations)
    return np.sum(weights * outers, axis=-1)

def polar_to_cartesian(t):
    return np.row_stack((np.cos(t), np.sin(t)))

def cartesian_to_polar(cartesian):
    values = np.arctan2(cartesian[:, 0], cartesian[:, 1]).reshape(-1, 1)
    values[values < 0] += 2 * np.pi
    return values

def normalize_radians(radians):
    return radians / (2 * np.pi)

def get_kernel(name):
    return globals()[name]()

def partitioned_matrix(M11, M12, M21, M22):
    assert M11.shape[0] == M12.shape[0]
    assert M11.shape[1] == M21.shape[1]
    assert M22.shape[0] == M21.shape[0]
    assert M22.shape[1] == M12.shape[1]
    return np.concatenate((np.concatenate((M11, M21)), np.concatenate((M12, M22))), axis=1)

def diagonally_partitioned_matrix(M11, M22):
    M12 = np.zeros((M11.shape[0], M22.shape[1]))
    M21 = np.zeros((M22.shape[0], M11.shape[1]))
    return partitioned_matrix(M11, M12, M21, M22)

@contextmanager
def pushd(new_dir, create=False):
    previous_dir = os.getcwd()
    if not os.path.exists(new_dir):
        if create:
            os.mkdir(new_dir)
        else:
            raise ValueError("directory: \"%s\" does not exists" % new_dir)

    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)

class MultivariateGaussHermiteQuad:
    def __init__(self, d, mu, Sigma, num_points):
        self.d = d
        self.mu = mu.reshape(self.d, 1)
        self.Sigma = Sigma.reshape(self.d, self.d).astype(np.float32)
        self.num_points = num_points
        self.X, self.W = self.get_quadrature_points()

    @classmethod
    def predict(cls, d, mu, Sigma, num_points, *argv, **kwargs):
        self = cls(d, mu, Sigma, num_points)
        return self.evaluate(*argv, **kwargs)

    @property
    def n(self):
        """ Number of points in X """
        return len(self.X.T)

    def get_quadrature_points(self):
        r""" Generating Quadrature Points

        Univariate Gauss-Hermite Quadrature
        Let :math:`\{w_i\}_{i = 1}^{n}, \{x_i\}_{i = 1}^{n}` be the weights and quadrature points.

        """
        # logger.info("Generating univariate Gauss-Hermite Quadrature points with num_points: %d", self.num_points)
        x, w = np.polynomial.hermite.hermgauss(self.num_points)
        # Apply normal pdf
        x *= np.sqrt(2)
        w /= np.sqrt(np.pi)
        
        # logger.debug("Computing the cartesian product of X")
        X = cartesian_product(x, self.d)
        # logger.debug("X.shape: %r", X.shape)
        # Transforming X into our prior distribution
        X = self.transform(X, self.mu, self.Sigma)

        # logger.debug("Computing the cartesian product of W")
        _W = cartesian_product(w, self.d)
        # Using sum of logs to reduce inaccuracy
        # W = np.array(list((compose(np.sum, np.log)(w) for w in _W.T)))
        W = np.vectorize(compose(np.sum, np.log), signature="(d)->()")(_W.T)

        return X, W

    def transform(self, X, mu, Sigma):
        r""" Transform :math:`X \in \mathbb{R}^{n \times d}` into the prior distribution  """
        lambdas, V = eig(Sigma)

        if np.any(lambdas < 0):
            raise ValueError("Sigma is not Positive Definite")
        
        return (V @ np.diag(np.sqrt(lambdas)) @ X) + mu


    def evaluate(self, f: callable, *argv, mean_only = False, **kwargs):
        r""" Evaluate the integral
        The evaluated function has to have a Multivariate Gaussian prior :math:`\Prob{X}{\mu, \Sigam}.

        The integral
        .. math::
            \E[X \given \mu, \Sigma]{f(X)}  &= \Prob{\int_{-\infty}^{\infty} f(X) \Prob[N}{X}{\mu, \Sigma}\ dX \\

        Args:
            f:      The log-likelihood of the parameter to be evaluated
            argv:   Argument vector to be passed into f
            kwargs: Keyword Arguments to be passed into f
        """
        if not callable(f):
            raise ValueError("f, the log-likelihood to be evaluated, must be a callable")

        # logger.info("self.X.shape: %r", self.X.shape)

        terms = np.vectorize(lambda x: f(*argv, x.reshape(-1, 1), **kwargs), signature="(d)->()")(self.X.T)
        # with np.printoptions(suppress=True, precision=4):
        #     logger.info("terms: %r", np.exp(terms))
        #     logger.info("terms.shape: %r", terms.shape)
        #     logger.info("self.W.shape: %r", self.W.shape)
        
        terms += self.W
        # terms = np.array(list((f(*argv, x.reshape(-1, 1), **kwargs) for x in self.X.T))).reshape(-1) + self.W

        # numerical safeguard: to prevent blowing up when taking exp
        # terms += 700 - np.max(terms)

        # Back to normal scale
        terms = np.exp(terms).reshape(1, -1)
        mean = np.sum(terms, axis=1)

        if mean_only:
            return mean
        return probabilities, self.X, mean





