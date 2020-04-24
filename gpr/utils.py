from functools import *

import weakref

import numpy as np

import hashlib

from scipy.stats import multivariate_normal as mn
from scipy.special import iv

def I_0(z):
    return iv(0, z)

def I_1(z):
    return iv(1, z)

def compose(*functions):
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)
