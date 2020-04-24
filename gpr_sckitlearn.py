import sys
import os

import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

if __name__ == "__main__":
    dataset = []
    for i in range(10):
        filename = os.path.join("social-data", "user_%d.csv" % (i + 1))
        dataset.append(np.loadtxt(filename, delimiter=","))

    dataset = np.concatenate(dataset, axis = 0)
    train_X, train_Y = dataset[:, [0, 1]], dataset[:, 2] * (2 * np.pi)
    print (train_X.shape)

    kernel = DotProduct() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(train_X, train_Y)

    predict_Y = gpr.predict(train_X)
    print (predict_Y)
    print (train_Y - predict_Y)

