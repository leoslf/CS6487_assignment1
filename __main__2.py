import sys
import os

import logging

from autologging import TRACE

from gpr.model_tf import *

if __name__ == "__main__":
    logging.basicConfig(level=TRACE, stream=sys.stdout, format="%(levelname)s:%(name)s:%(funcName)s:%(message)s")
    dataset = []
    for i in range(1):
        filename = os.path.join("social-data", "user_%d.csv" % (i + 1))
        dataset.append(np.loadtxt(filename, delimiter=","))

    dataset = np.concatenate(dataset, axis = 0)
    train_X, train_Y = dataset[:, [0, 1]], dataset[:, 2] * (2 * np.pi)

    print (train_X, train_Y)

    models = []
    for kernel in [RBF]:
        model = GPR(train_X, train_Y, kernel = kernel())

        f_star, y_star = model.predict(train_X)

        np.savetxt("f_star.txt", f_star)
        np.savetxt("y_star.txt", y_star)

        print (f_star, y_star)
        

