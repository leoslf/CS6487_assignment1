import sys
import os

import logging

from autologging import TRACE

from multiprocessing import Pool, log_to_stderr

from gpr.model import *

logger = logging.getLogger(__name__)

def process_user(id):
    def info(fmt, *argv, **kwargs):
        logger.info("[user %02d] %s" % (id, fmt), *argv, **kwargs)

    logger = log_to_stderr()
    filename = os.path.join("social-data", "user_%d.csv" % id)
    dataset = np.loadtxt(filename, delimiter=",")

    train_X, train_Y = dataset[:, [0, 1]], dataset[:, 2] * (2 * np.pi)

    models = []
    for kernel in [RBF]:
        info("Training with kernel %s", kernel.__class__.__name__)
        model = GPR(train_X, train_Y, kernel = kernel())

        f_star, y_star = model.predict(train_X)

        np.savetxt("f_star_%0d.txt" % id, f_star)
        np.savetxt("y_star_%0d.txt" % id, y_star)

        info("saved f_star, y_star")

    return True

if __name__ == "__main__":
    # logging.basicConfig(level=TRACE, stream=sys.stdout, format="%(levelname)s:%(name)s:%(funcName)s:%(message)s")
    logging.basicConfig(level=logging.DEBUG, format="%(name)s:%(levelname)s:%(filename)s:%(funcName)s:%(message)s")
    user_ids = [(i + 1) for i in range(10)]

    with Pool() as pool:
        result = pool.map_async(process_user, user_ids)
        logger.info(result.get()) # timeout=100))

