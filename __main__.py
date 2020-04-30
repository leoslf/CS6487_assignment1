import sys
import os

import logging

from autologging import TRACE

import pandas as pd

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

    with pushd("output", create=True):
        for kernel in ["RBF", "WhiteKernel"]:
            info("Training with kernel %s", kernel)
            model = GPR(train_X, train_Y, kernel = get_kernel(kernel))

            f_star, y_star, y_star_std = model.predict(train_X)

            with pushd(kernel, create=True):
                output_filename = "output_%02d.csv" % id
                output = np.column_stack((train_X, normalize_radians(train_Y), f_star, y_star, y_star_std))
                output_df = pd.DataFrame(output, columns = ["x", "y", "t", "f_star", "y_star", "y_star_std"])
                output_df.to_csv(output_filename, index=False)
                info("saved output: \"%s\"" % output_filename)

    return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(name)s:%(levelname)s:%(filename)s:%(funcName)s: %(message)s")
    user_ids = [(i + 1) for i in range(1)]

    with Pool() as pool:
        result = pool.map_async(process_user, user_ids)
        logger.info(result.get())

