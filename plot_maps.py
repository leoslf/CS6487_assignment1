import sys
import os

import numpy as np
import pandas as pd

import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, is_color_like
from gpr.utils import *


if __name__ == "__main__":
    map_base = imageio.imread("social-data/map.png")

    for i in range(10):
        id = i + 1

        df = pd.read_csv("output_%02d.csv" % id)

        hsvs = np.column_stack((df[["f_star", "y_star"]].values, np.ones(len(df))))

        print (hsvs)

        assert is_color_like(hsvs)
        
        plt.imshow(map_base)
        plt.scatter(df.x, df.y, c = hsvs, cmap="hsv")

        plt.colorbar()
        plt.show()
        

