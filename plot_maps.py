import sys
import os

import numpy as np
import pandas as pd

import imageio
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt4Agg')
from matplotlib.colors import hsv_to_rgb, is_color_like, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from gpr.utils import *


if __name__ == "__main__":
    map_base = imageio.imread("social-data/map.png") # , as_gray = True)

    for kernel in ["RBF", "Matern"]:
        for i in range(10):
            id = i + 1

            df = pd.read_csv("output/%s/output_%02d.csv" % (kernel, id))

            # hsvs = np.column_stack((df.f_star, np.ones(len(df)), minmax_normalization(df.kappa)))

            # print (hsvs)

            # assert is_color_like(hsvs)

            cmap = plt.cm.get_cmap("hsv")

            dpi = 96
            def plot(df, t, kappas, width = map_base.shape[1], height = map_base.shape[0], crop = False, is_gt = False):

                fig, ax = plt.subplots(1, 1, figsize = np.array((width, height)) / float(dpi), dpi = dpi)
                
                ax.imshow(map_base, cmap='gray', vmin=0, vmax=255)

                # rgb = list(map(hsv_to_rgb, [(f_star, kappa * 0.5, 0.5) for (f_star, kappa) in zip(df.f_star, minmax_normalization(df.kappa))]))
                rgb = list(map(hsv_to_rgb, [(t, 0.5 + kappa * 0.5, 0.8) for (t, kappa) in zip(t, kappas)]))
                # print (rgb)
                sc = ax.scatter(df.x, df.y, c = rgb, s=8, marker="x", linewidths=1) # , vmin=0, vmax=1) # , cmap=cmap) # cmap = "hsv") # 

                if crop:
                    plt.axis([np.min(df.x) - 100, np.max(df.x) + 100, np.max(df.y) + 100, np.min(df.y) - 100])
                plt.hsv()
                # sc._A = np.array([])

                # plt.colorbar(sc)
                mappable = plt.cm.ScalarMappable(norm=Normalize(0, 1, clip=False), cmap=cmap)
                mappable.set_array([])

                if is_gt:
                    plt.title("Ground Truth: user %02d" % (id))
                else:
                    plt.title("%s: user %02d" % (kernel, id))

                # create an axes on the right side of ax. The width of cax will be 5%
                # of ax and the padding between cax and ax will be fixed at 0.05 inch.
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(mappable, cax = cax)

                manager = plt.get_current_fig_manager()
                manager.window.showMaximized()

            plot(df, df.f_star, minmax_normalization(df.kappa))

            # plt.show()

            plt.savefig("output/%s/map_%02d.png" % (kernel, id), dpi = dpi)
            plt.close()

            plot(df, df.f_star, minmax_normalization(df.kappa), width = (np.max(df.x) + 100 -  (np.min(df.x) - 100)), height = (np.max(df.y) + 100) - (np.min(df.y) - 100), crop = True)
            plt.savefig("output/%s/zoomed_map_%02d.png" % (kernel, id))
            

            plot(df, df.t, np.ones(len(df.t)), is_gt = True)

            plt.savefig("output/%s/map_gt_%02d.png" % (kernel, id), dpi = dpi)


