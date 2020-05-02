import numpy as np
import pandas as pd
import imageio

import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from matplotlib.cbook import get_sample_data
from matplotlib._png import read_png
from mpl_toolkits.mplot3d import axes3d, Axes3D, art3d
from mpl_toolkits.mplot3d.axis3d import Axis
### patch start ###
if not hasattr(Axis, "_get_coord_info_old"):
    def _get_coord_info_new(self, renderer):
        mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
        mins += deltas / 4
        maxs -= deltas / 4
        return mins, maxs, centers, deltas, tc, highs
    Axis._get_coord_info_old = Axis._get_coord_info  
    Axis._get_coord_info = _get_coord_info_new
### patch end ###

def plot_image(img):
    # print ("img_x.shape: %r, img_y.shape: %r, img_z.shape: %r" % (img_x.shape, img_y.shape, img_z.shape))
    # rstride=20, cstride=20, 
    ax.plot_surface(img_x, img_y, img_z, facecolors=img, alpha = 1.0, linewidth=0, antialiased=False, zorder=-1)

def invert_yaxis():
    ax.set_ylim3d(ax.get_ylim3d()[::-1])

def sigma_coordinates(xs, ys, means, stds):
    pos = np.column_stack((xs, ys, means + stds))
    neg = np.column_stack((xs, ys, means - stds))

    # these needed to wrap around
    too_high_index = pos[:, 2] > 1
    too_low_index  = neg[:, 2] < 0

    too_high_pos = pos[too_high_index]
    too_high_neg = neg[too_high_index]
    too_high_pos[:, 2] %= 1

    too_low_pos  = pos[too_low_index]
    too_low_neg  = neg[too_low_index]
    too_low_neg[:, 2] %= 1

    # ---
    #  |  (will be at bottom)
    wrap_around_top = np.row_stack((too_high_pos, too_low_pos))
    wrap_around_top_complement = wrap_around_top.copy()
    wrap_around_top_complement[:, 2] = 0
    #  |
    # ---
    wrap_around_bot = np.row_stack((too_low_pos, too_high_pos))
    wrap_around_bot_complement = wrap_around_bot.copy()
    wrap_around_bot_complement[:, 2] = 1

    # these are good
    pos = pos[~(too_high_index | too_low_index)]
    neg = neg[~(too_high_index | too_low_index)]

    # coordinates, line_segment_pairs
    coordinates = (np.row_stack((pos, wrap_around_top)), np.row_stack((neg, wrap_around_bot)))
    line_segment_pairs = np.row_stack((pos, wrap_around_top, wrap_around_bot_complement)), np.row_stack((neg, wrap_around_top_complement, wrap_around_bot))
    return coordinates, (pos, neg) # , (wrap_around_top, wrap_around_top_complement), (wrap_around_bot_complement, wrap_around_bot) # line_segment_pairs

def errorbar3d_z(xs, ys, means, stds, mean_label, std_label, color, alpha, mean_shadow = False):
    kwargs = {
        "color": color,
        "zorder": 10,
    }

    # Means
    ax.scatter(xs, ys, means, label = mean_label, marker = "x", s = 1, **kwargs)

    # Mean shadow
    if mean_shadow:
        ax.scatter(xs, ys, np.zeros(len(means)), c = "black", alpha = 1.0, label = "shadow") # label = "shadow", 


    # Error bar lines
    # points, line_segment_pairs = sigma_coordinates(xs, ys, means, stds)
    # points, pairs_good, pairs_bad_1, pairs_bad_2 = sigma_coordinates(xs, ys, means, stds)
    points, pairs_good = sigma_coordinates(xs, ys, means, stds)
    barline_collection = art3d.Line3DCollection(list(zip(*pairs_good)), colors = color, alpha = alpha, linewidths = 0.5)
    ax.add_collection(barline_collection)

    # barline_collection = art3d.Line3DCollection(list(zip(*pairs_bad_1)), colors = "purple", alpha = alpha, linewidths = 0.5)
    # ax.add_collection(barline_collection)
    
    # barline_collection = art3d.Line3DCollection(list(zip(*pairs_bad_2)), colors = "blue", alpha = alpha, linewidths = 0.5)
    # ax.add_collection(barline_collection)

    ax.scatter(*points, marker = "_", label = std_label, c = color, alpha = alpha)

def set_axes_labels(labels):
    for f, label in zip([ax.set_xlabel, ax.set_ylabel, ax.set_zlabel], labels):
        f(label)

if __name__ == "__main__":
    stride = 50
    map_base = read_png("social-data/map.png") # , as_gray = True)
    height, width = map_base.shape[:2]
    img_y, img_x = np.ogrid[0:height:stride, 0:width:stride]
    img_z = np.zeros(img_x.shape)
    img = map_base[::stride, ::stride, :3]

    for kernel in ["RBF", "Matern"]: # 
        for i in range(10):
            id = i + 1

            df = pd.read_csv("output/%s/output_%02d.csv" % (kernel, id))

            fig = plt.figure()
            ax = Axes3D(fig)
            ax.set_zlim(0, 1)
            ax.set_xlim(0, width - 1)
            ax.set_ylim(0, height - 1)
            ax.margins(0, 0, 0)

            # print ("margins: ", ax.margins())

            # plot_image(img)
            errorbar3d_z(df.x, df.y, df.y_star, df.y_star_std, mean_label = r"$y^\ast$", std_label = r"$\sigma_{y^\ast}$", color = "red", alpha = 0.2) # , mean_shadow = True)

            ax.scatter(df.x, df.y, df.t, label = "ground truth $t$", c="green", alpha = 1.0, s=1)
            ax.scatter(df.x, df.y, np.zeros(len(df.t)), label = "shadow of $t$", c="black", alpha = 0.4, s=1)

            set_axes_labels(["x", "y", "t"])

            invert_yaxis()


            plt.legend()

            plt.title("%s: user %02d" % (kernel, id))

            manager = plt.get_current_fig_manager()
            manager.window.showMaximized()

            # plt.show()

            plt.savefig("output/%s/map3d_without_img_%02d.png" % (kernel, id))
            # plt.savefig("output/%s/map3d_%02d.png" % (kernel, id))

            # ax.set_xlim(left=np.min(df.x) - 100, right=np.max(df.x) + 100)
            # # ax.set_xlim3d(left=np.min(df.x) - 100, right=np.max(df.x) + 100)
            # ax.set_ylim(bottom=np.min(-df.y) - 100, top=np.max(-df.y) + 100)  # .axis([np.min(df.x) - 100, np.max(df.x) + 100, np.max(-df.y) + 100, np.min(-df.y) - 100])
            # # ax.set_ylim3d(bottom=np.min(-df.y) - 100, top=np.max(-df.y) + 100)  # .axis([np.min(df.x) - 100, np.max(df.x) + 100, np.max(-df.y) + 100, np.min(-df.y) - 100])
            # ax.set_zlim3d(bottom = 0, top = 1)
            # plt.savefig("output/%s/zoomed_map3d_%02d.png" % (kernel, id))

            plt.close(fig)

            # # x-t graphs
            # interval = 10
            # i = 1
            # for y in range(int(np.min(-df.y)), int(np.max(-df.y)) + 1, interval):
            #     fig, ax = plt.subplots(1, 1)
            #     index = ((-df.y) < y + interval) & ((-df.y) >= y)
            #     if np.count_nonzero(index) == 0:
            #         continue
            #     ax.scatter(df.x[index], df.t[index], label = "ground truth $t$", c = "green", alpha = 0.7)
            #     ax.scatter(df.x[index], df.y_star[index], label=r"$y^\ast$", color = "red", alpha = 0.7)
            #     ax.errorbar(df.x[index], df.y_star[index], yerr=df.y_star_std[index], color = "red", label = r"$\sigma_{y^\ast}$", alpha = 0.3, fmt='o')

            #     plt.title("%s: user %02d x-t, %d <= y < %d" % (kernel, id, y, y + interval))

            #     plt.savefig("output/%s/user_%02d_x_t_%03d.png" % (kernel, id, i))
            #     plt.close(fig)
            #     i += 1

            # i = 1
            # # y-t graphs
            # for x in range(int(np.min(df.x)), int(np.max(df.x)) + 1, interval):
            #     fig, ax = plt.subplots(1, 1)
            #     index = (df.x < x + interval) & (df.x >= x)
            #     if np.count_nonzero(index) == 0:
            #         continue
            #     ax.scatter(-df.y[index], df.t[index], label = "ground truth $t$", c = "green", alpha = 0.7)
            #     ax.scatter(-df.y[index], df.y_star[index], label=r"$y^\ast$", color = "red", alpha = 0.7)
            #     ax.errorbar(-df.y[index], df.y_star[index], yerr=df.y_star_std[index], color = "red", label = r"$\sigma_{y^\ast}$", alpha = 0.3, fmt='o')

            #     plt.title("%s: user %02d y-t, %d <= x < %d" % (kernel, id, x, x + interval))
            #     plt.savefig("output/%s/user_%02d_y_t_%03d.png" % (kernel, id, i))
            #     plt.close(fig)
            #     i += 1


