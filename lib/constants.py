import os

import matplotlib.colors as mcolors


class Constants(object):
    DATASET_DIR = os.path.join(os.path.expanduser("~"), "data_grid2op")
    # DATASET_DIR = "/local/rsikonja/data_grid2op"

    RESULTS_DIR = "./results"
    EXPERIENCE_DIR = "./experience"

    ENV_NAME = "l2rpn_wcci_2020"

    MATPLOTLIB_STYLE = "seaborn"
    COLORS = list(mcolors.TABLEAU_COLORS)
    FIG_SIZE = (8, 6)
    OUT_FORMAT = "pdf"

    LW = 0.5
    AXIS_FONT_SIZE = 18
    TICKS_FONT_SIZE = 16
    LEGEND_FONT_SIZE = 16
    FONT_SIZE = 16
