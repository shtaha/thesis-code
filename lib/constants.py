import os
import matplotlib.colors as mcolors


class Constants(object):
    DATASET_DIR = os.path.join(os.path.expanduser("~"), "data_grid2op")
    # DATASET_DIR = "/local/rsikonja/data_grid2_op"

    RESULTS_DIR = "./results"
    EXPERIENCE_DIR = "./experience"

    ENV_NAME = "l2rpn_wcci_2020"

    MATPLOTLIB_STYLE = "seaborn"
    COLORS = list(mcolors.TABLEAU_COLORS)
    FIG_SIZE = (8, 4)
    OUT_FORMAT = "pdf"
