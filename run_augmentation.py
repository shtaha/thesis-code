import importlib.util
import json
import os
from distutils.dir_util import copy_tree

import numpy as np
import pandas as pd

from lib.chronics import (
    is_prods_file,
    is_loads_file,
    read_bz2_to_dataframe,
    save_dataframe_to_bz2,
    augment_chronic,
)
from lib.data_utils import make_dir
from lib.visualizer import pprint

case_name = "l2rpn_2019"
art_case_name = case_name + "_art"

datasets_path = os.path.join(os.path.expanduser("~"), "data_grid2op")

case_path = os.path.join(datasets_path, case_name)
art_case_path = make_dir(os.path.join(datasets_path, art_case_name))

if not len(os.listdir(art_case_path)):
    copy_tree(case_path, art_case_path)

"""
    Load config file
"""
spec = importlib.util.spec_from_file_location(
    ".", os.path.join(art_case_path, "config.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
config = module.config

case_chronics = os.path.join(case_path, "chronics")
art_case_chronics = os.path.join(art_case_path, "chronics")

max_ps = 0.0
np.random.seed(5)
for chronic_idx, chronic in enumerate(os.listdir(art_case_chronics)):
    targets = dict()
    if case_name == "rte_case5_example":
        min_p, max_p = (3.0, 5.0)
        augmentation = "overload"
    elif case_name == "l2rpn_2019":
        if chronic_idx <= 10:
            min_p, max_p = (0.05, 0.20)
            augmentation = "overload"
        elif 40 < chronic_idx <= 50:
            min_p, max_p = (0.25, 0.30)
            augmentation = "targeted"
            targets["gen_ids"] = [2, 4]
            targets["load_ids"] = [3, 4]
        else:
            min_p, max_p = (0.15, 0.20)
            augmentation = "overload"
    else:
        min_p, max_p = (0.05, 0.10)
        augmentation = "overload"

    pprint("    - Augmenting:", chronic)

    chronic_dir = os.path.join(case_chronics, chronic)
    art_chronic_dir = os.path.join(art_case_chronics, chronic)

    prods_file = [file for file in os.listdir(chronic_dir) if is_prods_file(file)]
    loads_file = [file for file in os.listdir(chronic_dir) if is_loads_file(file)]
    assert len(prods_file) == 1 and len(loads_file) == 1

    prods = read_bz2_to_dataframe(os.path.join(chronic_dir, prods_file[0]), sep=";")
    loads = read_bz2_to_dataframe(os.path.join(chronic_dir, loads_file[0]), sep=";")

    prods, loads, p = augment_chronic(
        prods,
        loads,
        config=config,
        augmentation=augmentation,
        min_p=min_p,
        max_p=max_p,
        targets=targets,
    )

    with open(os.path.join(art_chronic_dir, "augmentation.json"), "w") as f:
        json.dump(
            dict(
                p=p,
                max_p=max_p,
                min_p=min_p,
                augmentation=augmentation,
                targets=targets,
            ),
            f,
            indent=2,
        )

    save_dataframe_to_bz2(prods, art_chronic_dir, prods_file[0], sep=";")
    save_dataframe_to_bz2(loads, art_chronic_dir, loads_file[0], sep=";")

    max_ps = np.maximum(max_ps, max_p)

    if chronic_idx == 70:
        break

prods_charac = pd.read_csv(os.path.join(case_path, "prods_charac.csv"))
prods_charac["Pmax"] = (1 + max_ps) * prods_charac["Pmax"]
prods_charac["max_ramp_up"] = (1 + max_ps) * prods_charac["max_ramp_up"]
prods_charac["max_ramp_down"] = (1 + max_ps) * prods_charac["max_ramp_down"]

prods_charac.to_csv(os.path.join(art_case_path, "prods_charac.csv"))
