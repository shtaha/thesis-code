import bz2
import datetime
import os
from io import StringIO

import numpy as np
import pandas as pd

from MIP_oracle2.visualizer import pprint


def get_sorted_chronics(env):
    chronics_dir = env.chronics_handler.path
    chronics = os.listdir(chronics_dir)

    # Filter meta files
    chronics = list(
        filter(lambda x: os.path.isdir(os.path.join(chronics_dir, x)), chronics)
    )

    if env.name in [
        "rte_case5_example",
        "rte_case5_example_art",
        "l2rpn_2019",
        "l2rpn_2019_art",
    ]:
        chronics_sorted = sorted(chronics, key=lambda x: int(x))
    else:
        chronics_sorted = sorted(
            chronics,
            key=lambda x: (
                datetime.datetime.strptime(x.split("_")[1].capitalize(), "%B").month,
                int(x.split("_")[-1]),
            ),
        )

    return chronics_dir, chronics, chronics_sorted


def ids_to_names(names_chronics_to_grid):
    mapping = dict()
    for org_name, grid_name in names_chronics_to_grid.items():
        el_id = int(grid_name.split("_")[-1])
        mapping[el_id] = org_name

    return mapping


def is_prods_file(file_name):
    return (
        ("prod_p" in file_name or "prods_p" in file_name)
        and "planned" not in file_name
        and "forecasted" not in file_name
    )


def is_loads_file(file_name):
    return (
        ("load_p" in file_name or "loads_p" in file_name)
        and "planned" not in file_name
        and "forecasted" not in file_name
    )


def read_bz2_to_dataframe(file_path, sep=";"):
    pprint("        - Loading from:", "/".join(file_path.split("\\")[-4:]))
    data_csv = bz2.BZ2File(file_path).read().decode()
    return pd.read_csv(StringIO(data_csv), sep=sep)


def save_dataframe_to_bz2(data, save_dir, file_name, sep=";"):
    file_path = os.path.join(save_dir, file_name)
    pprint("        - Saving to:", "/".join(file_path.split("\\")[-4:]))
    data.to_csv(file_path, index=False, sep=sep, compression="bz2")
    return file_path


def overload_injections(data, p=0.05):
    data = data * (1 + p)
    return data


def augment_chronic(
    prods,
    loads,
    config,
    augmentation="overload",
    min_p=0.05,
    max_p=0.15,
    targets=None,
    verbose=False,
):
    p = np.random.uniform(low=min_p, high=max_p)

    pprint("        - Augmenting:", f"p = {p}")
    pprint("        - Type:", augmentation)
    pprint("        - p:", [min_p, max_p])
    pprint("        - Targets:", str(targets))

    if verbose or True:
        pprint(
            "        - Loads:", ["{:.2f}".format(l) for l in loads.mean(axis=0).values]
        )
        pprint(
            "        - Prods:", ["{:.2f}".format(g) for g in prods.mean(axis=0).values]
        )

    if augmentation == "overload":
        prods = overload_injections(prods, p=p)
        loads = overload_injections(loads, p=p)
    else:
        if targets and config:
            prod_ids_to_names = ids_to_names(config["names_chronics_to_grid"]["prods"])
            load_ids_to_names = ids_to_names(config["names_chronics_to_grid"]["loads"])

            gen_names = [prod_ids_to_names[gen_id] for gen_id in targets["gen_ids"]]
            load_names = [load_ids_to_names[load_id] for load_id in targets["load_ids"]]

            prods[gen_names] = overload_injections(prods[gen_names], p=p)
            loads[load_names] = overload_injections(loads[load_names], p=p)

    if verbose or True:
        pprint(
            "        - Loads:", ["{:.2f}".format(l) for l in loads.mean(axis=0).values]
        )
        pprint(
            "        - Prods:", ["{:.2f}".format(g) for g in prods.mean(axis=0).values]
        )

    return prods, loads, p
