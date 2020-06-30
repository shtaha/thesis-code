import datetime
import os

import numpy as np
import pandas as pd


def make_dir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

    return directory


def create_results_dir(results_dir_path):
    results_dir = make_dir(
        os.path.join(
            results_dir_path,
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_model"),
        )
    )
    return results_dir


def indices_to_hot(
    hot_indices: np.ndarray, length: np.int, dtype=np.bool
) -> np.ndarray:
    """
    Only works for 1D-vector.
    """
    vector = np.zeros((length,), dtype=dtype)
    vector[hot_indices] = np.True_
    return vector


def hot_to_indices(bool_array: np.ndarray) -> np.ndarray:
    """
    Only works for 1D-vector.
    """
    index_array = np.flatnonzero(bool_array)
    return index_array


def update_backend(env, verbose=False):
    """
    Update backend grid with missing data.
    """
    grid = env.backend._grid.deepcopy()

    if env.name == "rte_case5_example":
        # Loads
        grid.load["controllable"] = False

        # Generators
        grid.gen["controllable"] = True

        bus_names = []
        for bus_id, bus_name in zip(grid.bus.index, grid.bus["name"]):
            sub_id = bus_name.split("_")[-1]
            bus_names.append(f"bus-{bus_id}-{sub_id}")

        grid.bus["name"] = bus_names
        grid.line["name"] = env.name_line
        grid.gen["name"] = env.name_gen
        grid.load["name"] = env.name_load

        # Update thermal limits with environment thermal limits
        grid.line["max_i_ka"] = env.get_thermal_limit()
    elif env.name == "l2rpn_2019":
        bus_names = [
            f"bus-{bus_id}-{sub_id}"
            for bus_id, sub_id in zip(grid.bus.index, grid.bus["name"])
        ]

        grid.bus["name"] = bus_names
        grid.line["name"] = env.name_line
        grid.gen["name"] = env.name_gen
        grid.load["name"] = env.name_load

        # Update thermal limits with environment thermal limits
        grid.line["max_i_ka"] = env.get_thermal_limit()
    elif env.name == "l2rpn_wcci_2020":
        n_bus = len(grid.bus.index)
        bus_to_sub_ids = np.concatenate((np.arange(0, n_bus), np.arange(0, n_bus)))
        bus_names = [
            f"bus-{bus_id}-{sub_id}"
            for bus_id, sub_id in zip(grid.bus.index, bus_to_sub_ids)
        ]

        grid.bus["name"] = bus_names  # Substations

        # Update thermal limits with environment thermal limits
        grid.line["max_i_ka"] = env.get_thermal_limit()[0:55] / 1000.0

        # TODO: HACK +
        # Update transformer parameters
        # trafo_params = {
        #     "id": {"0": 0, "1": 1, "2": 2, "3": 3,},
        #     "b_pu": {
        #         "0": 2666.6666666667,
        #         "1": 2590.6735751295,
        #         "2": 3731.3432835821,
        #         "3": 2702.7027027027,
        #     },
        #     "max_p_pu": {"0": 9900.0, "1": 9900.0, "2": 9900.0, "3": 9900.0},
        # }

        trafo_params = {
            "id": {"0": 0, "1": 1, "2": 2, "3": 3,},
            "b_pu": {
                "0": 2852.04991087,
                "1": 2698.61830743,
                "2": 3788.16577013,
                "3": 2890.59112589,
            },
            "max_p_pu": {"0": 9900.0, "1": 9900.0, "2": 9900.0, "3": 9900.0},
        }

        trafo_params = pd.DataFrame.from_dict(trafo_params)
        trafo_params.set_index("id", inplace=True)
        grid.trafo["b_pu"] = trafo_params["b_pu"]
        grid.trafo["max_p_pu"] = trafo_params["max_p_pu"]
        grid.trafo["in_service"] = False

    # Environment and backend inconsistency
    grid.gen["min_p_mw"] = env.gen_pmin
    grid.gen["max_p_mw"] = env.gen_pmax
    grid.gen["type"] = env.gen_type
    grid.gen["gen_redispatchable"] = env.gen_redispatchable
    grid.gen["gen_max_ramp_up"] = env.gen_max_ramp_up
    grid.gen["gen_max_ramp_down"] = env.gen_max_ramp_down
    grid.gen["gen_min_uptime"] = env.gen_min_uptime
    grid.gen["gen_min_downtime"] = env.gen_min_downtime

    if verbose:
        print(env.name.upper())
        print("bus\n" + grid.bus.to_string())
        print("gen\n" + grid.gen.to_string())
        print("load\n" + grid.load.to_string())
        print("line\n" + grid.line.to_string())

    return grid


def parse_gurobi_log(log):
    gap = 0.0
    for line in log.split("\n")[-5:]:
        if "Best objective" in line:
            gap = float(line.strip().split()[-1].replace("%", ""))

    return {"gap": gap}


def bus_names_to_sub_ids(bus_names):
    sub_ids = [int(bus_name.split("-")[-1]) for bus_name in bus_names]
    return sub_ids
