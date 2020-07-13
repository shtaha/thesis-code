import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from timeit import default_timer as timer
from lib.dc_opf import load_case, GridDCOPF, TopologyOptimizationDCOPF
from lib.data_utils import create_results_dir
from lib.constants import Constants as Const


save_dir = create_results_dir(Const.RESULTS_DIR)
n_measurements = 200
n_bins = 20
time_measurements = dict()

bounds = {
    "case3": [-0.2, 0.2],
    "case4": [-0.2, 0.2],
    "case6": [-0.5, 0.5],
    "l2rpn2019": [-1.0, 1.0],
    "l2rpn2020": [-1.0, 1.0],
}

# for case_name in ["case3", "case4", "case6", "l2rpn2019", "l2rpn2020"]:
for case_name in ["l2rpn2020"]:
    time_measurements[case_name] = list()

    for i in range(n_measurements + 1):
        print(f"Measurement: {i}")
        start_total = timer()
        start_load = timer()
        case = load_case(case_name)
        time_load = timer() - start_load

        start_grid = timer()
        grid = GridDCOPF(
            case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p
        )
        time_grid = timer() - start_grid

        # TODO: obs = env.reset()
        # load_p = obs.load_p
        load_p = case.grid_backend.load["p_mw"].values
        load_p = load_p + np.random.uniform(
            bounds[case_name][0], bounds[case_name][1], load_p.shape
        )
        grid.load["p_pu"] = load_p

        model = TopologyOptimizationDCOPF(
            f"{case.name} DC OPF Topology Optimization",
            grid=grid,
            grid_backend=case.grid_backend,
            base_unit_p=case.base_unit_p,
            base_unit_v=case.base_unit_v,
        )

        start_build = timer()
        model.build_model()
        time_build = timer() - start_build

        start_solve = timer()
        model.solve(tol=0.01, verbose=False)
        time_solve = timer() - start_solve

        time_total = timer() - start_total
        time_measurement = {
            "total": time_total,
            "grid": time_grid,
            "build": time_build,
            "solve": time_solve,
        }
        if i > 0:
            time_measurements[case_name].append(time_measurement)

    time_measurements[case_name] = pd.DataFrame(time_measurements[case_name])
    case_time = time_measurements[case_name]

    print(f"\n{case_name}\n")
    print(case_time.to_string())

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 9))
    case_time.hist(ax=ax[0, 0], column="total", bins=n_bins)
    ax[0, 0].set_xlabel("Time [s]")
    case_time.hist(ax=ax[0, 1], column="grid", bins=n_bins)
    ax[0, 1].set_xlabel("Time [s]")
    case_time.hist(ax=ax[1, 0], column="build", bins=n_bins)
    ax[1, 0].set_xlabel("Time [s]")
    case_time.hist(ax=ax[1, 1], column="solve", bins=n_bins)
    ax[1, 1].set_xlabel("Time [s]")
    fig.show()

    fig.savefig(os.path.join(save_dir, case_name + ".png"))
