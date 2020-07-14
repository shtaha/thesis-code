from experiments.experiment_timing_dc_opf import ExperimentDCOPFTiming
from lib.constants import Constants as Const
from lib.data_utils import create_results_dir
from lib.dc_opf import load_case

save_dir = create_results_dir(Const.RESULTS_DIR)
n_measurements = 200
n_bins = 25
time_measurements = dict()

experiment = ExperimentDCOPFTiming()

for case_name in ["rte_case5", "l2rpn2019", "l2rpn2020"]:
    case = load_case(case_name)
    experiment.compare_by_solver(
        case=case,
        save_dir=save_dir,
        solver_names=("gurobi",),
        tol=0.01,
        n_bins=n_bins,
        n_measurements=n_measurements,
    )

    experiment.compare_by_tolerance(
        case=case,
        save_dir=save_dir,
        tols=(0.01, 0.005, 0.001),
        solver_name="gurobi",
        n_bins=n_bins,
        n_measurements=n_measurements,
    )

    experiment.compare_by_switching_limits(
        case=case,
        save_dir=save_dir,
        switch_limits=[(1, 1), (2, 1), (3, 1), (1, 2), (1, 3), (2, 2)],
        tol=0.01,
        solver_name="gurobi",
        n_bins=n_bins,
        n_measurements=n_measurements,
    )

    experiment.compare_by_constraint_activations(
        case=case,
        save_dir=save_dir,
        constraint_activations=[
            (True, True, True, True),
            (False, True, True, True),
            (True, False, True, True),
            (True, True, False, True),
            (True, True, True, False),
        ],
        tol=0.01,
        solver_name="gurobi",
        n_bins=n_bins,
        n_measurements=n_measurements,
    )

    experiment.compare_by_objective(
        case=case,
        save_dir=save_dir,
        objectives=[
            (True, True, False),
            (True, False, False),
            (False, True, False),
            (False, False, True),
        ],
        tol=0.01,
        solver_name="gurobi",
        n_bins=n_bins,
        n_measurements=n_measurements,
    )
    break
