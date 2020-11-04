import os

from experiments import ExperimentDCOPFTiming, ExperimentMIPControl
from lib.constants import Constants as Const
from lib.data_utils import create_results_dir, make_dir
from lib.dc_opf import load_case

save_dir = create_results_dir(Const.RESULTS_DIR)
n_measurements = 100
n_bins = 25

experiment_timing = ExperimentDCOPFTiming()

for case_name in ["rte_case5", "l2rpn2019", "l2rpn2020"]:
    case = load_case(case_name)
    case_save_dir = make_dir(os.path.join(save_dir, case_name))

    experiment_timing.compare_by_solver(
        case=case,
        save_dir=case_save_dir,
        solver_names=("gurobi",),
        n_bins=n_bins,
        n_measurements=n_measurements,
        verbose=True,
    )

    experiment_timing.compare_by_tolerance(
        case=case,
        save_dir=case_save_dir,
        tols=(10 ** (-i) for i in range(2, 6)),
        n_bins=n_bins,
        n_measurements=n_measurements,
    )

    experiment_timing.compare_by_switching_limits(
        case=case,
        save_dir=case_save_dir,
        switch_limits=[(1, 0), (0, 1), (1, 1), (2, 1), (3, 1), (1, 2), (1, 3), (2, 2)],
        n_bins=n_bins,
        n_measurements=n_measurements,
    )

    experiment_timing.compare_by_constraint_activations(
        case=case,
        save_dir=case_save_dir,
        constraint_activations=[
            (False, False, True, True, True, True),
            (True, False, True, True, True, True),
            (False, True, True, True, True, True),
            (False, False, False, True, True, True),
            (False, False, True, False, True, True),
            (False, False, True, True, False, True),
            (False, False, True, True, True, False),
        ],  # Onesided-Implicit-Symmetry-Switching-Cooldown-Unitary
        n_bins=n_bins,
        n_measurements=n_measurements,
    )

    experiment_timing.compare_by_objective(
        case=case,
        save_dir=case_save_dir,
        objectives=[
            (True, False, False, False, False),  # Standard OPF objective
            (False, True, False, False, False),  # Linear objective for margins
            (False, False, True, False, False),  # Quadratic objective for margins
            (
                False,
                True,
                False,
                True,
                False,
            ),  # Linear objective for margins with linear generator penalty
            (
                False,
                True,
                False,
                False,
                True,
            ),  # Linear objective for margins with quadratic generator penalty
            (
                False,
                False,
                True,
                True,
                False,
            ),  # Quadratic objective for margins with linear generator penalty
            (
                False,
                False,
                True,
                False,
                True,
            ),  # Quadratic objective for margins with quadratic generator penalty
        ],
        n_bins=n_bins,
        n_measurements=n_measurements,
    )

    experiment_timing.compare_by_warmstart(
        case=case, save_dir=case_save_dir, n_bins=n_bins, n_measurements=n_measurements,
    )

    experiment_timing.compare_by_lambda(
        case=case,
        save_dir=case_save_dir,
        lambdas=(10 ** i for i in range(-1, 4)),
        n_bins=n_bins,
        n_measurements=n_measurements,
    )
