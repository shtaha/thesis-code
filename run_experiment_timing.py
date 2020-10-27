import os

from experiments import ExperimentDCOPFTiming
from lib.agents import make_test_agent
from lib.constants import Constants as Const
from lib.data_utils import make_dir, env_pf
from lib.dc_opf import load_case, CaseParameters
from lib.run_utils import create_logger
from lib.visualizer import Visualizer

visualizer = Visualizer()

save_dir = make_dir(os.path.join(Const.RESULTS_DIR, "final"))
save_dir = make_dir(os.path.join(save_dir, "timing-rew"))

env_dc = True
verbose = False

experiment_timing = ExperimentDCOPFTiming()
kwargs = dict(time_limit=40)

for case_name in ["rte_case5_example", "l2rpn_2019", "l2rpn_wcci_2020"]:
    if case_name == "l2rpn_wcci_2020":
        n_timings = 50
        continue
    elif case_name == "l2rpn_2019":
        n_timings = 100
    else:
        n_timings = 200
        continue

    case_save_dir = make_dir(os.path.join(save_dir, f"{case_name}-{env_pf(env_dc)}"))
    create_logger(logger_name=f"logger", save_dir=case_save_dir)

    """
        Initialize environment.
    """
    parameters = CaseParameters(case_name=case_name, env_dc=env_dc)
    case = load_case(case_name, env_parameters=parameters)

    for agent_name in [
        "agent-mip",
        "agent-multistep-mip",
    ]:
        x_right = None
        """
            Initialize agent.
        """
        agent = make_test_agent(agent_name, case, **kwargs)

        """
            Experiments.
        """
        experiment_timing.compare_by_solver_and_parts(
            case=case,
            agent=agent,
            save_dir=case_save_dir,
            solver_names=("glpk", "gurobi", "mosek"),
            x_right=x_right,
            n_timings=n_timings,
            verbose=verbose,
        )

        experiment_timing.compare_by_tolerance(
            case=case,
            agent=agent,
            save_dir=case_save_dir,
            tols=(10 ** (-i) for i in range(2, 6)),
            n_timings=n_timings,
            verbose=verbose,
        )

        experiment_timing.compare_by_delta_max(
            case=case,
            agent=agent,
            save_dir=case_save_dir,
            deltas=(0.45, 0.5, 0.6, 1.0),
            n_timings=n_timings,
            verbose=verbose,
        )

        experiment_timing.compare_by_switching_limits(
            case=case,
            agent=agent,
            save_dir=case_save_dir,
            switch_limits=[(1, 0), (0, 1), (1, 1), (2, 1)],
            n_timings=n_timings,
            x_right=x_right,
            verbose=verbose,
        )

        experiment_timing.compare_by_constraint_activations(
            case=case,
            agent=agent,
            save_dir=case_save_dir,
            constraint_activations=[
                ({}, "Default"),
                ({"con_symmetry": False}, "Symmetry"),
                ({"con_switching_limits": False}, "Switching limits"),
                ({"con_unitary_action": True}, "Unitary action"),
            ],
            n_timings=n_timings,
            x_right=x_right,
            verbose=verbose,
        )

        experiment_timing.compare_by_objective(
            case=case,
            agent=agent,
            save_dir=case_save_dir,
            objectives=[
                ({}, "Default"),
                (
                    {
                        "obj_gen_cost": True,
                        "obj_reward_max": False,
                        "obj_lin_gen_penalty": False,
                    },
                    "Standard",
                ),
                (
                    {"obj_reward_quad": True, "obj_reward_max": False},
                    "L2RPN quad.",
                ),
                ({"obj_reward_lin": True, "obj_reward_max": False}, "L2RPN lin.",),
                (
                    {"obj_lin_gen_penalty": False, "obj_quad_gen_penalty": True},
                    "Penalty quad.",
                ),
            ],
            n_timings=n_timings,
            x_right=x_right,
            verbose=verbose,
        )

        experiment_timing.compare_by_warmstart(
            case=case,
            agent=agent,
            save_dir=case_save_dir,
            n_timings=n_timings,
            verbose=verbose,
        )

        experiment_timing.compare_by_lambda_gen(
            case=case,
            agent=agent,
            save_dir=case_save_dir,
            lambdas=(10 ** i for i in range(-1, 3)),
            n_timings=n_timings,
            x_right=x_right,
            verbose=verbose,
        )

        experiment_timing.aggregate_by_agent(agent, save_dir=case_save_dir)
