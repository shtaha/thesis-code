import os

import grid2op
from grid2op.Environment import Environment

from experiments import ExperimentDCOPFTiming
from lib.action_space import ActionSpaceGenerator
from lib.agents import make_agent
from lib.constants import Constants as Const
from lib.data_utils import make_dir
from lib.dc_opf import load_case, CaseParameters
from lib.run_utils import create_logger
from lib.visualizer import Visualizer

visualizer = Visualizer()

save_dir = make_dir(os.path.join(Const.RESULTS_DIR, "timing"))

env_dc = True
verbose = False

experiment_timing = ExperimentDCOPFTiming()
kwargs = dict()

for case_name in ["rte_case5_example", "l2rpn_2019", "l2rpn_wcci_2020"]:
    if case_name == "l2rpn_wcci_2020":
        n_timings = 10
    elif case_name == "l2rpn_2019":
        n_timings = 10
    else:
        n_timings = 10

    env_pf = "dc" if env_dc else "ac"
    case_save_dir = make_dir(os.path.join(save_dir, f"{case_name}-{env_pf}"))

    create_logger(logger_name=f"{case_name}-{env_pf}", save_dir=case_save_dir)

    """
        Initialize environment parameters.    
    """
    parameters = CaseParameters(case_name=case_name, env_dc=env_dc)

    """
        Initialize environment and case.
    """
    env: Environment = grid2op.make_from_dataset_path(
        dataset_path=os.path.join(os.path.expanduser("~"), "data_grid2op", case_name),
        backend=grid2op.Backend.PandaPowerBackend(),
        action_class=grid2op.Action.TopologyAction,
        observation_class=grid2op.Observation.CompleteObservation,
        reward_class=grid2op.Reward.L2RPNReward,
        param=parameters,
    )
    case = load_case(case_name, env=env)

    """
        Initialize action set.
    """
    action_generator = ActionSpaceGenerator(env)
    action_set = action_generator.get_topology_action_set(
        save_dir=case_save_dir, verbose=verbose
    )

    for agent_name in [
        "mip_agent",
        "multi_mip_agent",
    ]:
        """
            Initialize agent.
        """
        agent = make_agent(
            agent_name,
            case,
            action_set,
            n_max_line_status_changed=case.env.parameters.MAX_LINE_STATUS_CHANGED,
            n_max_sub_changed=case.env.parameters.MAX_SUB_CHANGED,
            **kwargs,
        )

        """
            Experiments.
        """
        experiment_timing.compare_by_solver_and_parts(
            case=case,
            agent=agent,
            save_dir=case_save_dir,
            solver_names=("glpk", "gurobi", "mosek"),
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
            verbose=verbose,
        )

        experiment_timing.compare_by_constraint_activations(
            case=case,
            agent=agent,
            save_dir=case_save_dir,
            constraint_activations=[
                ({}, "Default"),
                (
                    {"con_allow_onesided_disconnection": False},
                    "One-sided disconnection",
                ),
                ({"con_allow_onesided_reconnection": False}, "One-sided reconnection",),
                ({"con_symmetry": False}, "Symmetry"),
                ({"con_requirement_balance": False}, "RI"),
                ({"con_requirement_at_least_two": False}, "RII"),
                ({"con_switching_limits": False}, "Switching limits"),
                (
                    {"con_cooldown": False, "con_maintenance": False},
                    "Cooldown and Maintenance",
                ),
                ({"con_unitary_action": False}, "Unitary action"),
            ],
            n_timings=n_timings,
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
                    "Standard DC-OPF",
                ),
                (
                    {"obj_reward_quad": True, "obj_reward_max": False},
                    "Quadratic margins",
                ),
                ({"obj_reward_lin": True, "obj_reward_max": False,}, "L2RPN Reward",),
                (
                    {"obj_lin_gen_penalty": False, "obj_quad_gen_penalty": True,},
                    "Quadratic gen. penalty",
                ),
            ],
            n_timings=n_timings,
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
            verbose=verbose,
        )

        experiment_timing.aggregate_by_agent(agent, save_dir=case_save_dir)
