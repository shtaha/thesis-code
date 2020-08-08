import os
import sys

import grid2op
from grid2op.Environment import Environment

from experiments import ExperimentDCOPFTiming, ExperimentMIPControl
from lib.action_space import ActionSpaceGenerator
from lib.agents import make_agent
from lib.constants import Constants as Const
from lib.data_utils import create_results_dir, make_dir
from lib.dc_opf import load_case, CaseParameters
from lib.run_utils import Logger
from lib.visualizer import Visualizer

visualizer = Visualizer()

save_dir = create_results_dir(Const.RESULTS_DIR)
sys.stdout = Logger(save_dir=save_dir)

do_experiment_timing = False
do_experiment_control = True
verbose = False

kwargs = dict(forecasts=False)

# for case_name in ["rte_case5_example", "l2rpn_2019", "l2rpn_wcci_2020"]:
for case_name in ["l2rpn_2019"]:
    if case_name == "l2rpn_wcci_2020":
        n_timings = 50
        n_steps = 1000
    else:
        n_timings = 100
        n_steps = 1000

    for env_dc in [True, False]:
        if not env_dc:
            continue

        env_pf = "dc" if env_dc else "ac"
        case_save_dir = make_dir(os.path.join(save_dir, f"{case_name}-{env_pf}"))

        for agent_name in [
            "mip_agent",
            "do_nothing_agent",
            "mixed_agent",
            "augmented_agent",
            "greedy_agent",
        ]:
            if agent_name == "greedy_agent" and case_name != "rte_case5_example":
                continue
            if agent_name != "mip_agent":
                break

            """
                Initialize environment parameters.    
            """
            parameters = CaseParameters(case_name=case_name, env_dc=env_dc)

            """
                Initialize environment and case.
            """
            env: Environment = grid2op.make_from_dataset_path(
                dataset_path=os.path.join(
                    os.path.expanduser("~"), "data_grid2op", case_name
                ),
                backend=grid2op.Backend.PandaPowerBackend(),
                action_class=grid2op.Action.TopologyAction,
                observation_class=grid2op.Observation.CompleteObservation,
                reward_class=grid2op.Reward.L2RPNReward,
                param=parameters,
            )
            case = load_case(case_name, env=env, verbose=True)

            """
                Initialize action set.
            """
            action_generator = ActionSpaceGenerator(env)
            action_set = action_generator.get_topology_action_set(verbose=verbose)

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
            if do_experiment_timing and agent_name != "do_nothing_agent":
                experiment_timing = ExperimentDCOPFTiming()

                experiment_timing.compare_by_solver_and_parts(
                    case=case,
                    agent=agent,
                    save_dir=case_save_dir,
                    solver_names=("gurobi",),
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
                    deltas=(0.5, 0.6, 1.0, 1.57),
                    n_timings=n_timings,
                    verbose=verbose,
                )

                experiment_timing.compare_by_switching_limits(
                    case=case,
                    agent=agent,
                    save_dir=case_save_dir,
                    switch_limits=[(1, 0), (0, 1), (1, 1), (2, 1), (3, 1)],
                    n_timings=n_timings,
                    verbose=verbose,
                )

                experiment_timing.compare_by_constraint_activations(
                    case=case,
                    agent=agent,
                    save_dir=case_save_dir,
                    constraint_activations=[
                        (True, False, False, True, True, True, True, True),
                        (True, True, False, True, True, True, True, True),
                        (True, False, True, True, True, True, True, True),
                        (True, False, False, False, True, True, True, True),
                        (True, False, False, True, False, True, True, True),
                        (True, False, False, True, True, False, True, True),
                        (True, False, False, True, True, True, False, True),
                        (True, False, False, True, True, True, True, False),
                    ],  # Onesided-Implicit-Reconnection-Symmetry-Balance-Switching-Cooldown-Unitary
                    n_timings=n_timings,
                    verbose=verbose,
                )

                experiment_timing.compare_by_objective(
                    case=case,
                    agent=agent,
                    save_dir=case_save_dir,
                    objectives=[
                        (True, False, False, False, False),  # Standard OPF objective
                        (
                            False,
                            True,
                            False,
                            False,
                            False,
                        ),  # Linear objective for margins
                        (
                            False,
                            False,
                            True,
                            False,
                            False,
                        ),  # Quadratic objective for margins
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

                experiment_timing.compare_by_lambda(
                    case=case,
                    agent=agent,
                    save_dir=case_save_dir,
                    lambdas=(10 ** i for i in range(-1, 4)),
                    n_timings=n_timings,
                    verbose=verbose,
                )

            if do_experiment_control:
                experiment_control = ExperimentMIPControl()

                experiment_control.evaluate_performance(
                    case=case,
                    agent=agent,
                    save_dir=case_save_dir,
                    n_steps=n_steps,
                    verbose=verbose,
                )
    break
