import os

import grid2op
from grid2op.Environment import Environment

from experiments import ExperimentDCOPFTiming, ExperimentMIPControl
from lib.agents import AgentMIPTest, AgentDoNothing
from lib.action_space import ActionSpaceGenerator
from lib.constants import Constants as Const
from lib.data_utils import create_results_dir, make_dir
from lib.dc_opf import load_case, CaseParameters

save_dir = create_results_dir(Const.RESULTS_DIR)

do_experiment_timing = True
do_experiment_mip_control = False

n_timings = 5
n_steps = 100

verbose = False
env_dc = True
lambd = 50.0

# for case_name in ["rte_case5_example", "l2rpn_2019", "l2rpn_wcci_2020"]:
for case_name in ["rte_case5_example"]:
    case_save_dir = make_dir(os.path.join(save_dir, case_name))

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
    case = load_case(case_name, env=env, verbose=verbose)

    """
        Initialize action set.
    """
    action_generator = ActionSpaceGenerator(env)
    action_set = action_generator.get_topology_action_set(verbose=verbose)

    """
        Initialize agent.
    """
    agent = AgentMIPTest(
        case=case,
        action_set=action_set,
        n_max_line_status_changed=env.parameters.MAX_LINE_STATUS_CHANGED,
        n_max_sub_changed=env.parameters.MAX_SUB_CHANGED,
    )
    agent_do_nothing = AgentDoNothing(case=case, action_set=action_set)

    """
        Experiments.
    """
    if do_experiment_timing:
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

        experiment_timing.compare_by_switching_limits(
            case=case,
            agent=agent,
            save_dir=case_save_dir,
            switch_limits=[(1, 0), (0, 1), (1, 1), (2, 1), (3, 1), (1, 3), (2, 2)],
            n_timings=n_timings,
            verbose=verbose,
        )

        experiment_timing.compare_by_constraint_activations(
            case=case,
            agent=agent,
            save_dir=case_save_dir,
            constraint_activations=[
                (False, False, True, True, True, True, True),
                (True, False, True, True, True, True, True),
                (False, True, True, True, True, True, True),
                (False, False, False, True, True, True, True),
                (False, False, True, False, True, True, True),
                (False, False, True, True, False, True, True),
                (False, False, True, True, True, False, True),
                (False, False, True, True, True, True, False),
            ],  # Onesided-Implicit-Symmetry-Balance-Switching-Cooldown-Unitary
            n_timings=n_timings,
            verbose=verbose,
        )

        experiment_timing.compare_by_objective(
            case=case,
            agent=agent,
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

    if do_experiment_mip_control:
        experiment_mip = ExperimentMIPControl()

        experiment_mip.evaluate_performance(
            case=case,
            agent=agent,
            save_dir=case_save_dir,
            n_steps=n_steps,
            lambd=lambd,
            verbose=verbose,
        )

        experiment_mip.evaluate_performance(
            case=case,
            agent=agent_do_nothing,
            save_dir=case_save_dir,
            n_steps=n_steps,
            verbose=verbose,
        )
