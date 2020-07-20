import os

import grid2op
from grid2op.Environment import Environment

from experiments import ExperimentMIPControl
from lib.agents import AgentMIP
from lib.constants import Constants as Const
from lib.data_utils import create_results_dir
from lib.dc_opf import load_case
from lib.visualizer import (
    describe_environment,
    print_parameters,
)

save_dir = create_results_dir(Const.RESULTS_DIR)
experiment_mip = ExperimentMIPControl()

env_dc = False
for case_name in ["rte_case5_example", "l2rpn_2019", "l2rpn_wcci_2020"]:
    """
        Initialize environment.
    """
    env: Environment = grid2op.make_from_dataset_path(
        dataset_path=os.path.join(os.path.expanduser("~"), "data_grid2op", case_name),
        backend=grid2op.Backend.PandaPowerBackend(),
        action_class=grid2op.Action.TopologyAction,
        observation_class=grid2op.Observation.CompleteObservation,
        reward_class=grid2op.Reward.L2RPNReward,
    )
    env.seed(0)

    if env_dc:
        env.env_dc = env_dc
        env.parameters.ENV_DC = env_dc
        env.parameters.FORECAST_DC = env_dc

    describe_environment(env)
    print_parameters(env)

    """
        Initialize agent.
    """
    agent = AgentMIP(
        case=load_case(case_name, env=env),
        n_max_line_status_changed=env.parameters.MAX_LINE_STATUS_CHANGED,
        n_max_sub_changed=env.parameters.MAX_SUB_CHANGED,
    )

    """
        Experiments.
    """
    experiment_mip.evaluate_performance(env=env, agent=agent, n_steps=100, verbose=False)
    break
