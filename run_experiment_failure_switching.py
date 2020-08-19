import os

import grid2op
from grid2op.Environment import Environment

from experiments import ExperimentFailure, ExperimentSwitching
from lib.action_space import ActionSpaceGenerator
from lib.agents import make_agent
from lib.constants import Constants as Const
from lib.data_utils import make_dir
from lib.dc_opf import load_case, CaseParameters
from lib.run_utils import create_logger
from lib.visualizer import Visualizer

visualizer = Visualizer()

save_dir = make_dir(os.path.join(Const.RESULTS_DIR, "failure-switching"))

do_experiment_failure = True
do_experiment_switching = True
verbose = False

experiment_failure = ExperimentFailure()
experiment_switching = ExperimentSwitching()

kwargs = dict(horizon=2, obj_lambda_action=0.0)

# for case_name in ["rte_case5_example", "l2rpn_2019", "l2rpn_wcci_2020"]:
for case_name in ["rte_case5_example"]:
    for env_dc in [False, True]:
        if not env_dc:
            continue

        env_pf = "dc" if env_dc else "ac"
        case_save_dir = make_dir(os.path.join(save_dir, f"{case_name}-{env_pf}"))

        create_logger(logger_name=f"{case_name}-{env_pf}", save_dir=save_dir)

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
        case = load_case(case_name, env=env)

        """
            Initialize action set.
        """
        action_generator = ActionSpaceGenerator(env)
        action_set = action_generator.get_topology_action_set(
            save_dir=save_dir, verbose=verbose
        )

        for agent_name in [
            "do_nothing_agent",
            "mip_agent",
            # "mixed_multi_agent",
            # "multi_mip_agent",
            # "mixed_agent",
            # "augmented_agent",
            # "greedy_agent",
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

            if do_experiment_failure and do_experiment_switching:
                experiment_failure.analyze_failures(
                    case=case, agent=agent, save_dir=case_save_dir, verbose=verbose,
                )

        if do_experiment_failure:
            experiment_failure.compare_agents(case, save_dir=case_save_dir)

        if do_experiment_switching:
            experiment_switching.compare_agents(case, save_dir=case_save_dir)
