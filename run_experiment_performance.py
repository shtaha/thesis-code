import os

import numpy as np

from experiments import ExperimentPerformance
from lib.agents import make_test_agent
from lib.constants import Constants as Const
from lib.data_utils import make_dir, env_pf
from lib.dc_opf import load_case, CaseParameters
from lib.run_utils import create_logger
from lib.visualizer import Visualizer

visualizer = Visualizer()

save_dir = make_dir(os.path.join(Const.RESULTS_DIR, "performance-sn"))

env_dc = True
verbose = False

kwargs = dict()

for case_name in ["rte_case5_example", "l2rpn_2019", "l2rpn_wcci_2020"]:
    if case_name != "l2rpn_2019":
        continue

    case_save_dir = make_dir(os.path.join(save_dir, f"{case_name}-{env_pf(env_dc)}"))
    create_logger(logger_name=f"logger", save_dir=case_save_dir)

    """
        Initialize environment.
    """
    parameters = CaseParameters(case_name=case_name, env_dc=env_dc)
    case = load_case(case_name, env_parameters=parameters)
    env = case.env
    action_set = case.generate_unitary_action_set(
        case, case_save_dir=case_save_dir, verbose=verbose
    )

    experiment_performance = ExperimentPerformance(save_dir=case_save_dir)
    for agent_name in [
        "do-nothing-agent",
        "agent-mip",
        # "agent-multistep-mip",
    ]:
        np.random.seed(0)
        if case_name == "rte_case5_example":
            kwargs["obj_lambda_action"] = 0.004
            do_chronics = [13, 14, 15, 16, 17, 16, 18, 19]
        elif case_name == "l2rpn_2019":
            kwargs["obj_lambda_action"] = 0.07
            do_chronics = [0, 10, 100, 196, 200, 201, 206, 226, 259, 375, 384, 491]
            do_chronics.extend(np.random.randint(0, 1000, 100).tolist())
        else:
            kwargs["obj_lambda_action"] = 0.05
            do_chronics = [*np.arange(0, 2880, 240), *(np.arange(0, 2880, 240) + 1)]

        """
            Initialize agent.
        """
        agent = make_test_agent(agent_name, case, action_set, **kwargs)

        """
            Experiments.
        """
        experiment_performance.analyse(
            case=case,
            agent=agent,
            do_chronics=do_chronics,
            n_chronics=-1,
            n_steps=250,
            verbose=verbose,
        )

    experiment_performance.compare_agents(case, save_dir=case_save_dir)
