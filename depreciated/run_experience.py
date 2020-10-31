import os

import numpy as np

from experience import ExperienceCollector
from lib.agents import make_test_agent
from lib.constants import Constants as Const
from lib.data_utils import make_dir, env_pf
from lib.dc_opf import load_case, CaseParameters
from lib.run_utils import create_logger
from lib.visualizer import Visualizer

visualizer = Visualizer()

save_dir = make_dir(os.path.join(Const.EXPERIENCE_DIR, "data-aug"))

env_dc = True
verbose = False

kwargs = dict()

np.random.seed(0)
for case_name in ["rte_case5_example", "l2rpn_2019", "l2rpn_wcci_2020"]:
    if "l2rpn_2019" not in case_name:
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

    for agent_name in [
        "agent-mip",
        "agent-multistep-mip",
    ]:
        np.random.seed(1)
        if "rte_case5" in case_name:
            kwargs["obj_lambda_action"] = 0.004
            do_chronics = [13, 14, 15, 16, 17, 16, 18, 19]
        elif "l2rpn_2019" in case_name:
            kwargs["obj_lambda_action"] = 0.07
            do_chronics = [0, 10, 100, 196, 200, 201, 206, 226, 259, 375, 384, 491]
            do_chronics.extend(np.random.randint(0, 1000, 500).tolist())
        else:
            kwargs["obj_lambda_action"] = 0.05
            do_chronics = [*np.arange(0, 2880, 240), *(np.arange(0, 2880, 240) + 1)]

        do_chronics = np.unique(do_chronics)

        """
            Initialize agent.
        """
        agent = make_test_agent(agent_name, case, action_set, **kwargs)

        """
            Collect experience.
        """
        collector = ExperienceCollector(save_dir=case_save_dir)
        collector.collect(
            env, agent, do_chronics=do_chronics, n_chronics=100, n_steps=-1
        )
