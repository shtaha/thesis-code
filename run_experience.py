import os

from experience import ExperienceCollector
from lib.agents import make_test_agent
from lib.constants import Constants as Const
from lib.data_utils import make_dir
from lib.dc_opf import load_case, CaseParameters
from lib.run_utils import create_logger
from lib.visualizer import Visualizer, pprint

visualizer = Visualizer()

save_dir = make_dir(os.path.join(Const.EXPERIENCE_DIR, "data"))

env_dc = True
verbose = False

kwargs = dict()

# for case_name in ["rte_case5_example", "l2rpn_2019", "l2rpn_wcci_2020"]:
for case_name in ["rte_case5_example"]:
    env_pf = "dc" if env_dc else "ac"
    case_save_dir = make_dir(os.path.join(save_dir, f"{case_name}-{env_pf}"))

    create_logger(logger_name=f"{case_name}-{env_pf}", save_dir=case_save_dir)

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
        # "do-nothing-agent",
        "agent-mip",
        # "agent-multistep-mip",
    ]:
        if case_name == "rte_case5_example":
            kwargs["obj_lambda_action"] = 0.004
        elif case_name == "l2rpn_2019":
            kwargs["obj_lambda_action"] = 0.07
        else:
            kwargs["obj_lambda_action"] = 0.05

        """
            Initialize agent.
        """
        agent = make_test_agent(agent_name, case, action_set, **kwargs)

        """
            Collect experience.
        """
        collector = ExperienceCollector(save_dir=case_save_dir)
        collector.collect(env, agent, n_chronics=2, n_steps=-1)
