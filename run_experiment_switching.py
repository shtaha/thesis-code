import os

from experiments import ExperimentSwitching
from lib.agents import make_test_agent
from lib.constants import Constants as Const
from lib.data_utils import make_dir, env_pf
from lib.dc_opf import load_case, CaseParameters
from lib.run_utils import create_logger
from lib.visualizer import Visualizer

visualizer = Visualizer()

save_dir = make_dir(os.path.join(Const.RESULTS_DIR, "switching"))

env_dc = True
verbose = False

experiment_switching = ExperimentSwitching()

kwargs = dict()

for case_name in ["rte_case5_example", "l2rpn_2019", "l2rpn_wcci_2020"]:
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
        "do-nothing-agent",
    ]:
        """
            Initialize agent.
        """
        agent = make_test_agent(agent_name, case, action_set, **kwargs)

        """
            Experiments.
        """
        experiment_switching.analyse(
            case=case, agent=agent, save_dir=case_save_dir, verbose=verbose,
        )

    experiment_switching.compare_agents(case, save_dir=case_save_dir)
