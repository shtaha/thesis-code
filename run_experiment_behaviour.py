import os

from experiments import ExperimentBehaviour
from lib.agents import make_test_agent
from lib.constants import Constants as Const
from lib.data_utils import make_dir, env_pf
from lib.dc_opf import load_case, CaseParameters
from lib.run_utils import create_logger
from lib.visualizer import Visualizer

visualizer = Visualizer()

save_dir = make_dir(os.path.join(Const.RESULTS_DIR, "behaviour"))

env_dc = True
verbose = False

experiment_behaviour = ExperimentBehaviour()
kwargs = dict()

for case_name in [
    "rte_case5_example",
    "rte_case5_example_art",
    "l2rpn_2019",
    "l2rpn_2019_art",
    "l2rpn_wcci_2020",
]:
    if "l2rpn_wcci_2020" in case_name:
        continue

    if "l2rpn_wcci_2020" in case_name:
        n_steps = 100
    elif "l2rpn_2019" in case_name:
        n_steps = 500
    else:
        n_steps = 500

    case_save_dir = make_dir(os.path.join(save_dir, f"{case_name}-{env_pf(env_dc)}"))
    create_logger(logger_name=f"logger", save_dir=case_save_dir)

    """
        Initialize environment.
    """
    parameters = CaseParameters(case_name=case_name, env_dc=env_dc)
    case = load_case(case_name, env_parameters=parameters)

    for agent_name in [
        "agent-mip",
        # "agent-multistep-mip",
    ]:
        if "rte_case5" in case_name:
            kwargs["obj_lambda_action"] = 0.004
        elif "l2rpn_2019" in case_name:
            kwargs["obj_lambda_action"] = 0.07
        else:
            kwargs["obj_lambda_action"] = 0.05

        """
            Initialize agent.
        """
        agent = make_test_agent(agent_name, case, **kwargs)

        """
            Experiments.
        """
        experiment_behaviour.evaluate_performance(
            case=case,
            agent=agent,
            save_dir=case_save_dir,
            n_steps=n_steps,
            verbose=verbose,
        )
        experiment_behaviour.aggregate_by_agent(agent=agent, save_dir=case_save_dir)

    experiment_behaviour.compare_agents(save_dir=case_save_dir)
