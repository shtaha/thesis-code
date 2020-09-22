import os

import numpy as np

from experience import load_experience
from experiments import analyse_actions, analyse_loading
from lib.constants import Constants as Const
from lib.data_utils import make_dir, env_pf
from lib.run_utils import create_logger
from lib.visualizer import Visualizer, pprint

visualizer = Visualizer()

experience_dir = make_dir(os.path.join(Const.RESULTS_DIR, "performance-aug"))
case_name = "l2rpn_2019_art"

env_dc = True
verbose = False

case_save_dir = make_dir(os.path.join(experience_dir, f"{case_name}-{env_pf(env_dc)}"))
create_logger(logger_name=f"logger", save_dir=case_save_dir)

for agent_name in [
    "do-nothing-agent",
    "agent-mip",
    "agent-multistep-mip",
]:

    case, collector = load_experience(
        case_name, agent_name, experience_dir, env_dc=env_dc
    )
    obses, actions, rewards, dones = collector.aggregate_data()

    if obses:
        labels = np.array(
            [action != case.env.action_space({}) for action in actions], dtype=np.float
        )

        pprint("    - Labels:", labels.shape, f"{100 * labels.mean()} %")
        pprint("    - Number of chronics:", dones.sum())
        pprint("    - Observations:", len(obses))

        """
            Action analysis
        """
        case.env.reset()
        fig = case.env.render()
        fig.savefig(os.path.join(case_save_dir, "_sample-grid"))

        if agent_name != "do-nothing-agent":
            results_dir = make_dir(os.path.join(case_save_dir, "analysis-actions"))
            analyse_actions(actions, case, agent_name, save_dir=results_dir)

        results_dir = make_dir(os.path.join(case_save_dir, "analysis-loading"))
        analyse_loading(obses, case, agent_name, save_dir=results_dir)
