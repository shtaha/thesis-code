import os

import numpy as np

from experience import ExperienceCollector
from experiments import analyse_actions
from lib.constants import Constants as Const
from lib.data_utils import make_dir, env_pf
from lib.dc_opf import load_case, CaseParameters
from lib.run_utils import create_logger
from lib.visualizer import Visualizer, pprint

visualizer = Visualizer()

experience_data_dir = make_dir(os.path.join(Const.EXPERIENCE_DIR, "data-s"))

agent_name = "agent-mip"
case_name = "l2rpn_2019"

env_dc = True
verbose = False

case_experience_data_dir = make_dir(
    os.path.join(experience_data_dir, f"{case_name}-{env_pf(env_dc)}")
)
case_results_dir = make_dir(os.path.join(case_experience_data_dir, "actions-analysis"))
create_logger(logger_name=f"{case_name}-{env_pf(env_dc)}", save_dir=case_results_dir)

parameters = CaseParameters(case_name=case_name, env_dc=env_dc)
case = load_case(case_name, env_parameters=parameters)
env = case.env

"""
    Load dataset
"""

collector = ExperienceCollector(save_dir=case_experience_data_dir)
collector.load_data(agent_name=agent_name, env=env)

observations, actions, rewards, dones = collector.aggregate_data()
labels = np.array(
    [action != env.action_space({}) for action in actions], dtype=np.float
)

pprint("Labels:", labels.shape, f"{100 * labels.mean()} %")

"""
    Action analysis
"""
analyse_actions(actions, env, agent_name, save_dir=case_results_dir)
