import os

import grid2op
from grid2op.Environment import Environment

from experiments import ExperimentFailure
from lib.constants import Constants as Const
from lib.data_utils import create_results_dir
from lib.dc_opf import load_case, CaseParameters
from lib.run_utils import create_logger
from lib.visualizer import Visualizer

visualizer = Visualizer()

save_dir = create_results_dir(Const.RESULTS_DIR)
create_logger(save_dir=save_dir)

do_experiment_failure = True
verbose = False

experiment_failure = ExperimentFailure()

env_dc = True
case_name = "l2rpn_2019"

env_pf = "dc" if env_dc else "ac"
case_save_dir = "./results/2020-08-18_21-22-45_results/l2rpn_2019-dc"

parameters = CaseParameters(case_name=case_name, env_dc=env_dc)

env: Environment = grid2op.make_from_dataset_path(
    dataset_path=os.path.join(os.path.expanduser("~"), "data_grid2op", case_name),
    backend=grid2op.Backend.PandaPowerBackend(),
    action_class=grid2op.Action.TopologyAction,
    observation_class=grid2op.Observation.CompleteObservation,
    reward_class=grid2op.Reward.L2RPNReward,
    param=parameters,
)
case = load_case(case_name, env=env)

if do_experiment_failure:
    experiment_failure.compare_agents(case, save_dir=case_save_dir)
