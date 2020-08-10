import os
import sys

import grid2op
from grid2op.Environment import Environment

from lib.action_space import ActionSpaceGenerator
from lib.constants import Constants as Const
from lib.data_utils import create_results_dir, make_dir
from lib.dc_opf import (
    GridDCOPF,
    MultistepTopologyDCOPF,
    Forecasts,
    MultistepTopologyParameters,
)
from lib.dc_opf import load_case, CaseParameters
from lib.run_utils import Logger
from lib.visualizer import Visualizer

visualizer = Visualizer()

save_dir = create_results_dir(Const.RESULTS_DIR)
sys.stdout = Logger(save_dir=save_dir)

verbose = False
kwargs = {}

case_name = "rte_case5_example"
# case_name = "l2rpn_2019"

for env_dc in [True, False]:
    if not env_dc:
        continue

    env_pf = "dc" if env_dc else "ac"
    case_save_dir = make_dir(os.path.join(save_dir, f"{case_name}-{env_pf}"))

    """
        Initialize environment parameters.    
    """
    parameters = CaseParameters(case_name=case_name, env_dc=env_dc)

    """
        Initialize environment and case.
    """
    env: Environment = grid2op.make_from_dataset_path(
        dataset_path=os.path.join(os.path.expanduser("~"), "data_grid2op", case_name),
        backend=grid2op.Backend.PandaPowerBackend(),
        action_class=grid2op.Action.TopologyAction,
        observation_class=grid2op.Observation.CompleteObservation,
        reward_class=grid2op.Reward.L2RPNReward,
        param=parameters,
    )
    case = load_case(case_name, env=env, verbose=True)

    """
        Initialize action set.
    """
    action_generator = ActionSpaceGenerator(env)
    action_set = action_generator.get_topology_action_set(verbose=verbose)

    """
        Initialize agent.
    """
    horizon = 2
    params = MultistepTopologyParameters(horizon=horizon)

    grid = GridDCOPF(case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p)
    grid.print_grid()

    done = False
    obs = env.reset()

    forecasts = Forecasts(env=env, horizon=horizon)
    grid.update(obs_new=obs, reset=done, verbose=verbose)

    model = MultistepTopologyDCOPF(
        f"{case.name} Multiple Time Step DC OPF Topology Optimization",
        grid=grid,
        forecasts=forecasts,
        base_unit_p=case.base_unit_p,
        base_unit_v=case.base_unit_v,
        params=params,
    )

    model.build_model()
    model.print_model()

    model.solve(verbose=False)
