import os

import grid2op
import numpy as np
from grid2op.Environment import Environment

from lib.action_space import ActionSpaceGenerator
from lib.dc_opf import (
    GridDCOPF,
    OPFRTECase5,
    OPFL2RPN2019,
    OPFL2RPN2020,
    TopologyOptimizationDCOPF,
)
from lib.rewards import RewardL2RPN2019
from lib.visualizer import (
    describe_environment,
    print_info,
    print_parameters,
)

"""
    Load environment and initialize grid.
"""
case_name = "rte_case5_example"
# case_name = "l2rpn_2019"
# case_name = "l2rpn_wcci_2020"

parameters = grid2op.Parameters.Parameters()
parameters.ENV_DC = True
parameters.FORECAST_DC = True

env: Environment = grid2op.make_from_dataset_path(
    dataset_path=os.path.join(os.path.expanduser("~"), "data_grid2op", case_name),
    backend=grid2op.Backend.PandaPowerBackend(),
    action_class=grid2op.Action.TopologyAction,
    observation_class=grid2op.Observation.CompleteObservation,
    reward_class=grid2op.Reward.L2RPNReward,
    param=parameters,
)

if case_name == "rte_case5_example":
    case = OPFRTECase5(env=env)
elif case_name == "l2rpn_2019":
    case = OPFL2RPN2019(env=env)
elif case_name == "l2rpn_wcci_2020":
    case = OPFL2RPN2020(env=env)
else:
    raise ValueError("Invalid environment name.")

parameters = env.get_params_for_runner()["parameters_path"]

reward_function = RewardL2RPN2019()

describe_environment(env)
print_parameters(env)

for key in env.get_params_for_runner():
    if "opponent" not in key:
        print("{:<35}{}".format(key, env.get_params_for_runner()[key]))

"""
    GRID AND MIP MODEL.
"""
grid = GridDCOPF(case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p)
grid.print_grid()

"""
    Generate actions.
"""
action_generator = ActionSpaceGenerator(env)
(
    actions_line_set,
    actions_line_set_info,
) = action_generator.get_all_unitary_line_status_set()
(
    actions_topology_set,
    actions_topology_set_info,
) = action_generator.get_all_unitary_topologies_set()
(
    actions_topology_set_filtered,
    actions_topology_set_filtered_info,
) = action_generator.filter_one_line_disconnections(
    actions_topology_set, actions_topology_set_info
)
actions_do_nothing = env.action_space({})

"""
    Initialize topology converter.
"""
# TODO: Maintenance support, CHECK constraints
# TODO: Infeasible problem?
# TODO: Reconnection of a powerline: status + buses

np.random.seed(1)
obs = env.reset()
print(
    "\n{:<35}{}\t{}".format("ENV", str(obs.topo_vect), str(obs.line_status.astype(int)))
)

for t in range(10):
    """
        Action selection.
    """
    model = TopologyOptimizationDCOPF(
        f"{case.name} DC OPF Topology Optimization",
        grid=grid,
        grid_backend=case.grid_backend,
        base_unit_p=case.base_unit_p,
        base_unit_v=case.base_unit_v,
        n_max_line_status_changed=parameters["MAX_LINE_STATUS_CHANGED"],
        n_max_sub_changed=parameters["MAX_SUB_CHANGED"],
    )
    model.build_model(
        line_disconnection=True,
        symmetry=True,
        switching_limits=True,
        cooldown=True,
        gen_cost=False,
        line_margin=False,
        min_rho=False,
        bound_max_flow=True,
    )

    result = model.solve(tol=0.001, verbose=False)
    mip_topo_vect, mip_line_status, action = grid.convert_mip_to_topology_vector(result)

    """
        Act.
    """
    print(f"\n\nSTEP {t}")
    print(action)
    obs_next, reward, done, info = env.step(action)
    print(
        "{:<35}{}\t{}".format(
            "ENV", str(obs_next.topo_vect), str(obs_next.line_status.astype(int))
        )
    )
    print(
        "{:<35} {}\t{}".format(
            "REWARD ESTIMATE:",
            reward_function.from_observation(obs_next),
            reward_function.from_mip_solution(result),
        )
    )
    print_info(info, done, reward)
    model.compare_with_observation(result, obs_next, verbose=True)

    """
        Update grid.
    """
    grid.update(obs_next, reset=False, verbose=True)

    obs = obs_next

    if done:
        print("\n\nDONE")
        obs = env.reset()
        print(
            "\n{:<35}{}\t{}".format(
                "ENV", str(obs.topo_vect), str(obs.line_status.astype(int))
            )
        )
        grid.update(obs_next, reset=True, verbose=True)

    # if t == 3:
    #     break
    break
