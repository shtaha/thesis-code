import numpy as np

from lib.action_space import ActionSpaceGenerator
from lib.dc_opf import (
    GridDCOPF,
    load_case,
)
from lib.visualizer import describe_environment, print_info

"""
    Load environment and initialize grid.
"""
case = load_case("rte_case5")
env = case.env
describe_environment(env)
grid = GridDCOPF(case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p)

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

np.random.seed(1)
obs = env.reset()
topo_vect = obs.topo_vect
line_status = obs.line_status
for t in range(10):
    if t == 0:
        action_idx = np.random.randint(0, len(actions_topology_set_filtered))
        action = actions_topology_set_filtered[action_idx]
    elif t == 1:
        action = actions_line_set[3]
    elif t == 2:
        action = actions_do_nothing
    elif t == 3:
        action_idx = np.random.randint(0, len(actions_topology_set_filtered))
        action = actions_topology_set_filtered[action_idx]
    else:
        action = actions_do_nothing

    print(f"\n\nSTEP {t}")
    print(action)

    print("{:<35}{}\t{}".format("ENV", str(topo_vect), str(line_status.astype(int))))

    obs_next, reward, done, info = env.step(action)
    topo_vect_next = obs_next.topo_vect
    line_status_next = obs_next.line_status
    print_info(info, done, reward)

    """
        Update grid.
    """
    grid.update(obs_next, reset=False, verbose=True)

    topo_vect = topo_vect_next
    line_status = line_status_next
    obs = obs_next

    if done:
        print("\n\nDONE")
        obs = env.reset()
        topo_vect_next = obs.topo_vect
        line_status_next = obs.line_status
        grid.update(obs_next, reset=True, verbose=True)

    if t == 3:
        break
