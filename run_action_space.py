import argparse

import grid2op
from grid2op.Action import TopologyAction, TopologySetAction, TopologyAndDispatchAction

from lib.action_space import ActionSpaceGenerator
from lib.constants import Constants as Const
from lib.data_utils import create_results_dir
from lib.visualizer import describe_environment


def parse_arguments():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--env_name", default=Const.ENV_NAME, type=str, help="Environment name.")
    parser.add_argument(
        "--env_name", default="rte_case5_example", type=str, help="Environment name."
    )
    # parser.add_argument("--env_name", default="l2rpn_2019", type=str, help="Environment name.")
    parser.add_argument(
        "--action_cls", default="topology_dispatch", type=str, help="Action class used."
    )
    parser.add_argument("--n_bus", default=2, type=int, help="Number of buses per substation. Tested only for 2.")

    parser.add_argument("-v", "--verbose", help="Set verbosity level.", action="store_false")
    return parser.parse_args()


def get_action_class(action_cls):
    """
    Returns an action class given a action class name.

    The default action class includes grid topology manipulation and redispatching.
    """

    if action_cls == "topology":
        return TopologyAction
    elif action_cls == "topology_set":
        return TopologySetAction
    else:
        return TopologyAndDispatchAction


if __name__ == "__main__":
    args = parse_arguments()

    save_dir = create_results_dir(Const.RESULTS_DIR)

    env_name = args.env_name
    env = grid2op.make(dataset=env_name, action_class=get_action_class(args.action_cls))
    action_space = env.action_space
    describe_environment(env)

    # Generator
    action_generator = ActionSpaceGenerator(env)

    # Grid2Op Generator
    # grid2op_actions_line_set = action_generator.grid2op_get_all_unitary_line_status_set()
    # grid2op_actions_line_change = action_generator.grid2op_get_all_unitary_line_status_change()
    # grid2op_actions_topology_set = action_generator.grid2op_get_all_unitary_topologies_set()
    # grid2op_actions_redispatch = action_generator.grid2op_get_all_unitary_redispatch()

    # Custom Generator with Analysis and Action Information
    actions_topology_set = action_generator.get_all_unitary_topologies_set(n_bus=args.n_bus,
                                                                           verbose=args.verbose)
    # actions_line_set = action_generator.get_all_unitary_line_status_set()
    # actions_line_change = action_generator.get_all_unitary_line_status_change()
    # actions_redispatch = action_generator.get_all_unitary_redispatch()

    # for subid, n_elements in enumerate(env.sub_info):
    #     print(f"\n\nSUBSTATION {subid} with {n_elements} elements")
    #     describe_substation(subid, env)
    #
    #     own_actions, _ = action_generator.get_all_unitary_topologies_set_subid(
    #         subid, n_elements, verbose=False
    #     )
    #
    #     actions_sub = action_generator.grid2op_get_all_unitary_topologies_set_subid(
    #         subid, n_elements
    #     )
    #     print(f"actions sub: grid2op {len(actions_sub)} vs own {len(own_actions)}")

    # obs = env.reset()
    # render_and_save(env, save_dir, env_name)
