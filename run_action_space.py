import argparse

import grid2op
from grid2op.Action import TopologyAction, TopologySetAction, TopologyAndDispatchAction

from lib.action_space import ActionSpaceGenerator
from lib.constants import Constants as Const
from lib.data_utils import create_results_dir
from lib.visualizer import describe_environment, render_and_save


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name", default=Const.ENV_NAME, type=str, help="Environment name."
    )
    # parser.add_argument(
    #     "--env_name", default="l2rpn_2019", type=str, help="Environment name."
    # )
    # parser.add_argument("--env_name", default="l2rpn_2019", type=str, help="Environment name.")
    parser.add_argument(
        "--action_cls", default="topology_dispatch", type=str, help="Action class used."
    )
    parser.add_argument(
        "--n_bus",
        default=2,
        type=int,
        help="Number of buses per substation. Tested only for 2.",
    )
    parser.add_argument(
        "--n_redispatch",
        default=4,
        type=int,
        help="Number of redispatching actions per generator, the actual number "
        "of actions is doubled - positive and negative.",
    )
    parser.add_argument(
        "-v", "--verbose", help="Set verbosity level.", action="store_false"
    )
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
    grid2op_actions_topology_set = (
        action_generator.grid2op_get_all_unitary_topologies_set()
    )
    grid2op_actions_line_set = (
        action_generator.grid2op_get_all_unitary_line_status_set()
    )
    grid2op_actions_line_change = (
        action_generator.grid2op_get_all_unitary_line_status_change()
    )
    grid2op_actions_redispatch = action_generator.grid2op_get_all_unitary_redispatch()

    # Custom Generator with Analysis and Action Information
    actions_do_nothing = action_space({})
    (
        actions_topology_set,
        actions_topology_set_info,
    ) = action_generator.get_all_unitary_topologies_set(n_bus=args.n_bus, verbose=False)
    (
        actions_topology_set_filtered,
        actions_topology_set_filtered_info,
    ) = action_generator.filter_one_line_disconnections(
        actions_topology_set, actions_topology_set_info, verbose=args.verbose
    )

    (
        actions_line_set,
        actions_line_set_info,
    ) = action_generator.get_all_unitary_line_status_set(
        n_bus=args.n_bus, verbose=args.verbose
    )

    (
        actions_line_change,
        actions_line_change_info,
    ) = action_generator.get_all_unitary_line_status_change(verbose=False)

    (
        actions_redispatch,
        actions_redispatch_info,
    ) = action_generator.get_all_unitary_redispatch(
        n_redispatch=args.n_redispatch, verbose=False
    )

    print("\n")
    print("actions: 1 do-nothing action")

    print("Topology set actions:")
    print("{:<20}\t{}".format("grid2op", len(grid2op_actions_topology_set)))
    print("{:<20}\t{}".format("custom", len(actions_topology_set)))
    print("{:<20}\t{}".format("custom filtered", len(actions_topology_set_filtered)))

    print("Line set actions:")
    print("{:<20}\t{}".format("grid2op", len(grid2op_actions_line_set)))
    print("{:<20}\t{}".format("custom", len(actions_line_set)))

    print("Line change actions:")
    print("{:<20}\t{}".format("grid2op", len(grid2op_actions_line_change)))
    print("{:<20}\t{}".format("custom", len(actions_line_change)))

    print("Redispatching actions:")
    print("{:<20}\t{}".format("grid2op", len(grid2op_actions_redispatch)))
    print("{:<20}\t{}".format("custom", len(actions_redispatch)))

    obs = env.reset()
    render_and_save(env, save_dir, env_name)
