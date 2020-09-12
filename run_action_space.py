import argparse

from lib.action_space import ActionSpaceGenerator
from lib.dc_opf import load_case


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name", default="l2rpn_2019", type=str, help="Environment name."
    )
    return parser.parse_args()


if __name__ == "__main__":
    verbose = True
    args = parse_arguments()

    case = load_case(args.env_name, verbose=verbose)
    env = case.env
    action_space = env.action_space

    # Generator
    action_generator = ActionSpaceGenerator(env)

    # Grid2Op Generator
    grid2op_actions_topology_set = (
        action_generator.grid2op_get_all_unitary_topologies_set()
    )
    grid2op_actions_line_set = (
        action_generator.grid2op_get_all_unitary_line_status_set()
    )

    # Custom Generator with Analysis and Action Information
    actions_do_nothing = action_space({})
    (
        actions_topology_set,
        actions_topology_set_info,
    ) = action_generator.get_all_unitary_topologies_set(
        verbose=False, filter_one_line_disconnections=False
    )
    (
        actions_topology_set_filtered,
        actions_topology_set_filtered_info,
    ) = action_generator.filter_one_line_disconnections(
        actions_topology_set, actions_topology_set_info, verbose=verbose
    )

    (
        actions_line_set,
        actions_line_set_info,
    ) = action_generator.get_all_unitary_line_status_set(verbose=verbose)

    print("actions: 1 do-nothing action")

    print("Topology set actions:")
    print("{:<20}\t{}".format("grid2op", len(grid2op_actions_topology_set)))
    print("{:<20}\t{}".format("custom", len(actions_topology_set)))
    print("{:<20}\t{}".format("custom filtered", len(actions_topology_set_filtered)))

    print("Line set actions:")
    print("{:<20}\t{}".format("grid2op", len(grid2op_actions_line_set)))
    print("{:<20}\t{}".format("custom", len(actions_line_set)))
