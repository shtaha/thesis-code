from lib.action_space import ActionSpaceGenerator
from lib.dc_opf import load_case
from lib.visualizer import pprint

if __name__ == "__main__":
    case_name = "l2rpn_2019"
    verbose = True

    case = load_case(case_name, verbose=verbose)
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

    pprint("actions: 1 do-nothing action")

    pprint("Topology set actions:")
    pprint("grid2op", len(grid2op_actions_topology_set))
    pprint("custom", len(actions_topology_set))
    pprint("custom filtered", len(actions_topology_set_filtered))

    pprint("Line set actions:")
    pprint("grid2op", len(grid2op_actions_line_set))
    pprint("custom", len(actions_line_set))
