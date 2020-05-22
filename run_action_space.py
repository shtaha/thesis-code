import argparse
import itertools
import collections
from typing import List, Tuple, Dict

import grid2op
import numpy as np
from grid2op.Action import TopologyAction, TopologySetAction, SerializableActionSpace
from grid2op.dtypes import dt_int, dt_bool

from lib.constants import Constants as Const
from lib.data_utils import create_results_dir, hot_vector
from lib.visualizer import describe_environment, describe_substation, print_action


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default=Const.ENV_NAME, type=str, help="Environment name.")
    # parser.add_argument(
    #     "--env_name", default="rte_case5_example", type=str, help="Environment name."
    # )
    # parser.add_argument("--env_name", default="l2rpn_2019", type=str, help="Environment name.")
    parser.add_argument(
        "--action_cls", default="topology_set", type=str, help="Action class used."
    )

    return parser.parse_args()


def get_action_class(action_cls):
    """
    Returns an action class given a action class name.
    """

    if action_cls == "topology":
        return TopologyAction
    else:
        return TopologySetAction


class ActionSpaceGenerator(object):
    def __init__(self, env):
        self.env = env
        self.action_space = self.env.action_space

    def grid2op_get_all_unitary_topologies_set(self) -> List[TopologyAction]:
        """
        Returns a list of all unitary topology configurations within each substation. This is
        the standard grid2op implementation.
        """
        return SerializableActionSpace.get_all_unitary_topologies_set(self.action_space)

    def get_all_unitary_topologies_set(self, verbose=False) -> Tuple[List[TopologyAction], List[Dict]]:
        """
        Returns a list of valid topology substation splitting actions. Currently, it returns
        """
        action_info = []
        actions = []
        for subid, n_elements in enumerate(self.action_space.sub_info):
            substation_actions, substation_topologies = self.get_all_unitary_topologies_set_subid(subid, n_elements,
                                                                                                  verbose=verbose)

            # Check if there is only one valid topology on a given substation, then this topology is fixed and thus
            # the corresponding action redundant.
            if len(substation_actions) > 1:
                substation_info = [{"subid": subid}]
                action_info.extend(substation_info)
                actions.extend(substation_actions)
            else:
                print(f"Substation id {subid} has only a single valid topology, thus no actions are affecting it.")

        return actions, action_info

    def get_all_unitary_topologies_set_subid(
            self, subid, n_elements, n_bus=2, verbose=False
    ) -> Tuple[List[TopologyAction], List[np.ndarray]]:
        """
        Tested only for n_bus = 2.
        """
        bus_set = np.arange(1, n_bus + 1)

        substation_topologies = []
        substation_actions = []

        # Get line positions within a substation
        lines_or_pos = self.action_space.line_or_to_sub_pos[
            self.action_space.line_or_to_subid == subid
            ]
        lines_ex_pos = self.action_space.line_ex_to_sub_pos[
            self.action_space.line_ex_to_subid == subid
            ]
        lines_pos = np.concatenate((lines_or_pos, lines_ex_pos))

        # Get load and generator positions within a substation
        gen_pos = self.action_space.gen_to_sub_pos[
            self.action_space.gen_to_subid == subid
            ]
        load_pos = self.action_space.load_to_sub_pos[
            self.action_space.load_to_subid == subid
            ]
        not_lines_pos = np.concatenate((gen_pos, load_pos))

        # Get binary positions
        lines_pos = hot_vector(lines_pos, length=n_elements, dtype=np.bool)
        not_lines_pos = hot_vector(not_lines_pos, length=n_elements, dtype=np.bool)

        # Check if the positions of lines, loads and generators are correct.
        if not np.equal(~lines_pos, not_lines_pos).all():
            raise ValueError(
                "Positions of lines, loads and generators do not match within a substation."
            )

        if verbose:
            print("lines {:>30}".format(" ".join([str(int(pos)) for pos in lines_pos])))
            print(
                "not lines {:>26}".format(
                    " ".join([str(int(pos)) for pos in not_lines_pos])
                )
            )

        for topology_id, topology in enumerate(
                itertools.product(bus_set, repeat=n_elements - 1)
        ):
            # Fix the first element on bus 1 -> [1, _, _, _] to break the symmetry.
            topology = np.concatenate(
                (np.ones((1,), dtype=dt_int), np.array(topology, dtype=dt_int))
            )

            if verbose:
                print(
                    "id: {:>3}{:>29}".format(
                        topology_id, " ".join([str(bus) for bus in topology])
                    )
                )

            # Check if any generator or load is connected to a bus that does not include a line.
            check_gen_load = self._check_gen_load_requirement(
                topology, lines_pos, not_lines_pos, n_bus
            )

            if check_gen_load:
                # Check if there exists a bus with exactly one line, thus this line is implicitly disconnected.
                check_one_line = self._check_one_line_on_bus(topology, n_bus)
                if check_one_line and verbose:
                    print("There is a bus with exactly one line connected.")

                action = self.action_space(
                    {"set_bus": {"substations_id": [(subid, topology)]}}
                )
                substation_topologies.append(topology)
                substation_actions.append(action)
            else:
                if verbose:
                    print(
                        "Illegal action. Does not satisfy load-generator requirement."
                    )

        if verbose:
            print(
                f"Found {len(substation_actions)} distinct valid substation switching actions."
            )

        return substation_actions, substation_topologies

    @staticmethod
    def _check_gen_load_requirement(topology, lines_pos, not_lines_pos, n_bus):
        for bus in np.arange(1, n_bus + 1):

            # Check if at least one load or generator connected to the bus.
            gen_load_connected_to_bus = np.any(topology[not_lines_pos] == bus)
            if gen_load_connected_to_bus:

                # Since at least one generator or load is connected to the bus, then at least one line must be also.
                # Otherwise the action is illegal, thus return False.
                line_connected_to_bus = np.any(topology[lines_pos] == bus)
                if not line_connected_to_bus:
                    return False

        return True

    @staticmethod
    def _check_one_line_on_bus(topology, n_bus):
        """

        """

        counts = collections.Counter(topology)
        counts_per_bus = np.array([counts[bus] for bus in np.arange(1, n_bus + 1)])

        # Check if there is a bus with exactly one element.
        # Since this is a valid topology, therefore the standalone element is a line.
        check = np.equal(counts_per_bus, 1).any()

        return check

    def get_all_unitary_line_status_set(self):
        """

        """

    @staticmethod
    def get_all_unitary_line_set(action_space):
        res = []

        # powerline switch: disconnection
        for i in range(action_space.n_line):
            res.append(action_space.disconnect_powerline(line_id=i))

        # powerline switch: reconnection
        for bus_or in [1, 2]:
            for bus_ex in [1, 2]:
                for i in range(action_space.n_line):
                    act = action_space.reconnect_powerline(line_id=i, bus_ex=bus_ex, bus_or=bus_or)
                    res.append(act)

        return res

    @staticmethod
    def get_all_unitary_line_change(action_space):
        res = []

        for i in range(action_space.n_line):
            status = action_space.get_change_line_status_vect()
            status[i] = True
            res.append(action_space({"change_line_status": status}))

        return res


if __name__ == "__main__":
    args = parse_arguments()

    save_dir = create_results_dir(Const.RESULTS_DIR)

    env_name = args.env_name
    env = grid2op.make(dataset=env_name, action_class=get_action_class(args.action_cls))
    action_space = env.action_space

    describe_environment(env)

    action_generator = ActionSpaceGenerator(env)
    actions = action_generator.grid2op_get_all_unitary_topologies_set()
    own_actions, own_info = action_generator.get_all_unitary_topologies_set()
    print(len(actions), len(own_actions))
    #
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
