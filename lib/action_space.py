import collections
import itertools
from typing import List, Tuple, Dict

import numpy as np
from grid2op.Action import (
    TopologyAction,
    TopologyAndDispatchAction,
    SerializableActionSpace,
)
from grid2op.dtypes import dt_int

from lib.data_utils import hot_vector


class ActionSpaceGenerator(object):
    def __init__(self, env):
        self.env = env
        self.action_space = self.env.action_space

    """
    grid2op action generator functions.
    """

    def grid2op_get_all_unitary_topologies_set(self) -> List[TopologyAction]:
        """
        Returns a list of all unitary topology configurations within each substation. This is
        the standard grid2op implementation.
        """
        return SerializableActionSpace.get_all_unitary_topologies_set(self.action_space)

    def grid2op_get_all_unitary_line_status_set(self) -> List[TopologyAction]:
        """
        Returns a list of all unitary line status configurations. This is
        the standard grid2op implementation.
        """
        return SerializableActionSpace.get_all_unitary_line_set(self.action_space)

    def grid2op_get_all_unitary_line_status_change(self) -> List[TopologyAction]:
        """
        Returns a list of all unitary line status switch configurations. This is
        the standard grid2op implementation.
        """
        return SerializableActionSpace.get_all_unitary_line_change(self.action_space)

    def grid2op_get_all_unitary_redispatch(self) -> List[TopologyAndDispatchAction]:
        """
        Returns a list of unitary redispatch actions equally spaced between maximum generator up and down ramps.
        The number of actions for each generator is fixed.
        """
        return SerializableActionSpace.get_all_unitary_redispatch(self.action_space)

    """
    Customized action generation functions. 
    """

    def get_all_unitary_topologies_set(
            self, n_bus=2, verbose=False
    ) -> Tuple[List[TopologyAction], List[Dict]]:
        """
        Returns a list of valid topology substation splitting actions. Currently, it returns
        """
        action_info = list()
        actions = list()
        for sub_id, _ in enumerate(self.action_space.sub_info):
            (
                substation_actions,
                substation_topologies,
            ) = self.get_all_unitary_topologies_set_sub_id(
                sub_id, n_bus=n_bus, verbose=verbose
            )

            # Check if there is only one valid topology on a given substation, then this topology is fixed and thus
            # the corresponding action redundant.
            if len(substation_actions) > 1:
                substation_info = [{"sub_id": sub_id}]
                action_info.extend(substation_info)
                actions.extend(substation_actions)
            else:
                print(
                    f"Substation id {sub_id} has only a single valid topology, thus no actions are affecting it."
                )

        return actions, action_info

    def get_all_unitary_topologies_set_sub_id(
            self, sub_id, n_bus=2, verbose=False
    ) -> Tuple[List[TopologyAction], List[np.ndarray]]:
        """
        Tested only for n_bus = 2.
        """
        count_valid, count_disconnection = 0, 0
        n_elements = self.action_space.sub_info[sub_id]
        bus_set = np.arange(1, n_bus + 1)

        substation_topologies = list()
        substation_actions = list()

        # Get line positions within a substation
        lines_or_pos = self.action_space.line_or_to_sub_pos[
            self.action_space.line_or_to_subid == sub_id
            ]
        lines_ex_pos = self.action_space.line_ex_to_sub_pos[
            self.action_space.line_ex_to_subid == sub_id
            ]
        lines_pos = np.concatenate((lines_or_pos, lines_ex_pos))

        # Get load and generator positions within a substation
        gen_pos = self.action_space.gen_to_sub_pos[
            self.action_space.gen_to_subid == sub_id
            ]
        load_pos = self.action_space.load_to_sub_pos[
            self.action_space.load_to_subid == sub_id
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
                count_valid = count_valid + 1  # Add 1 to valid action count.

                # Check if there exists a bus with exactly one line, thus this line is implicitly disconnected.
                check_one_line = self._check_one_line_on_bus(topology, n_bus)
                if check_one_line and verbose:
                    count_disconnection = count_disconnection + 1  # Add 1 to one line disconnection count.
                    print("There is a bus with exactly one line connected.")

                action = self.action_space(
                    {"set_bus": {"substations_id": [(sub_id, topology)]}}
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
            _, n_valid, n_disconnection = self.get_number_topologies_set_sub_id(sub_id, n_bus=n_bus, verbose=verbose)
            assert n_valid == count_valid
            assert n_disconnection == count_disconnection

        return substation_actions, substation_topologies

    def get_all_unitary_line_status_set(self, n_bus=2):
        actions = list()

        _, substation_action_info = self.get_all_unitary_topologies_set(
            n_bus=n_bus, verbose=False
        )
        counts = collections.Counter(
            [info["sub_id"] for info in substation_action_info]
        )
        print(counts)
        substation_counts = np.array(
            [counts[sub_id] for sub_id in range(self.action_space.sub_info)]
        )
        print(substation_counts)
        print(np.equal(substation_counts, 0))

        for i in range(self.action_space.n_line):
            actions.append(self.action_space.disconnect_powerline(line_id=i))

        for bus_or in np.arange(1, n_bus + 1):
            for bus_ex in np.arange(1, n_bus + 1):
                print(bus_or, bus_ex)
                for i in range(self.action_space.n_line):
                    action = self.action_space.reconnect_powerline(
                        line_id=i, bus_ex=bus_ex, bus_or=bus_or
                    )

                    actions.append(action)

        return actions

    def get_all_unitary_line_status_change(self, verbose=False):
        actions = list()

        default_status = self.action_space.get_change_line_status_vect()
        for line_id in range(self.action_space.n_line):
            line_status = default_status.copy()
            line_status[line_id] = True

            action = self.action_space({"change_line_status": line_status})

            if verbose:
                pass
            print(action)
            actions.append(action)

        return actions

    def get_all_unitary_redispatch(self):
        """

        """

        actions = list()
        n_gen = len(self.action_space.gen_redispatchable)

        for gen_id in range(n_gen):
            # Skip non-dispatchable generators
            if self.action_space.gen_redispatchable[gen_id]:
                # Create evenly spaced positive interval
                ramps_up = np.linspace(
                    0.0, self.action_space.gen_max_ramp_up[gen_id], num=5
                )
                ramps_up = ramps_up[1:]  # Exclude redispatch of 0MW

                # Create evenly spaced negative interval
                ramps_down = np.linspace(
                    -self.action_space.gen_max_ramp_down[gen_id], 0.0, num=5
                )
                ramps_down = ramps_down[:-1]  # Exclude redispatch of 0MW

                # Merge intervals
                ramps = np.append(ramps_up, ramps_down)

                # Create ramp up actions
                for ramp in ramps:
                    action = self.action_space({"redispatch": [(gen_id, ramp)]})
                    actions.append(action)
            else:
                print(f"Generator id {gen_id} is non-dispatchable.")

        return actions

    def get_number_topologies_set_sub_id(self, sub_id, n_bus=2, verbose=False):
        n_lines = np.sum(self.action_space.line_ex_to_subid == sub_id) + np.sum(
            self.action_space.line_or_to_subid == sub_id
        )
        n_gens = np.sum(self.action_space.gen_to_subid == sub_id)
        n_loads = np.sum(self.action_space.load_to_subid == sub_id)

        (
            n_actions,
            n_valid,
            n_disconnection,
        ) = self._get_number_topologies_set(n_lines, n_gens, n_loads, n_bus=n_bus)

        if verbose:
            print(
                f"Substation id {sub_id} with {n_lines} lines, {n_gens} generators and {n_loads} loads. "
                f"There are {n_actions} possible actions, {n_valid} are valid and include "
                f"{n_disconnection} actions that have a standalone line."
            )
        return n_actions, n_valid, n_disconnection

    @staticmethod
    def _get_number_topologies_set(n_lines, n_gens, n_loads, n_bus=2) -> Tuple[int, int, int]:
        """
        Works only with n_bus = 2.
        """
        n_elements = n_lines + n_gens + n_loads

        if n_bus == 2:
            n_actions = 2 ** n_elements
            n_valid = 2 ** (n_elements - 1) - (2 ** (n_gens + n_loads) - 1)
            n_disconnection = n_lines
        else:
            n_actions = None
            n_valid = None
            n_disconnection = None

        return n_actions, n_valid, n_disconnection

    @staticmethod
    def _check_gen_load_requirement(topology, lines_pos, not_lines_pos, n_bus=2):
        """

        """

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
    def _check_one_line_on_bus(topology, n_bus=2):
        """

        """
        counts = collections.Counter(topology)
        counts_per_bus = np.array([counts[bus] for bus in np.arange(1, n_bus + 1)])

        # Check if there is a bus with exactly one element.
        # Since this is a valid topology, therefore the standalone element is a line.
        check = np.equal(counts_per_bus, 1).any()
        return check
