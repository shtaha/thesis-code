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

    """
    Topology actions.
    """

    def get_all_unitary_topologies_set(
        self, n_bus=2, verbose=False
    ) -> Tuple[List[TopologyAction], List[Dict]]:
        """
        Returns a list of valid topology substation splitting actions. Currently, it returns
        """
        actions_info = list()
        actions = list()
        for sub_id, _ in enumerate(self.action_space.sub_info):
            (
                substation_actions,
                substation_actions_info,
            ) = self.get_all_unitary_topologies_set_sub_id(
                sub_id, n_bus=n_bus, verbose=verbose
            )

            # Check if there is only one valid topology on a given substation, then this topology is fixed and thus
            # the corresponding action redundant.
            if len(substation_actions) > 1:
                actions_info.extend(substation_actions_info)
                actions.extend(substation_actions)
            else:
                print(
                    f"Substation id {sub_id} has only a single valid topology, thus no actions are affecting it."
                )

        # Check if every actions has it corresponding information.
        assert len(actions) == len(actions_info)
        return actions, actions_info

    def get_all_unitary_topologies_set_sub_id(
        self, sub_id, n_bus=2, verbose=False
    ) -> Tuple[List[TopologyAction], List[Dict]]:
        """
        Tested only for n_bus = 2.
        """
        count_valid, count_disconnection = 0, 0
        n_elements = self.action_space.sub_info[sub_id]
        bus_set = np.arange(1, n_bus + 1)

        substation_actions = list()
        substation_actions_info = list()

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
                if check_one_line:
                    count_disconnection = (
                        count_disconnection + 1
                    )  # Add 1 to one line disconnection count.

                    if verbose:
                        print("There is a bus with exactly one line connected.")

                action = self.action_space(
                    {"set_bus": {"substations_id": [(sub_id, topology)]}}
                )
                action_info = {
                    "sub_id": sub_id,
                    "action_type": "topology_set",
                    "topology": topology.tolist(),
                    "check_gen_load": check_gen_load,
                    "check_one_line": check_one_line,
                }

                substation_actions.append(action)
                substation_actions_info.append(action_info)
            else:
                if verbose:
                    print(
                        "Illegal action. Does not satisfy load-generator requirement."
                    )

        if verbose:
            print(
                f"Found {len(substation_actions)} distinct valid substation switching actions."
            )

        _, n_valid, n_disconnection = self.get_number_topologies_set_sub_id(
            sub_id, n_bus=n_bus, verbose=True
        )

        # If there are only 2 or less elements per substation, then there is no action possible, despite one topology
        # being valid.
        if n_elements < 3:
            substation_actions = list()
            substation_actions_info = list()
            count_valid = 1
            count_disconnection = 0

        assert len(substation_actions) == len(substation_actions_info)
        assert n_valid == count_valid
        assert n_disconnection == count_disconnection
        return substation_actions, substation_actions_info

    def get_number_topologies_set_sub_id(self, sub_id, n_bus=2, verbose=False):
        (n_lines, n_gens, n_loads), _ = self._get_substation_info(sub_id)

        (n_actions, n_valid, n_disconnection,) = self._get_number_topologies_set(
            n_lines, n_gens, n_loads, n_bus=n_bus
        )

        if verbose:
            print(
                f"Substation id {sub_id} with {n_lines} lines, {n_gens} generators and {n_loads} loads. "
                f"There are {n_actions} possible actions, {n_valid} are valid with "
                f"{n_disconnection} actions that have a standalone line."
            )
        return n_actions, n_valid, n_disconnection

    """
    Line status actions.
    """

    def get_all_unitary_line_status_set(
        self, n_bus=2, verbose=False
    ) -> Tuple[List[TopologyAction], List[Dict]]:
        """
        Not customized.
        """
        n_lines = self.action_space.n_line

        actions = list()
        actions_info = list()

        for line_id in range(n_lines):
            action = self.action_space.disconnect_powerline(line_id=line_id)
            action_info = {
                "line_id": line_id,
                "action_type": "line_status_set",
                "line_set": "disconnect",
            }

            actions.append(action)
            actions_info.append(action_info)

        for bus_or in np.arange(1, n_bus + 1):
            for bus_ex in np.arange(1, n_bus + 1):
                for line_id in range(n_lines):
                    action = self.action_space.reconnect_powerline(
                        line_id=line_id, bus_ex=bus_ex, bus_or=bus_or
                    )
                    action_info = {
                        "line_id": line_id,
                        "action_type": "line_status_set",
                        "line_set": "reconnect",
                    }

                    actions.append(action)
                    actions_info.append(action_info)

        if verbose:
            print(
                f"Generated {len(actions)} line status set actions, {n_bus ** 2 * n_lines} reconnections and "
                f"{n_lines} disconnections."
            )

        assert len(actions) == len(actions_info)
        return actions, actions_info

    def get_all_unitary_line_status_change(
        self, verbose=False
    ) -> Tuple[List[TopologyAction], List[Dict]]:
        actions = list()
        actions_info = list()

        default_status = self.action_space.get_change_line_status_vect()
        for line_id in range(self.action_space.n_line):
            (
                (substation_or, substation_ex),
                (n_valid_or, n_valid_ex),
            ) = self._get_line_info(line_id)

            line_status = default_status.copy()
            line_status[line_id] = True
            action = self.action_space({"change_line_status": line_status})

            action_info = {
                "line_id": line_id,
                "action_type": "line_status_change",
                "substation_or": (substation_or, n_valid_or),
                "substation_ex": (substation_ex, n_valid_ex),
            }

            actions.append(action)
            actions_info.append(action_info)

        if verbose:
            print(
                f"Generated {len(actions)} line status switching actions, one for each line."
            )

        assert len(actions) == len(actions_info)
        return actions, actions_info

    """
    Redispatching actions.
    """

    def get_all_unitary_redispatch(self, n_redispatch, verbose=False):
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
                # print("ramps up", ramps_up)
                ramps_up = ramps_up[1:]  # Exclude redispatch of 0MW

                # Create evenly spaced negative interval
                ramps_down = np.linspace(
                    -self.action_space.gen_max_ramp_down[gen_id], 0.0, num=5
                )
                ramps_down = ramps_down[:-1]  # Exclude redispatch of 0MW
                # print("ramps down", ramps_down)

                # Merge intervals
                ramps = np.append(ramps_up, ramps_down)
                # print("ramps", ramps)
                # Create ramp up actions
                for ramp in ramps:
                    action = self.action_space({"redispatch": [(gen_id, ramp)]})
                    # print(action)
                    actions.append(action)
            else:
                print(f"Generator id {gen_id} is non-dispatchable.")

        return actions, []

    """
    Filtering functions.
    """

    @staticmethod
    def filter_one_line_disconnections(
        actions: List[TopologyAndDispatchAction],
        actions_info: List[Dict],
        verbose=False,
    ) -> Tuple[List[TopologyAndDispatchAction], List[Dict]]:

        substation_actions = dict()
        substation_actions_info = dict()
        filtered_actions = list()
        filtered_actions_info = list()
        for i, action in enumerate(actions):
            action_info = actions_info[i]
            sub_id = action_info["sub_id"]

            # Initialize substation action list first time it is called.
            if sub_id not in substation_actions:
                substation_actions[sub_id] = list()
                substation_actions_info[sub_id] = list()

            if not action_info["check_one_line"]:
                substation_actions[sub_id].append(action)
                substation_actions_info[sub_id].append(action_info)

        for sub_id in substation_actions:
            # Discard single substation actions.
            if len(substation_actions[sub_id]) > 1:
                filtered_actions.extend(substation_actions[sub_id])
                filtered_actions_info.extend(substation_actions_info[sub_id])
            else:
                if verbose:
                    print(
                        f"There are {len(substation_actions[sub_id])} actions on substation id {sub_id}, "
                        f"thus discarded."
                    )

        assert len(filtered_actions) == len(filtered_actions_info)
        return filtered_actions, filtered_actions_info

    """
    Helper functions.
    """

    def _get_substation_info(
        self, sub_id
    ) -> Tuple[Tuple[int, int, int], Tuple[List, List, List]]:
        lines_or = [
            line
            for line, sub in enumerate(self.action_space.line_or_to_subid)
            if sub == sub_id
        ]
        lines_ex = [
            line
            for line, sub in enumerate(self.action_space.line_ex_to_subid)
            if sub == sub_id
        ]
        lines = lines_or + lines_ex

        gens = [
            gen
            for gen, sub in enumerate(self.action_space.gen_to_subid)
            if sub == sub_id
        ]
        loads = [
            gen
            for gen, sub in enumerate(self.action_space.load_to_subid)
            if sub == sub_id
        ]

        n_lines = len(lines)
        n_gens = len(gens)
        n_loads = len(loads)

        return (n_lines, n_gens, n_loads), (lines, gens, loads)

    def _get_line_info(self, line_id) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        substation_or = self.action_space.line_or_to_subid[line_id]
        substation_ex = self.action_space.line_ex_to_subid[line_id]

        _, n_valid_or, _ = self.get_number_topologies_set_sub_id(substation_or)
        _, n_valid_ex, _ = self.get_number_topologies_set_sub_id(substation_ex)

        return (substation_or, substation_ex), (n_valid_or, n_valid_ex)

    @staticmethod
    def _get_number_topologies_set(
        n_lines, n_gens, n_loads, n_bus=2
    ) -> Tuple[int, int, int]:
        """
        Works only with n_bus = 2.
        """
        n_elements = n_lines + n_gens + n_loads

        if n_bus == 2:
            n_actions = 2 ** n_elements
            n_valid = 2 ** (n_elements - 1) - (2 ** (n_gens + n_loads) - 1)

            # Check for special cases, where only one configuration is possible.
            if n_lines == 1 and ((n_gens + n_loads) > 0):
                n_disconnection = 0
            else:
                n_disconnection = n_lines

            # If there are only one or two elements per substation, then there is only one valid configuration.
            if (n_lines + n_gens + n_loads) < 3:
                n_valid = 1
                n_disconnection = 0
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
