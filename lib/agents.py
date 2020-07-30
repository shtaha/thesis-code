from timeit import default_timer as timer

import numpy as np
import pandas as pd

from lib.dc_opf import (
    GridDCOPF,
    TopologyOptimizationDCOPF,
)
from lib.rewards import RewardL2RPN2019
from lib.visualizer import pprint


class AgentMIPTest:
    """
        Agent class used for experimentation and testing.
    """

    def __init__(
        self, case, action_set, reward_class=RewardL2RPN2019, **kwargs,
    ):
        self.name = "Agent MIP"
        self.case = case
        self.env = case.env

        self.grid = GridDCOPF(
            case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p
        )

        self.model = None
        self.default_kwargs = kwargs
        self.model_kwargs = self.default_kwargs

        self.result = None

        self.reward_function = reward_class()
        self.actions, self.actions_info = action_set

    def set_kwargs(self, **kwargs):
        self.model_kwargs = {**self.default_kwargs, **kwargs}

    def act(self, observation, reward, done=False):
        self._update(observation, reset=done)
        self.model = TopologyOptimizationDCOPF(
            f"{self.case.env.name} DC OPF Topology Optimization",
            grid=self.grid,
            base_unit_p=self.grid.base_unit_p,
            base_unit_v=self.grid.base_unit_v,
            **self.model_kwargs,
        )

        self.model.build_model()
        self.result = self.model.solve()

        action = self.grid.convert_mip_to_topology_vector(self.result, observation)[-1]
        return action

    def act_with_timing(self, observation, reward, done=False):
        timing = dict()
        start_build = timer()
        self._update(observation, reset=done)
        timing["update"] = timer() - start_build

        self.model = TopologyOptimizationDCOPF(
            f"{self.case.env.name} DC OPF Topology Optimization",
            grid=self.grid,
            base_unit_p=self.grid.base_unit_p,
            base_unit_v=self.grid.base_unit_v,
            **self.model_kwargs,
        )
        self.model.build_model()
        timing["build"] = timer() - start_build

        start_solve = timer()
        self.result = self.model.solve(verbose=False)
        action = self.grid.convert_mip_to_topology_vector(self.result, observation)[-1]
        timing["solve"] = timer() - start_solve

        return action, timing

    def _update(self, obs, reset=False, verbose=False):
        self.grid.update(obs, reset=reset, verbose=verbose)

    def get_reward(self):
        return self.reward_function.from_mip_solution(self.result)

    def compare_with_observation(self, obs, verbose=False):
        res_gen = self.result["res_gen"][["min_p_pu", "p_pu", "max_p_pu"]].copy()
        res_gen["env_p_pu"] = self.grid.convert_mw_to_per_unit(obs.prod_p)
        res_gen["diff"] = np.divide(
            np.abs(res_gen["p_pu"] - res_gen["env_p_pu"]), res_gen["env_p_pu"] + 1e-9
        )

        res_line = self.result["res_line"][["p_pu", "max_p_pu"]].copy()
        res_line = res_line.append(
            self.result["res_trafo"][["p_pu", "max_p_pu"]].copy(), ignore_index=True
        )
        res_line["env_p_pu"] = self.grid.convert_mw_to_per_unit(
            obs.p_or
        )  # i_pu * v_pu * sqrt(3)
        res_line["env_max_p_pu"] = np.abs(
            np.divide(res_line["env_p_pu"], obs.rho + 1e-9)
        )

        res_line["rho"] = self.result["res_line"]["loading_percent"] / 100.0
        res_line["env_rho"] = obs.rho

        # Reactive/Active power ratio
        res_gen["env_q_pu"] = self.grid.convert_mw_to_per_unit(obs.prod_q)
        res_gen["env_gen_q_p"] = np.greater(obs.prod_p, 1e-9).astype(float) * np.abs(
            np.divide(obs.prod_q, obs.prod_p + 1e-9)
        )

        res_line["diff_p"] = np.abs(
            np.divide(
                res_line["p_pu"] - res_line["env_p_pu"], res_line["env_p_pu"] + 1e-9
            )
        )
        res_line["diff_rho"] = np.abs(
            np.divide(res_line["rho"] - res_line["env_rho"], res_line["env_rho"] + 1e-9)
        )

        if verbose:
            print("GEN\n" + res_gen.to_string())
            print("LINE\n" + res_line.to_string())

        return res_line, res_gen

    def distance_to_ref_topology(self, topo_vect, line_status):
        """
        Count the number of unitary topological actions a topology is from the reference topology.
        The reference topology is the base case topology, fully meshed, with every line in service and a single
        electrical node, bus, per substation.
        """
        topo_vect = topo_vect.copy()
        line_status = line_status.copy()

        ref_topo_vect = np.ones_like(topo_vect)
        ref_line_status = np.ones_like(line_status)

        dist_status = 0
        for line_id, status in enumerate(line_status):
            if not status:
                line_or = self.grid.line_or_topo_pos[line_id]
                line_ex = self.grid.line_ex_topo_pos[line_id]

                # Reconnect power lines as in reference topology
                line_status[line_id] = ref_line_status[line_id]
                topo_vect[line_or] = ref_topo_vect[line_or]
                topo_vect[line_ex] = ref_topo_vect[line_ex]

                # Reconnection amounts to 1 unitary action
                dist_status = dist_status + 1

        assert np.equal(topo_vect, -1).sum() == 0  # All element are connected

        dist_sub = 0
        for sub_id in range(self.grid.n_sub):
            sub_topology_mask = self.grid.substation_topology_mask[sub_id, :]
            sub_topo_vect = topo_vect[sub_topology_mask]
            ref_sub_topo_vect = ref_topo_vect[sub_topology_mask]

            sub_count = np.not_equal(
                sub_topo_vect, ref_sub_topo_vect
            ).sum()  # Count difference
            if sub_count > 0:
                # Reconfigure buses as in reference topology
                topo_vect[sub_topology_mask] = ref_sub_topo_vect

                # Substation bus reconfiguration amounts to 1 unitary action
                dist_sub = dist_sub + 1

        assert np.equal(
            topo_vect, ref_topo_vect
        ).all()  # Modified topology must be equal to reference topology

        dist = dist_status + dist_sub
        return dist, dist_status, dist_sub

    def print_agent(self, default=False):
        model = TopologyOptimizationDCOPF(name="Default", grid=self.grid)
        default_kwargs = model.get_model_parameters()

        pprint("\nAgent:", self.name, shift=36)
        if default:
            for arg in default_kwargs:
                model_arg = self.model_kwargs[arg] if arg in self.model_kwargs else "-"
                pprint(
                    f"  - {arg}:", "{:<10}".format(str(model_arg)), default_kwargs[arg]
                )
        else:
            for arg in self.model_kwargs:
                pprint(
                    f"  - {arg}:",
                    "{:<10}".format(str(self.model_kwargs[arg])),
                    default_kwargs[arg],
                )
        print("-" * 80)


class AgentDoNothing:
    def __init__(self, case, action_set):
        self.name = "Do-nothing Agent"

        self.env = case.env

        self.grid = GridDCOPF(
            case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p
        )

        self.actions = [self.env.action_space({})]

        self.reward = None
        self.obs_next = None

        self.actions, self.actions_info = action_set

    def set_kwargs(self, **kwargs):
        pass

    def act(self, observation, reward, done=False):
        action = self.actions[0]

        obs_next, reward, done, info = observation.simulate(action)
        self.reward = reward
        self.obs_next = obs_next

        return action

    def act_with_timing(self, observation, reward, done=False):
        timing = dict()

        start_solve = timer()
        action = self.actions[0]
        obs_next, reward, done, info = observation.simulate(action)
        self.reward = reward
        self.obs_next = obs_next
        timing["solve"] = timer() - start_solve

        return action, timing

    def get_reward(self):
        return self.reward

    def compare_with_observation(self, obs, verbose=False):
        res_gen = pd.DataFrame()
        res_gen["p_pu"] = self.grid.convert_mw_to_per_unit(self.obs_next.prod_p)
        res_gen["max_p_pu"] = self.grid.gen["max_p_pu"]
        res_gen["min_p_pu"] = self.grid.gen["min_p_pu"]

        res_gen["env_p_pu"] = self.grid.convert_mw_to_per_unit(obs.prod_p)
        res_gen["diff"] = np.divide(
            np.abs(res_gen["p_pu"] - res_gen["env_p_pu"]), res_gen["env_p_pu"] + 1e-9
        )

        res_line = pd.DataFrame()
        res_line["p_pu"] = self.grid.convert_mw_to_per_unit(
            self.obs_next.p_or
        )  # i_pu * v_pu * sqrt(3)
        res_line["max_p_pu"] = np.abs(
            np.divide(res_line["p_pu"], self.obs_next.rho + 1e-9)
        )

        res_line["env_p_pu"] = self.grid.convert_mw_to_per_unit(
            obs.p_or
        )  # i_pu * v_pu * sqrt(3)
        res_line["env_max_p_pu"] = np.abs(
            np.divide(res_line["env_p_pu"], obs.rho + 1e-9)
        )

        res_line["rho"] = self.obs_next.rho
        res_line["env_rho"] = obs.rho

        # Reactive/Active power ratio
        res_gen["env_q_pu"] = self.grid.convert_mw_to_per_unit(obs.prod_q)
        res_gen["env_gen_q_p"] = np.greater(obs.prod_p, 1e-9).astype(float) * np.abs(
            np.divide(obs.prod_q, obs.prod_p + 1e-9)
        )

        res_line["diff_p"] = np.abs(
            np.divide(
                res_line["p_pu"] - res_line["env_p_pu"], res_line["env_p_pu"] + 1e-9
            )
        )

        if verbose:
            print("GEN\n" + res_gen.to_string())
            print("LINE\n" + res_line.to_string())

        return res_line, res_gen

    @staticmethod
    def distance_to_ref_topology(topo_vect, line_status):
        return 0, 0, 0

    def print_agent(self, default=False):
        pprint("\nAgent:", self.name, shift=36)
        print("-" * 80)
