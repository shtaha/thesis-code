from timeit import default_timer as timer

import numpy as np
import pandas as pd

from lib.dc_opf import (
    GridDCOPF,
    TopologyOptimizationDCOPF,
)
from lib.rewards import RewardL2RPN2019


class AgentMIPTest:
    """
        Agent class used for experimentation and testing.
    """

    def __init__(
        self,
        case,
        action_set,
        n_max_line_status_changed=1,
        n_max_sub_changed=1,
        reward_class=RewardL2RPN2019,
    ):
        self.name = "Agent MIP"

        self.case = case
        self.env = case.env

        self.grid = GridDCOPF(
            case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p
        )
        self.model = None
        self.result = None

        self.n_max_line_status_changed = n_max_line_status_changed
        self.n_max_sub_changed = n_max_sub_changed

        self.reward_function = reward_class()

        self.actions, self.actions_info = action_set

    def act(self, obs, done, **kwargs):
        self._update(obs, reset=done)
        self.model = TopologyOptimizationDCOPF(
            f"{self.case.env.name} DC OPF Topology Optimization",
            grid=self.grid,
            base_unit_p=self.grid.base_unit_p,
            base_unit_v=self.grid.base_unit_v,
            n_max_line_status_changed=self.n_max_line_status_changed,
            n_max_sub_changed=self.n_max_sub_changed,
        )
        self.model.build_model(**kwargs)
        self.result = self.model.solve()

        action = self.grid.convert_mip_to_topology_vector(self.result, obs)[-1]
        return action

    def act_with_timing(
        self,
        obs,
        done,
        tol=0.001,
        solver_name="gurobi",
        warm_start=False,
        n_max_line_status_changed=1,
        n_max_sub_changed=1,
        verbose=False,
        **kwargs,
    ):
        timing = dict()

        start_build = timer()
        self._update(obs, reset=done)
        timing["update"] = timer() - start_build

        self.model = TopologyOptimizationDCOPF(
            f"{self.case.env.name} DC OPF Topology Optimization",
            grid=self.grid,
            base_unit_p=self.grid.base_unit_p,
            base_unit_v=self.grid.base_unit_v,
            solver_name=solver_name,
            n_max_line_status_changed=n_max_line_status_changed,
            n_max_sub_changed=n_max_sub_changed,
        )
        self.model.build_model(**kwargs)
        timing["build"] = timer() - start_build

        start_solve = timer()
        self.result = self.model.solve(tol=tol, verbose=verbose, warm_start=warm_start)
        action = self.grid.convert_mip_to_topology_vector(self.result, obs)[-1]
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

        # res_line["env_i_pu"] = self.convert_a_to_per_unit(obs.a_or)
        # res_line["env_max_i_pu"] = np.abs(
        #     np.divide(res_line["env_i_pu"], obs.rho + 1e-9)
        # )
        # res_line["env_v_pu"] = self.convert_kv_to_per_unit(obs.v_or)
        #
        # res_line["ratio"] = np.divide(res_line["max_p_pu"], res_line["env_max_p_pu"])

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

    def act(self, obs, done, **kwargs):
        action = self.actions[0]

        obs_next, reward, done, info = obs.simulate(action)
        self.reward = reward
        self.obs_next = obs_next

        return action

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
