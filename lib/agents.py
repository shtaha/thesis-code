import numpy as np

from lib.dc_opf import (
    GridDCOPF,
    TopologyOptimizationDCOPF,
)
from lib.rewards import RewardL2RPN2019


class AgentMIP:
    def __init__(
        self,
        case,
        n_max_line_status_changed=1,
        n_max_sub_changed=1,
        reward_class=RewardL2RPN2019,
    ):
        self.case = case

        self.grid = GridDCOPF(
            case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p
        )
        self.model = None
        self.result = None

        self.n_max_line_status_changed = n_max_line_status_changed
        self.n_max_sub_changed = n_max_sub_changed

        self.reward_function = reward_class()

    def act(self, obs, **kwargs):
        self.model = TopologyOptimizationDCOPF(
            f"{self.case.env.name} DC OPF Topology Optimization",
            grid=self.grid,
            base_unit_p=self.grid.base_unit_p,
            base_unit_v=self.grid.base_unit_v,
            n_max_line_status_changed=self.n_max_line_status_changed,
            n_max_sub_changed=self.n_max_sub_changed,
        )
        self.model.build_model(**kwargs)
        self.result = self.model.solve(verbose=False)

        return self.grid.convert_mip_to_topology_vector(self.result, obs)

    def update(self, obs, reset=False, verbose=False):
        self.grid.update(obs, reset=reset, verbose=verbose)

    def compare_with_observation(self, obs, verbose=False):
        res_gen = self.result["res_gen"][["min_p_pu", "p_pu", "max_p_pu"]].copy()
        res_gen["env_p_pu"] = self.grid.convert_mw_to_per_unit(obs.prod_p)
        res_gen["diff"] = np.divide(
            np.abs(res_gen["p_pu"] - res_gen["env_p_pu"]), res_gen["env_p_pu"] + 1e-9
        )

        res_line = self.result["res_line"][["p_pu", "max_p_pu"]].copy()
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

        res_line["diff_p"] = np.abs(
            np.divide(
                res_line["p_pu"] - res_line["env_p_pu"], res_line["env_p_pu"] + 1e-9
            )
        )
        res_line["diff_rho"] = np.abs(
            np.divide(res_line["rho"] - res_line["env_rho"], res_line["env_rho"] + 1e-9)
        )

        max_rho = res_line["rho"].max()
        env_max_rho = res_line["env_rho"].max()

        if verbose:
            print("GEN\n" + res_gen.to_string())
            print("LINE\n" + res_line.to_string())
            print("RHO:\t{}\t{}".format(max_rho, env_max_rho))
