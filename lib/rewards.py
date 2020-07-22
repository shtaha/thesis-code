import numpy as np
import pandas as pd


class RewardL2RPN2019:
    @staticmethod
    def from_observation(obs):
        relative_flows = obs.rho
        relative_flows = np.minimum(relative_flows, 1.0)  # Clip if rho > 1.0

        line_scores = np.maximum(
            1.0 - relative_flows ** 2, np.zeros_like(relative_flows)
        )

        reward = line_scores.sum()
        return reward

    @staticmethod
    def from_mip_solution(result):
        def f(x):
            return 1.0 - np.square((1.0 - x))

        line_flow = pd.concat(
            [result["res_line"]["p_pu"], result["res_trafo"]["p_pu"]], ignore_index=True
        )
        max_line_flow = pd.concat(
            [result["res_line"]["max_p_pu"], result["res_trafo"]["max_p_pu"]],
            ignore_index=True,
        )

        relative_flows = np.abs(
            np.divide(line_flow, max_line_flow)
        )  # rho_l = abs(F_l / F_l^max)
        relative_flows = np.minimum(relative_flows, 1.0)  # Clip if rho > 1.0

        line_margins = np.maximum(0.0, 1.0 - relative_flows)  # Clip if margin < 0.0
        line_scores = f(line_margins)

        reward = line_scores.sum()
        return reward
