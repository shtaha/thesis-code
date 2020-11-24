from timeit import default_timer as timer

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from ..constants import Constants as Const
from ..dc_opf import (
    GridDCOPF,
    TopologyOptimizationDCOPF,
    MultistepTopologyDCOPF,
    SinglestepTopologyParameters,
    MultistepTopologyParameters,
    Forecasts,
)
from ..rewards import RewardL2RPN2019
from ..visualizer import pprint


def make_test_agent(
    agent_name,
    case,
    save_dir=None,
    verbose=False,
    delta_max_p_pu=0.10,
    horizon=2,
    **kwargs,
):
    action_set = case.generate_unitary_action_set(
        case, case_save_dir=save_dir, verbose=verbose
    )

    case_name = case.env.name

    if agent_name == "agent-multistep-mip":
        if "obj_lambda_action" not in kwargs:
            if "l2rpn_2019" in case_name:
                kwargs["obj_lambda_action"] = 0.07
            elif "rte_case5_example" in case_name:
                kwargs["obj_lambda_action"] = 0.006
            else:
                kwargs["obj_lambda_action"] = 0.05

        agent = AgentMultistepMIPTest(
            case=case, action_set=action_set, horizon=horizon, **kwargs
        )
    elif agent_name == "agent-mip":
        if "obj_lambda_action" not in kwargs:
            if "l2rpn_2019" in case_name:
                kwargs["obj_lambda_action"] = 0.07
            elif "rte_case5_example" in case_name:
                kwargs["obj_lambda_action"] = 0.006
            else:
                kwargs["obj_lambda_action"] = 0.05

        agent = AgentMIPTest(case=case, action_set=action_set, **kwargs)
    elif agent_name == "agent-mip-l2rpn":
        if "obj_lambda_action" not in kwargs:
            if "l2rpn_2019" in case_name:
                kwargs["obj_lambda_action"] = 0.07
            elif "rte_case5_example" in case_name:
                kwargs["obj_lambda_action"] = 0.006
            else:
                kwargs["obj_lambda_action"] = 0.05

        agent = AgentMIPTest(
            case=case,
            action_set=action_set,
            name="Agent MIP L2RPN",
            obj_reward_lin=True,
            obj_reward_max=False,
            **kwargs,
        )
    elif agent_name == "agent-mip-q":
        if "obj_lambda_action" not in kwargs:
            if "l2rpn_2019" in case_name:
                kwargs["obj_lambda_action"] = 0.02
            elif "rte_case5_example" in case_name:
                kwargs["obj_lambda_action"] = 0.006
            else:
                kwargs["obj_lambda_action"] = 0.05

        agent = AgentMIPTest(
            case=case,
            action_set=action_set,
            name="Agent MIP Q",
            obj_reward_quad=True,
            obj_reward_max=False,
            **kwargs,
        )
    elif agent_name == "do-nothing-agent":
        agent = AgentDoNothingTest(case=case, action_set=action_set)
    elif agent_name == "mixed_multi_agent":
        agent = AgentMixedMultistepTest(
            case=case, action_set=action_set, horizon=horizon, **kwargs
        )
    elif agent_name == "mixed_agent":
        agent = AgentMixedTest(case=case, action_set=action_set, **kwargs)
    elif agent_name == "augmented_agent":
        agent = AgentMIPAugmentedTest(
            case=case, action_set=action_set, delta_max_p_pu=delta_max_p_pu, **kwargs,
        )
    elif agent_name == "greedy_agent":
        agent = AgentGreedy(case=case, action_set=action_set)
    else:
        raise ValueError(f"Agent name {agent_name} is invalid.")

    return agent


def get_agent_color(agent_name):
    colors = Const.COLORS
    agent_names = [
        "do-nothing-agent",
        "agent-mip",
        "agent-multistep-mip",
        "agent-mip-l2rpn",
        "agent-mip-q",
    ]

    if agent_name in agent_names:
        color_id = agent_names.index(agent_name)
    else:
        color_id = len(colors) - 1

    color_id = color_id % len(colors)
    color = colors[color_id]
    return color


class BaseAgentTest:
    def __init__(self, name, case):
        self.name = name

        self.case = case
        self.env = case.env

        self.grid = GridDCOPF(
            case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p
        )

        # Grid precision
        if case.name == "Case RTE 5":
            self.grid.line["max_p_pu"] = [
                106.00150299072265625,
                38.867218017578125,
                28.26706695556640625,
                28.26706695556640625,
                106.0015106201171875,
                51.96152496337890625,
                51.96152496337890625,
                27.7128143310546875,
            ]
        elif case.name == "Case L2RPN 2019":
            self.grid.line["max_p_pu"] = [
                183.8731231689453125,
                73.7669219970703125,
                74.86887359619140625,
                65.43161773681640625,
                38.25582122802734375,
                82.47359466552734375,
                52.29061126708984375,
                54.646198272705078125,
                25.980762481689453125,
                44.721553802490234375,
                22.72505950927734375,
                17.978687286376953125,
                37.557476043701171875,
                67.636581,
                61.26263427734375,
                36.684833526611328125,
                30.32820892333984375,
                27.9899425506591796875,
                17.3205089569091796875,
                26.89875030517578125,
            ]

    def act(self, observation, reward, done=False):
        pass

    def reset(self, obs):
        pass

    def set_kwargs(self, **kwargs):
        pass

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
        pprint("\nAgent:", self.name, shift=36)
        print("-" * 80)


class AgentMIPTest(BaseAgentTest):
    """
        Agent class used for experimentation and testing.
    """

    def __init__(
        self,
        case,
        action_set,
        name="Agent MIP",
        reward_class=RewardL2RPN2019,
        **kwargs,
    ):
        BaseAgentTest.__init__(self, name=name, case=case)

        if "n_max_line_status_changed" not in kwargs:
            kwargs[
                "n_max_line_status_changed"
            ] = case.env.parameters.MAX_LINE_STATUS_CHANGED

        if "n_max_sub_changed" not in kwargs:
            kwargs["n_max_sub_changed"] = case.env.parameters.MAX_SUB_CHANGED

        if "n_max_timestep_overflow" not in kwargs:
            kwargs[
                "n_max_timestep_overflow"
            ] = case.env.parameters.NB_TIMESTEP_OVERFLOW_ALLOWED

        self.default_kwargs = kwargs
        self.model_kwargs = self.default_kwargs
        self.params = SinglestepTopologyParameters(**self.model_kwargs)

        self.forecasts = None
        self.reset(obs=None)

        self.model = None
        self.result = None

        self.reward_function = reward_class()
        self.actions, self.actions_info = action_set

    def set_kwargs(self, **kwargs):
        self.model_kwargs = {**self.default_kwargs, **kwargs}
        self.params = SinglestepTopologyParameters(**self.model_kwargs)

    def act(self, observation, reward, done=False):
        self._update(observation, reset=done)
        self.model = TopologyOptimizationDCOPF(
            self.case.env.name,
            grid=self.grid,
            forecasts=self.forecasts,
            base_unit_p=self.grid.base_unit_p,
            base_unit_v=self.grid.base_unit_v,
            params=self.params,
        )

        self.model.build_model()
        self.result = self.model.solve()

        action = self.grid.convert_mip_to_topology_vector(self.result, observation)[-1]
        return action

    def act_with_objectives(self, observation, reward, done=False):
        self._update(observation, reset=done)

        """
            Agent
        """
        self.model = TopologyOptimizationDCOPF(
            self.case.env.name,
            grid=self.grid,
            forecasts=self.forecasts,
            base_unit_p=self.grid.base_unit_p,
            base_unit_v=self.grid.base_unit_v,
            params=self.params,
        )

        self.model.build_model()

        self.result = self.model.solve()
        action = self.grid.convert_mip_to_topology_vector(self.result, observation)[-1]

        """
            Do-nothing agent
        """
        model_dn = TopologyOptimizationDCOPF(
            self.case.env.name,
            grid=self.grid,
            forecasts=self.forecasts,
            base_unit_p=self.grid.base_unit_p,
            base_unit_v=self.grid.base_unit_v,
            params=self.params,
        )
        model_dn.build_model_do_nothing()

        result_dn = model_dn.solve()
        action_dn = self.grid.convert_mip_to_topology_vector(result_dn, observation)[-1]

        # Check if Do-nothing
        assert action_dn == self.actions[0]

        model = self.model.model
        model_dn = model_dn.model

        info = dict(
            obj=pyo.value(model.objective),
            mu_max=pyo.value(model.mu_max),
            mu_gen=pyo.value(model.mu_gen),
            obj_dn=pyo.value(model_dn.objective),
            mu_max_dn=pyo.value(model_dn.mu_max),
            mu_gen_dn=pyo.value(model_dn.mu_gen),
            solution_status=self.result["solution_status"],
        )

        return action, info

    def act_with_timing(self, observation, reward, done=False):
        timing = dict()
        start_build = timer()
        self._update(observation, reset=done)

        self.model = TopologyOptimizationDCOPF(
            f"{self.case.env.name} DC OPF Topology Optimization",
            grid=self.grid,
            base_unit_p=self.grid.base_unit_p,
            base_unit_v=self.grid.base_unit_v,
            params=self.params,
        )
        self.model.build_model()
        timing["build"] = timer() - start_build

        start_solve = timer()
        self.result = self.model.solve(verbose=False)

        action = self.grid.convert_mip_to_topology_vector(self.result, observation)[-1]
        timing["solve"] = timer() - start_solve

        return action, timing

    def reset(self, obs):
        if self.params.forecasts:
            self.forecasts = Forecasts(
                env=self.env,
                t=self.env.chronics_handler.real_data.data.current_index,
                horizon=1,
            )

    def _update(self, obs, reset=False, verbose=False):
        if self.params.forecasts:
            self.forecasts.t = self.forecasts.t + 1
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
        res_line["env_p_pu"] = self.grid.convert_mw_to_per_unit(obs.p_or)
        res_line["env_max_p_pu"] = np.abs(
            np.divide(res_line["env_p_pu"], obs.rho + 1e-9)
        )

        res_line["rho"] = self.result["res_line"]["loading_percent"] / 100.0
        res_line["env_rho"] = obs.rho

        res_line["env_q_pu"] = self.grid.convert_mw_to_per_unit(
            obs.q_or
        )  # i_pu * v_pu * sqrt(3) * sin(fi)

        # Reactive/Active power ratio
        res_gen["env_q_pu"] = self.grid.convert_mw_to_per_unit(obs.prod_q)
        res_gen["env_gen_q_p"] = np.greater(obs.prod_p, 1e-9).astype(float) * np.abs(
            np.divide(obs.prod_q, obs.prod_p + 1e-9)
        )

        if verbose:
            print("GEN\n" + res_gen.to_string())
            print("LINE\n" + res_line.to_string())

            # Grid precision - Manual
            # from decimal import Decimal
            # max_p = list(res_line["env_max_p_pu"].values)
            # print("[" + ", ".join([str(Decimal(float(d))) for d in max_p]) + "]")

        return res_line, res_gen

    def print_agent(self, default=False):
        default_kwargs = SinglestepTopologyParameters().to_dict()

        pprint("\nAgent:", self.name, shift=36)
        if default:
            for arg in default_kwargs:
                model_arg = self.model_kwargs[arg] if arg in self.model_kwargs else "-"
                pprint(
                    f"  - {arg}:", "{:<10}".format(str(model_arg)), default_kwargs[arg]
                )
        else:
            for arg in self.model_kwargs:
                if arg in default_kwargs:
                    pprint(
                        f"  - {arg}:",
                        "{:<10}".format(str(self.model_kwargs[arg])),
                        default_kwargs[arg],
                    )
        print("-" * 80)


class AgentDoNothingTest(BaseAgentTest):
    def __init__(self, case, action_set):
        BaseAgentTest.__init__(self, name="Do-nothing Agent", case=case)

        self.model_kwargs = dict()

        self.reward = None
        self.obs_next = None
        self.done = None
        self.result = None

        self.actions, self.actions_info = action_set

    def act(self, observation, reward, done=False):
        self._update(observation, reset=done)
        action = self.actions[0]

        obs_next, reward, done, info = observation.simulate(action)
        self.reward = reward
        self.obs_next = obs_next
        self.done = done

        return action

    def act_with_objectives(self, observation, reward, done=False):
        return self.act(observation, reward, done=done), {}

    def act_with_timing(self, observation, reward, done=False):
        timing = dict()
        start_solve = timer()
        action = self.act(observation, reward, done)
        timing["solve"] = timer() - start_solve

        return action, timing

    def _update(self, obs, reset=False, verbose=False):
        self.grid.update(obs, reset=reset, verbose=verbose)

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


class AgentMultistepMIPTest(BaseAgentTest):
    """
        Agent class used for experimentation and testing.
    """

    def __init__(
        self,
        case,
        action_set,
        name="Agent Multistep MIP",
        reward_class=RewardL2RPN2019,
        **kwargs,
    ):
        BaseAgentTest.__init__(self, name=name, case=case)

        if "n_max_line_status_changed" not in kwargs:
            kwargs[
                "n_max_line_status_changed"
            ] = case.env.parameters.MAX_LINE_STATUS_CHANGED

        if "n_max_sub_changed" not in kwargs:
            kwargs["n_max_sub_changed"] = case.env.parameters.MAX_SUB_CHANGED

        if "n_max_timestep_overflow" not in kwargs:
            kwargs[
                "n_max_timestep_overflow"
            ] = case.env.parameters.NB_TIMESTEP_OVERFLOW_ALLOWED

        self.default_kwargs = kwargs
        self.model_kwargs = self.default_kwargs
        self.params = MultistepTopologyParameters(**self.model_kwargs)

        self.forecasts = None
        self.reset(obs=None)

        self.model = None
        self.result = None

        self.reward_function = reward_class()
        self.actions, self.actions_info = action_set

    def set_kwargs(self, **kwargs):
        self.model_kwargs = {**self.default_kwargs, **kwargs}
        self.params = MultistepTopologyParameters(**self.model_kwargs)

    def act(self, observation, reward, done=False):
        self._update(observation, reset=done)
        self.model = MultistepTopologyDCOPF(
            self.case.env.name,
            grid=self.grid,
            forecasts=self.forecasts,
            base_unit_p=self.grid.base_unit_p,
            base_unit_v=self.grid.base_unit_v,
            params=self.params,
        )

        self.model.build_model()
        self.result = self.model.solve()

        action = self.grid.convert_mip_to_topology_vector(self.result, observation)[-1]
        return action

    def act_with_objectives(self, observation, reward, done=False):
        self._update(observation, reset=done)

        """
            Agent
        """
        self.model = MultistepTopologyDCOPF(
            self.case.env.name,
            grid=self.grid,
            forecasts=self.forecasts,
            base_unit_p=self.grid.base_unit_p,
            base_unit_v=self.grid.base_unit_v,
            params=self.params,
        )

        self.model.build_model()
        self.result = self.model.solve()
        action = self.grid.convert_mip_to_topology_vector(self.result, observation)[-1]

        """
            Do-nothing agent
        """
        model_dn = MultistepTopologyDCOPF(
            self.case.env.name,
            grid=self.grid,
            forecasts=self.forecasts,
            base_unit_p=self.grid.base_unit_p,
            base_unit_v=self.grid.base_unit_v,
            params=self.params,
        )
        model_dn.build_model_do_nothing()
        result_dn = model_dn.solve()
        action_dn = self.grid.convert_mip_to_topology_vector(result_dn, observation)[-1]

        # Check if Do-nothing
        assert action_dn == self.actions[0]

        model = self.model.model
        model_dn = model_dn.model

        info = dict(
            obj=pyo.value(model.objective),
            mu_max=self.model._access_pyomo_variable(model.mu_max),
            mu_gen=self.model._access_pyomo_variable(model.mu_gen),
            obj_dn=pyo.value(model_dn.objective),
            mu_max_dn=self.model._access_pyomo_variable(model_dn.mu_max),
            mu_gen_dn=self.model._access_pyomo_variable(model_dn.mu_gen),
            solution_status=self.result["solution_status"],
        )

        return action, info

    def act_with_timing(self, observation, reward, done=False):
        timing = dict()
        start_build = timer()
        self._update(observation, reset=done)

        self.model = MultistepTopologyDCOPF(
            f"{self.case.env.name} DC OPF Topology Optimization",
            grid=self.grid,
            forecasts=self.forecasts,
            base_unit_p=self.grid.base_unit_p,
            base_unit_v=self.grid.base_unit_v,
            params=self.params,
        )
        self.model.build_model()
        timing["build"] = timer() - start_build

        start_solve = timer()
        self.result = self.model.solve(verbose=False)

        action = self.grid.convert_mip_to_topology_vector(self.result, observation)[-1]
        timing["solve"] = timer() - start_solve

        return action, timing

    def reset(self, obs):
        if self.params.forecasts:
            self.forecasts = Forecasts(
                env=self.env,
                t=self.env.chronics_handler.real_data.data.current_index,
                horizon=self.params.horizon,
            )

    def _update(self, obs, reset=False, verbose=False):
        if self.params.forecasts:
            self.forecasts.t = self.forecasts.t + 1
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

        res_line["max_p_pu_ac"] = np.sqrt(
            np.abs(
                np.square(res_line["max_p_pu"])
                - np.square(self.grid.convert_mw_to_per_unit(obs.q_or))
            )
        )

        if verbose:
            print("GEN\n" + res_gen.to_string())
            print("LINE\n" + res_line.to_string())

        return res_line, res_gen

    def print_agent(self, default=False):
        default_kwargs = MultistepTopologyParameters().to_dict()

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


class AgentMixedTest(BaseAgentTest):
    def __init__(self, case, action_set, reward_class=RewardL2RPN2019, **kwargs):
        BaseAgentTest.__init__(self, name="Mixed agent", case=case)

        self.agent_mip = AgentMIPTest(
            case=case, action_set=action_set, reward_class=reward_class, **kwargs
        )

        self.agent_dn = AgentDoNothingTest(case=case, action_set=action_set)
        self.model_kwargs = self.agent_mip.model_kwargs

        self.agent = None
        self.actions, self.actions_info = action_set

    def set_kwargs(self, **kwargs):
        self.agent_mip.set_kwargs(**kwargs)

    def act(self, observation, reward, done=False):
        self.agent = self.agent_dn
        self.agent_dn.act(observation, reward, done)
        action = self.agent_dn.actions[0]

        # If do-nothing agent fails, use MIP
        # cond = self.agent_dn.done
        cond = self.agent_dn.obs_next.rho.max() > 0.90
        if cond:
            self.agent = self.agent_mip
            action = self.agent_mip.act(observation, reward, done)

        return action

    def act_with_timing(self, observation, reward, done=False):
        timing = dict()

        start_solve = timer()
        action = self.act(observation, reward, done)
        timing["solve"] = timer() - start_solve

        return action, timing

    def get_reward(self):
        return self.agent.get_reward()

    def compare_with_observation(self, obs, verbose=False):
        return self.agent.compare_with_observation(obs, verbose)

    def distance_to_ref_topology(self, topo_vect, line_status):
        return self.agent.distance_to_ref_topology(topo_vect, line_status)

    def print_agent(self, default=False):
        pprint("\nAgent:", self.name, shift=36)
        self.agent_mip.print_agent(default)
        self.agent_dn.print_agent(default)


class AgentGreedy(BaseAgentTest):
    def __init__(self, case, action_set):
        BaseAgentTest.__init__(self, name="Greedy agent", case=case)

        if case.name != "Case RTE 5":
            raise ValueError(f"Action space for {case.name} is too big.")

        self.model_kwargs = dict()

        self.reward = None
        self.obs_next = None
        self.done = None

        self.actions, self.actions_info = action_set

    def act(self, observation, reward, done=False):
        self._update(observation, reset=done)

        best_action = self.actions[0]
        best_simulation = (None, -np.inf, None, None)
        for action_id, action in enumerate(self.actions):
            simulation = observation.simulate(action)

            if simulation[1] > best_simulation[1] + 1e-3:
                best_simulation = simulation
                best_action = action

        action = best_action
        obs_next, reward, done, info = best_simulation

        self.reward = reward
        self.obs_next = obs_next
        self.done = done

        return action

    def act_with_timing(self, observation, reward, done=False):
        timing = dict()
        start_solve = timer()
        action = self.act(observation, reward, done)
        timing["solve"] = timer() - start_solve

        return action, timing

    def _update(self, obs, reset=False, verbose=False):
        self.grid.update(obs, reset=reset, verbose=verbose)

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


class AgentMIPAugmentedTest(AgentMIPTest):
    def __init__(
        self,
        case,
        action_set,
        reward_class=RewardL2RPN2019,
        delta_max_p_pu=0.10,
        **kwargs,
    ):
        AgentMIPTest.__init__(
            self, case, action_set, reward_class=reward_class, **kwargs
        )

        self.name = "Augmented Agent"

        self.grid_inc = GridDCOPF(
            case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p
        )
        self.grid.line["max_p_pu"] = (1.0 + delta_max_p_pu) * self.grid.line["max_p_pu"]

        self.model_inc = None

        self.delta_max_p_pu = delta_max_p_pu

    def act(self, observation, reward, done=False):
        self._update(observation, reset=done)

        self.model = TopologyOptimizationDCOPF(
            f"{self.case.env.name} DC OPF Topology Optimization - {self.delta_max_p_pu}",
            grid=self.grid,
            base_unit_p=self.grid.base_unit_p,
            base_unit_v=self.grid.base_unit_v,
            **self.model_kwargs,
        )
        self.model_inc = TopologyOptimizationDCOPF(
            f"{self.case.env.name} DC OPF Topology Optimization - Incremented {self.delta_max_p_pu}",
            grid=self.grid_inc,
            base_unit_p=self.grid.base_unit_p,
            base_unit_v=self.grid.base_unit_v,
            **self.model_kwargs,
        )

        best_model = None
        best_result = None
        best_action = self.actions[0]
        best_simulation = (None, -np.inf, None, None)
        for model in [self.model, self.model_inc]:
            model.build_model()
            result = self.model.solve()
            action = self.grid.convert_mip_to_topology_vector(result, observation)[-1]

            simulation = observation.simulate(action)

            if simulation[1] > best_simulation[1] + 1e-3:
                best_model = model
                best_simulation = simulation
                best_action = action
                best_result = result

        print(best_model.name)

        action = best_action
        self.result = best_result
        return action

    def _update(self, obs, reset=False, verbose=False):
        self.grid.update(obs, reset=reset, verbose=verbose)
        self.grid_inc.update(obs, reset=reset, verbose=verbose)


class AgentMixedMultistepTest(BaseAgentTest):
    def __init__(self, case, action_set, reward_class=RewardL2RPN2019, **kwargs):
        BaseAgentTest.__init__(self, name="Mixed Multistep agent", case=case)

        self.agent_mip = AgentMultistepMIPTest(
            case=case, action_set=action_set, reward_class=reward_class, **kwargs
        )

        self.agent_dn = AgentDoNothingTest(case=case, action_set=action_set)
        self.name = "Mixed Multistep agent"

        self.model_kwargs = self.agent_mip.model_kwargs

        self.agent = None
        self.actions, self.actions_info = action_set

    def set_kwargs(self, **kwargs):
        self.agent_mip.set_kwargs(**kwargs)

    def act(self, observation, reward, done=False):
        self.agent = self.agent_dn
        self.agent_dn.act(observation, reward, done)
        action = self.agent_dn.actions[0]

        # If do-nothing agent fails, use MIP
        # cond = self.agent_dn.done
        cond = self.agent_dn.obs_next.rho.max() > 0.95
        if cond:
            self.agent = self.agent_mip
            # pprint("    - Agent:", self.agent.name)
            action = self.agent_mip.act(observation, reward, done)

        return action

    def act_with_timing(self, observation, reward, done=False):
        timing = dict()

        start_solve = timer()
        action = self.act(observation, reward, done)
        timing["solve"] = timer() - start_solve

        return action, timing

    def get_reward(self):
        return self.agent.get_reward()

    def compare_with_observation(self, obs, verbose=False):
        return self.agent.compare_with_observation(obs, verbose)

    def distance_to_ref_topology(self, topo_vect, line_status):
        return self.agent.distance_to_ref_topology(topo_vect, line_status)

    def print_agent(self, default=False):
        pprint("\nAgent:", self.name, shift=36)
        self.agent_mip.print_agent(default)
        self.agent_dn.print_agent(default)
