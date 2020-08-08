import sys
from abc import ABC

from grid2op.Parameters import Parameters


class CaseParameters(Parameters):
    def __init__(self, case_name, env_dc=False):
        Parameters.__init__(self, parameters_path=None)

        param_dict = self._get_param_dict(case_name=case_name)

        self.init_from_dict(dict_=param_dict)
        if env_dc:
            self.ENV_DC = env_dc
            self.FORECAST_DC = env_dc

    @staticmethod
    def _get_param_dict(case_name):
        if case_name in ["rte_case5", "rte_case5_example"]:
            param_dict = {
                "NO_OVERFLOW_DISCONNECTION": False,
                "IGNORE_MIN_UP_DOWN_TIME": True,
                "ALLOW_DISPATCH_GEN_SWITCH_OFF": True,
                "NB_TIMESTEP_OVERFLOW_ALLOWED": 2,
                "NB_TIMESTEP_RECONNECTION": 10,
                "HARD_OVERFLOW_THRESHOLD": 2.0,
                "ENV_DC": False,
                "FORECAST_DC": False,
                "MAX_SUB_CHANGED": 1,
                "MAX_LINE_STATUS_CHANGED": 1,
                "NB_TIMESTEP_COOLDOWN_LINE": 0,
                "NB_TIMESTEP_COOLDOWN_SUB": 0,
            }
        elif case_name in ["l2rpn2019", "l2rpn_2019"]:
            param_dict = {
                "NO_OVERFLOW_DISCONNECTION": False,
                "IGNORE_MIN_UP_DOWN_TIME": True,
                "ALLOW_DISPATCH_GEN_SWITCH_OFF": True,
                "NB_TIMESTEP_OVERFLOW_ALLOWED": 2,
                "NB_TIMESTEP_RECONNECTION": 10,
                "HARD_OVERFLOW_THRESHOLD": 2.0,
                "ENV_DC": False,
                "FORECAST_DC": False,
                "MAX_SUB_CHANGED": 1,
                "MAX_LINE_STATUS_CHANGED": 1,
                "NB_TIMESTEP_COOLDOWN_LINE": 0,
                "NB_TIMESTEP_COOLDOWN_SUB": 0,
            }
        elif case_name in ["l2rpn2020", "l2rpn_wcci_2020", "l2rpn_2020"]:
            param_dict = {
                "NO_OVERFLOW_DISCONNECTION": False,
                "IGNORE_MIN_UP_DOWN_TIME": True,
                "ALLOW_DISPATCH_GEN_SWITCH_OFF": True,
                "NB_TIMESTEP_OVERFLOW_ALLOWED": 3,
                "NB_TIMESTEP_RECONNECTION": 12,
                "HARD_OVERFLOW_THRESHOLD": 200.0,
                "ENV_DC": False,
                "FORECAST_DC": False,
                "MAX_SUB_CHANGED": 1,
                "MAX_LINE_STATUS_CHANGED": 1,
                "NB_TIMESTEP_COOLDOWN_LINE": 3,
                "NB_TIMESTEP_COOLDOWN_SUB": 3,
            }
        else:
            raise ValueError(f"Invalid case name. Case {case_name} does not exist.")

        return param_dict


class AbstractParameters(ABC):
    def to_dict(self):
        return self.__dict__


class SolverParameters(AbstractParameters):
    def __init__(
        self, solver_name="gurobi", tol=0.01, warm_start=False,
    ):
        if sys.platform != "win32":
            solver_name = "glpk"

        self.solver_name = solver_name
        self.tol = tol
        self.warm_start = warm_start


class StandardParameters(SolverParameters):
    def __init__(self, delta_max=0.5, **kwargs):
        SolverParameters.__init__(self, **kwargs)
        self.delta_max = delta_max


class LineSwitchingParameters(StandardParameters):
    def __init__(
        self,
        n_max_line_status_changed=1,
        big_m=True,
        gen_cost=True,
        line_margin=True,
        **kwargs,
    ):
        StandardParameters.__init__(self, **kwargs)

        self.n_max_line_status_changed = n_max_line_status_changed

        self.big_m = big_m

        self.gen_cost = gen_cost
        self.line_margin = line_margin


class SinglestepTopologyParameters(StandardParameters):
    def __init__(
        self,
        forecasts=True,
        n_max_line_status_changed=1,
        n_max_sub_changed=1,
        allow_onesided_disconnection=True,
        allow_onesided_reconnection=False,
        symmetry=True,
        requirement_at_least_two=True,
        requirement_balance=True,
        switching_limits=True,
        cooldown=True,
        unitary_action=True,
        gen_cost=False,
        lin_line_margins=True,
        quad_line_margins=False,
        lambda_gen=10.0,
        lin_gen_penalty=True,
        quad_gen_penalty=False,
        lambda_action=0.0,
        **kwargs,
    ):
        StandardParameters.__init__(self, **kwargs)

        self.forecasts = forecasts

        self.n_max_line_status_changed = n_max_line_status_changed
        self.n_max_sub_changed = n_max_sub_changed

        self.allow_onesided_disconnection = allow_onesided_disconnection
        self.allow_onesided_reconnection = allow_onesided_reconnection
        self.symmetry = symmetry
        self.requirement_at_least_two = requirement_at_least_two
        self.requirement_balance = requirement_balance

        self.switching_limits = switching_limits
        self.cooldown = cooldown
        self.unitary_action = unitary_action

        self.gen_cost = gen_cost
        self.lin_line_margins = lin_line_margins
        self.quad_line_margins = quad_line_margins
        self.lambda_gen = lambda_gen
        self.lin_gen_penalty = lin_gen_penalty
        self.quad_gen_penalty = quad_gen_penalty
        self.lambda_action = lambda_action


class MultistepTopologyParameters(SinglestepTopologyParameters):
    def __init__(
        self, horizon=2, **kwargs,
    ):
        SinglestepTopologyParameters.__init__(self, **kwargs)
        self.horizon = horizon
