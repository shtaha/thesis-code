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
