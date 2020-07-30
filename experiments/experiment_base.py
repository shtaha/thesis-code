from lib.visualizer import pprint


class ExperimentBase:
    @staticmethod
    def print_experiment(exp_name):
        print("\n" + "-" * 80)
        pprint("Experiment:", exp_name)
        print("-" * 80)

    @staticmethod
    def _get_case_name(case):
        env_pf = "AC"
        if case.env.parameters.ENV_DC:
            env_pf = "DC"

        return f"{case.name} ({env_pf})"
