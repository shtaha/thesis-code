import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from lib.constants import Constants as Const
from lib.visualizer import print_info, print_action, pprint
from .experiment_base import ExperimentBase


class ExperimentDCOPFTiming(ExperimentBase):
    def compare_by_solver_and_parts(
        self, case, agent, save_dir, solver_names, n_bins=25, **kwargs,
    ):
        file_name = "solvers"
        case_name = self._get_case_name(case)

        self.print_experiment("Timing - Solver and Parts")

        data_dict = dict()
        for solver_name in solver_names:
            data_dict[solver_name] = self._runner_timing(
                env=case.env, agent=agent, solver_name=solver_name, **kwargs,
            )

        self._plot_and_save(
            times=[data_dict[param]["solve"] for param in data_dict],
            labels=[param.capitalize() for param in data_dict],
            n_bins=n_bins,
            title=f"{case_name} - MIP solver comparison",
            legend_title="Solver",
            save_path=os.path.join(save_dir, file_name),
        )

        data_parts = data_dict[solver_names[0]]
        self._plot_and_save(
            times=[data_parts[part] for part in data_parts.columns],
            labels=[part.capitalize() for part in data_parts.columns],
            n_bins=n_bins,
            title=f"{case_name} - Step parts comparison",
            legend_title="Part",
            save_path=os.path.join(save_dir, "parts"),
        )

        self._save_csv(
            data_dict=data_dict, save_path=os.path.join(save_dir, file_name + ".csv")
        )

    def compare_by_tolerance(self, case, agent, save_dir, tols, n_bins=25, **kwargs):
        file_name = "tolerances"
        case_name = self._get_case_name(case)

        self.print_experiment("Timing - Tolerances")

        data_dict = dict()
        for tol in tols:
            data_dict[str(tol)] = self._runner_timing(
                env=case.env, agent=agent, tol=tol, **kwargs,
            )

        self._plot_and_save(
            times=[data_dict[param]["solve"] for param in data_dict],
            labels=[param for param in data_dict],
            n_bins=n_bins,
            title=f"{case_name} - Duality gap tolerance comparison",
            legend_title="Tolerance",
            save_path=os.path.join(save_dir, file_name),
        )

        self._save_csv(
            data_dict=data_dict, save_path=os.path.join(save_dir, file_name + ".csv")
        )

    def compare_by_delta_max(self, case, agent, save_dir, deltas, n_bins=25, **kwargs):
        file_name = "deltas"
        case_name = self._get_case_name(case)

        self.print_experiment("Timing - Bounds on bus voltage angles")

        data_dict = dict()
        for delta_max in deltas:
            data_dict[str(delta_max)] = self._runner_timing(
                env=case.env, agent=agent, delta_max=delta_max, **kwargs,
            )

        self._plot_and_save(
            times=[data_dict[param]["solve"] for param in data_dict],
            labels=[param for param in data_dict],
            n_bins=n_bins,
            title=f"{case_name} - Bounds on bus voltage angles",
            legend_title=r"$\delta^{\text{max}}$",
            save_path=os.path.join(save_dir, file_name),
        )

        self._save_csv(
            data_dict=data_dict, save_path=os.path.join(save_dir, file_name + ".csv")
        )

    def compare_by_switching_limits(
        self, case, agent, save_dir, switch_limits, n_bins=25, **kwargs,
    ):
        file_name = "limits"
        case_name = self._get_case_name(case)

        self.print_experiment("Timing - Maximum switching limit")

        data_dict = dict()
        for limits in switch_limits:
            n_max_line_status_changed, n_max_sub_changed = limits
            limits_str = f"{n_max_line_status_changed}-{n_max_sub_changed}"

            data_dict[limits_str] = self._runner_timing(
                env=case.env,
                agent=agent,
                n_max_line_status_changed=n_max_line_status_changed,
                n_max_sub_changed=n_max_sub_changed,
                **kwargs,
            )

        self._plot_and_save(
            times=[data_dict[param]["solve"] for param in data_dict],
            labels=[param for param in data_dict],
            n_bins=n_bins,
            title=f"{case_name} - Maximum switching limit comparison",
            legend_title=r"$\alpha > \beta$" r"$n_\mathcal{P}$-$n_\mathcal{P}$",
            save_path=os.path.join(save_dir, file_name),
        )

        self._save_csv(
            data_dict=data_dict, save_path=os.path.join(save_dir, file_name + ".csv")
        )

    def compare_by_constraint_activations(
        self, case, agent, save_dir, constraint_activations, n_bins=25, **kwargs,
    ):
        file_name = "activations"
        case_name = self._get_case_name(case)

        self.print_experiment("Timing - Constraint Activations")

        data_dict = dict()
        for act_dict, act_str in constraint_activations:
            data_dict[act_str] = self._runner_timing(
                env=case.env, agent=agent, **act_dict, **kwargs,
            )

        self._plot_and_save(
            times=[data_dict[param]["solve"] for param in data_dict],
            labels=[param for param in data_dict],
            n_bins=n_bins,
            title=f"{case_name} - Constraint activations comparison",
            legend_title="Constraint deactivation",
            save_path=os.path.join(save_dir, file_name),
        )

        self._save_csv(
            data_dict=data_dict, save_path=os.path.join(save_dir, file_name + ".csv")
        )

    def compare_by_objective(
        self, case, agent, save_dir, objectives, n_bins=25, **kwargs,
    ):
        file_name = "objectives"
        case_name = self._get_case_name(case)

        self.print_experiment("Timing - Objective")

        data_dict = dict()
        for obj_dict, obj_str in objectives:
            data_dict[obj_str] = self._runner_timing(
                env=case.env, agent=agent, **obj_dict, **kwargs,
            )

        self._plot_and_save(
            times=[data_dict[param]["solve"] for param in data_dict],
            labels=[param for param in data_dict],
            n_bins=n_bins,
            title=f"{case_name} - Objective function comparison",
            legend_title="Objective modification",
            save_path=os.path.join(save_dir, file_name),
        )

        self._save_csv(
            data_dict=data_dict, save_path=os.path.join(save_dir, file_name + ".csv")
        )

    def compare_by_warmstart(
        self, case, agent, save_dir, n_bins=25, **kwargs,
    ):
        file_name = "warmstart"
        case_name = self._get_case_name(case)

        self.print_experiment("Timing - Warm Start")

        data_dict = dict()
        for warm_start in [True, False]:
            data_dict[str(warm_start)] = self._runner_timing(
                env=case.env, agent=agent, warm_start=warm_start, **kwargs,
            )

        self._plot_and_save(
            times=[data_dict[param]["solve"] for param in data_dict],
            labels=[param for param in data_dict],
            n_bins=n_bins,
            title=f"{case_name} - Solver warm start comparison",
            legend_title="Warm start",
            save_path=os.path.join(save_dir, file_name),
        )

        self._save_csv(
            data_dict=data_dict, save_path=os.path.join(save_dir, file_name + ".csv")
        )

    def compare_by_lambda(self, case, agent, save_dir, lambdas, n_bins=25, **kwargs):
        file_name = "lambdas"
        case_name = self._get_case_name(case)

        self.print_experiment("Timing - Penalty scaling")

        data_dict = dict()
        for lambd in lambdas:
            data_dict[str(lambd)] = self._runner_timing(
                env=case.env, agent=agent, lambd=lambd, **kwargs,
            )

        self._plot_and_save(
            times=[data_dict[param]["solve"] for param in data_dict],
            labels=[param for param in data_dict],
            n_bins=n_bins,
            title=f"{case_name} - Penalty scaling comparison",
            legend_title="Regularization parameter",
            save_path=os.path.join(save_dir, file_name),
        )

        self._save_csv(
            data_dict=data_dict, save_path=os.path.join(save_dir, file_name + ".csv")
        )

    @staticmethod
    def _runner_timing(
        env, agent, n_timings=100, verbose=False, **kwargs,
    ):
        np.random.seed(0)
        env.seed(0)

        agent.set_kwargs(**kwargs)
        agent.print_agent(default=True)

        timings = []

        done = False
        obs = env.reset()
        for t in range(n_timings):
            action, timing = agent.act_with_timing(obs, done)

            obs_next, reward, done, info = env.step(action)

            pprint("Step:", t)
            if verbose:
                print_action(action)
                print_info(info, done, reward)

            obs = obs_next
            if done:
                print("DONE\n")
                obs = env.reset()

            timings.append(timing)

        return pd.DataFrame(timings)

    @staticmethod
    def _plot_and_save(
        times, labels, n_bins=25, title=None, legend_title=None, save_path=None,
    ):
        fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
        ax.set_title(title)
        for time, label in zip(times, labels):
            sns.distplot(
                time,
                label=label,
                ax=ax,
                bins=n_bins,
                hist=True,
                kde=True,
                norm_hist=True,
            )

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("PDF")
        ax.set_xlim(left=0)

        ax.legend(title=legend_title)

        fig.show()
        if save_path:
            fig.savefig(save_path)

    @staticmethod
    def _save_csv(data_dict, save_path):
        data = pd.DataFrame(
            columns=["params", "total", "update", "build", "solve", "step"]
        )
        for key in data_dict:
            sub_data = data_dict[key]
            sub_data["params"] = key
            data = data.append(sub_data)

        if save_path:
            file_extension = os.path.splitext(save_path)[1]
            if not file_extension:
                save_path = save_path + ".csv"

            data.to_csv(save_path)
