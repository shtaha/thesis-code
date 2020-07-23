import os
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from lib.visualizer import print_info, print_action, pprint
from lib.constants import Constants as Const


class ExperimentDCOPFTiming:
    @staticmethod
    def _plot_and_save(
        times,
        labels,
        n_bins=25,
        title=None,
        legend_title=None,
        save_path=None,
        fig_format=Const.OUT_FORMAT,
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
            file_extension = os.path.splitext(save_path)[1]
            if not file_extension:
                save_path = save_path + fig_format

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

    @staticmethod
    def _runner_timing(
        env, agent, n_timings=100, verbose=False, **kwargs,
    ):
        np.random.seed(0)
        env.seed(0)

        timings = []

        done = False
        obs = env.reset()
        for t in range(n_timings):
            start_total = timer()
            action, timing = agent.act_with_timing(
                obs, done, verbose=verbose, **kwargs,  # Arguments for building a model
            )

            start_step = timer()
            obs_next, reward, done, info = env.step(action)
            timing["step"] = timer() - start_step

            pprint("Step:", t)
            if verbose:
                print_action(action)
                print_info(info, done, reward)

            obs = obs_next
            if done:
                print("DONE\n")
                obs = env.reset()

            timing["total"] = timer() - start_total
            timings.append(timing)

        return pd.DataFrame(timings)

    def compare_by_solver_and_parts(
        self, case, agent, save_dir, solver_names, n_bins=25, **kwargs,
    ):
        file_name = "solvers"
        case_name = self._get_case_name(case)

        data_dict = dict()
        for idx, solver_name in enumerate(solver_names):
            data_dict[f"{solver_name}-{idx}"] = self._runner_timing(
                env=case.env, agent=agent, solver_name=solver_name, **kwargs,
            )

        self._plot_and_save(
            times=[data_dict[param]["solve"] for param in data_dict],
            labels=[param.capitalize() for param in data_dict],
            n_bins=n_bins,
            title=f"{case.name} - MIP solver comparison",
            legend_title="Solver",
            save_path=os.path.join(save_dir, file_name),
        )

        data_parts = data_dict[f"{solver_names[0]}-0"]
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

    def compare_by_switching_limits(
        self, case, agent, save_dir, switch_limits, n_bins=25, **kwargs,
    ):
        file_name = "limits"
        case_name = self._get_case_name(case)

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
            legend_title="Line-Substation",
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

        data_dict = dict()
        for activations in constraint_activations:
            (
                allow_onesided_disconnection,
                allow_implicit_diconnection,
                symmmetry,
                gen_load_bus_balance,
                switching_limits,
                cooldown,
                unitary_action,
            ) = activations
            activations_str = "-".join(["T" if a else "F" for a in activations])

            data_dict[activations_str] = self._runner_timing(
                env=case.env,
                agent=agent,
                allow_onesided_disconnection=allow_onesided_disconnection,
                allow_implicit_diconnection=allow_implicit_diconnection,
                symmetry=symmmetry,
                gen_load_bus_balance=gen_load_bus_balance,
                switching_limits=switching_limits,
                cooldown=cooldown,
                unitary_action=unitary_action,
                **kwargs,
            )

        self._plot_and_save(
            times=[data_dict[param]["solve"] for param in data_dict],
            labels=[param for param in data_dict],
            n_bins=n_bins,
            title=f"{case_name} - Constraint activations comparison",
            legend_title="Onesided-Implicit-Symmetry-Balance-Switching-Cooldown-Unitary",
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

        data_dict = dict()
        for objective in objectives:
            (
                gen_cost,
                lin_line_margins,
                quad_line_margins,
                lin_gen_penalty,
                quad_gen_penalty,
            ) = objective
            objective_str = "-".join(["T" if obj else "F" for obj in objective])

            data_dict[objective_str] = self._runner_timing(
                env=case.env,
                agent=agent,
                gen_cost=gen_cost,
                lin_line_margins=lin_line_margins,
                quad_line_margins=quad_line_margins,
                lin_gen_penalty=lin_gen_penalty,
                quad_gen_penalty=quad_gen_penalty,
                **kwargs,
            )

        self._plot_and_save(
            times=[data_dict[param]["solve"] for param in data_dict],
            labels=[param for param in data_dict],
            n_bins=n_bins,
            title=f"{case_name} - Objective function comparison",
            legend_title="GenCost-LinMar-QuadMar-LinGen-QuadGen",
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
    def _get_case_name(case):
        env_pf = "AC"
        if case.env.parameters.ENV_DC:
            env_pf = "DC"

        return f"{case.name} ({env_pf})"
