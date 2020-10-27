import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PyPDF2 import PdfFileMerger

from lib.constants import Constants as Const
from lib.visualizer import pprint
from .experiment_base import ExperimentBase


class ExperimentDCOPFTiming(ExperimentBase):
    def compare_by_solver_and_parts(
            self, case, agent, save_dir, solver_names, x_right=None, **kwargs,
    ):
        file_name = agent.name.replace(" ", "-").lower() + "-solvers"
        case_name = self._get_case_name(case)

        self.print_experiment("Timing - Solver and Parts")

        data_df = self._load_csv(save_path=os.path.join(save_dir, file_name + ".csv"))
        if not len(data_df) > 0:
            data_dict = dict()
            for solver_name in solver_names:
                data_dict[solver_name] = self._runner_timing(
                    env=case.env, agent=agent, solver_name=solver_name, **kwargs,
                )
        else:
            params = np.unique(data_df["params"].values).tolist()
            data_dict = dict()
            for param in params:
                data_dict[param] = data_df[data_df["params"] == param]

        self._plot_and_save(
            times=[data_dict[param]["solve"] for param in data_dict],
            labels=[param.capitalize() for param in data_dict],
            title=f"{case_name}, {agent.name} - MIP solver comparison",
            x_right=x_right,
            legend_title="Solver",
            save_path=os.path.join(save_dir, file_name),
        )

        data_parts = data_dict["gurobi"]
        self._plot_and_save(
            times=[data_parts[part] for part in ["build", "solve"]],
            labels=[part.capitalize() for part in ["build", "solve"]],
            title=f"{case_name}, {agent.name} - Step parts comparison",
            x_right=x_right,
            legend_title="Part",
            save_path=os.path.join(
                save_dir, agent.name.replace(" ", "-").lower() + "-parts"
            ),
        )

        self._save_csv(
            data_dict=data_dict, save_path=os.path.join(save_dir, file_name + ".csv")
        )

    def compare_by_tolerance(self, case, agent, save_dir, tols, x_right=None, **kwargs):
        file_name = agent.name.replace(" ", "-").lower() + "-tolerances"
        case_name = self._get_case_name(case)

        self.print_experiment("Timing - Tolerances")

        data_df = self._load_csv(save_path=os.path.join(save_dir, file_name + ".csv"))
        if not len(data_df) > 0:
            data_dict = dict()
            for tol in tols:
                data_dict[str(tol)] = self._runner_timing(
                    env=case.env, agent=agent, tol=tol, **kwargs,
                )
        else:
            params = np.unique(data_df["params"].values).tolist()
            data_dict = dict()
            for param in params:
                data_dict[param] = data_df[data_df["params"] == param]

        self._plot_and_save(
            times=[data_dict[param]["solve"] for param in data_dict],
            labels=[param for param in data_dict],
            title=f"{case_name}, {agent.name} - Duality gap tolerance comparison",
            x_right=x_right,
            legend_title="Tolerance",
            save_path=os.path.join(save_dir, file_name),
        )

        self._save_csv(
            data_dict=data_dict, save_path=os.path.join(save_dir, file_name + ".csv")
        )

    def compare_by_delta_max(self, case, agent, save_dir, deltas, x_right=None, **kwargs):
        file_name = agent.name.replace(" ", "-").lower() + "-deltas"
        case_name = self._get_case_name(case)

        self.print_experiment("Timing - Bounds on bus voltage angles")

        data_df = self._load_csv(save_path=os.path.join(save_dir, file_name + ".csv"))
        if not len(data_df) > 0:
            data_dict = dict()
            for delta_max in deltas:
                data_dict[str(delta_max)] = self._runner_timing(
                    env=case.env, agent=agent, delta_max=delta_max, **kwargs,
                )
        else:
            params = np.unique(data_df["params"].values).tolist()
            data_dict = dict()
            for param in params:
                data_dict[param] = data_df[data_df["params"] == param]

        self._plot_and_save(
            times=[data_dict[param]["solve"] for param in data_dict],
            labels=[param for param in data_dict],
            title=f"{case_name}, {agent.name} - Bounds on bus voltage angles",
            x_right=x_right,
            legend_title=r"$\delta^{\mathrm{max}}$",
            save_path=os.path.join(save_dir, file_name),
        )

        self._save_csv(
            data_dict=data_dict, save_path=os.path.join(save_dir, file_name + ".csv")
        )

    def compare_by_switching_limits(
            self, case, agent, save_dir, switch_limits, x_right=None, **kwargs,
    ):
        file_name = agent.name.replace(" ", "-").lower() + "-limits"
        case_name = self._get_case_name(case)

        self.print_experiment("Timing - Maximum switching limit")

        data_df = self._load_csv(save_path=os.path.join(save_dir, file_name + ".csv"))
        if not len(data_df) > 0:
            data_dict = dict()
            for limits in switch_limits:
                n_max_line_status_changed, n_max_sub_changed = limits
                limits_str = f"{n_max_line_status_changed}-{n_max_sub_changed}"

                data_dict[limits_str] = self._runner_timing(
                    env=case.env,
                    agent=agent,
                    n_max_line_status_changed=n_max_line_status_changed,
                    n_max_sub_changed=n_max_sub_changed,
                    con_unitary_action=False,  # Otherwise
                    **kwargs,
                )
        else:
            params = np.unique(data_df["params"].values).tolist()
            data_dict = dict()
            for param in params:
                data_dict[param] = data_df[data_df["params"] == param]

        self._plot_and_save(
            times=[data_dict[param]["solve"] for param in data_dict],
            labels=[param for param in data_dict],
            title=f"{case_name}, {agent.name} - Maximum switching limit comparison",
            x_right=x_right,
            legend_title=r"$n_\mathcal{P}$-$n_\mathcal{S}$",
            save_path=os.path.join(save_dir, file_name),
        )

        self._save_csv(
            data_dict=data_dict, save_path=os.path.join(save_dir, file_name + ".csv")
        )

    def compare_by_constraint_activations(
            self, case, agent, save_dir, constraint_activations, x_right=None, **kwargs,
    ):
        file_name = agent.name.replace(" ", "-").lower() + "-activations"
        case_name = self._get_case_name(case)

        self.print_experiment("Timing - Constraint Activations")

        data_df = self._load_csv(save_path=os.path.join(save_dir, file_name + ".csv"))
        if not len(data_df) > 0:
            data_dict = dict()
            for act_dict, act_str in constraint_activations:
                data_dict[act_str] = self._runner_timing(
                    env=case.env, agent=agent, **act_dict, **kwargs,
                )
        else:
            params = np.unique(data_df["params"].values).tolist()
            data_dict = dict()
            for param in params:
                data_dict[param] = data_df[data_df["params"] == param]

        self._plot_and_save(
            times=[data_dict[param]["solve"] for param in data_dict],
            labels=[param for param in data_dict],
            title=f"{case_name}, {agent.name} - Constraint activations comparison",
            x_right=x_right,
            legend_title="Constraint deactivation",
            save_path=os.path.join(save_dir, file_name),
        )

        self._save_csv(
            data_dict=data_dict, save_path=os.path.join(save_dir, file_name + ".csv")
        )

    def compare_by_objective(
            self, case, agent, save_dir, objectives, x_right=None, **kwargs,
    ):
        file_name = agent.name.replace(" ", "-").lower() + "-objectives"
        case_name = self._get_case_name(case)

        self.print_experiment("Timing - Objective")

        data_df = self._load_csv(save_path=os.path.join(save_dir, file_name + ".csv"))
        if not len(data_df) > 0:
            data_dict = dict()
            for obj_dict, obj_str in objectives:
                data_dict[obj_str] = self._runner_timing(
                    env=case.env, agent=agent, **obj_dict, **kwargs,
                )
        else:
            params = np.unique(data_df["params"].values).tolist()
            data_dict = dict()
            for param in params:
                data_dict[param] = data_df[data_df["params"] == param]

        self._plot_and_save(
            times=[data_dict[param]["solve"] for param in data_dict],
            labels=[param for param in data_dict],
            title=f"{case_name}, {agent.name} - Objective function comparison",
            x_right=x_right,
            legend_title="Objective modification",
            save_path=os.path.join(save_dir, file_name),
        )

        self._save_csv(
            data_dict=data_dict, save_path=os.path.join(save_dir, file_name + ".csv")
        )

    def compare_by_warmstart(
            self, case, agent, save_dir, x_right=None, **kwargs,
    ):
        file_name = agent.name.replace(" ", "-").lower() + "-warmstart"
        case_name = self._get_case_name(case)

        self.print_experiment("Timing - Warm Start")

        data_df = self._load_csv(save_path=os.path.join(save_dir, file_name + ".csv"))
        if not len(data_df) > 0:
            data_dict = dict()
            for warm_start in [True, False]:
                data_dict[str(warm_start)] = self._runner_timing(
                    env=case.env, agent=agent, warm_start=warm_start, **kwargs,
                )
        else:
            params = np.unique(data_df["params"].values).tolist()
            data_dict = dict()
            for param in params:
                data_dict[param] = data_df[data_df["params"] == param]

        self._plot_and_save(
            times=[data_dict[param]["solve"] for param in data_dict],
            labels=[param for param in data_dict],
            title=f"{case_name}, {agent.name} - Solver warm start comparison",
            x_right=x_right,
            legend_title="Warm start",
            save_path=os.path.join(save_dir, file_name),
        )

        self._save_csv(
            data_dict=data_dict, save_path=os.path.join(save_dir, file_name + ".csv")
        )

    def compare_by_lambda_gen(self, case, agent, save_dir, lambdas, x_right=None, **kwargs):
        file_name = agent.name.replace(" ", "-").lower() + "-lambdas"
        case_name = self._get_case_name(case)

        self.print_experiment("Timing - Generator penalty scaling")

        data_df = self._load_csv(save_path=os.path.join(save_dir, file_name + ".csv"))
        if not len(data_df) > 0:
            data_dict = dict()
            for lambd in lambdas:
                data_dict[str(lambd)] = self._runner_timing(
                    env=case.env, agent=agent, obj_lambda_gen=lambd, **kwargs,
                )
        else:
            params = np.unique(data_df["params"].values).tolist()
            data_dict = dict()
            for param in params:
                data_dict[param] = data_df[data_df["params"] == param]

        self._plot_and_save(
            times=[data_dict[param]["solve"] for param in data_dict],
            labels=[param for param in data_dict],
            title=f"{case_name}, {agent.name} - Generator penalty scaling comparison",
            x_right=x_right,
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
        env.chronics_handler.tell_id(-1)

        agent.set_kwargs(**kwargs)
        agent.print_agent(default=verbose)

        timings = []

        done = False
        obs = env.reset()
        pprint("    - Chronic:", env.chronics_handler.get_id())
        for _ in range(n_timings):
            action, timing = agent.act_with_timing(obs, done)

            obs_next, reward, done, info = env.step(action)
            obs = obs_next
            if done:
                obs = env.reset()
                pprint("    - Done! Next chronic:", env.chronics_handler.get_id())

            timings.append(timing)

        return pd.DataFrame(timings)

    @staticmethod
    def _plot_and_save(
            times, labels, title=None, x_right=None, legend_title=None, save_path=None,
    ):
        colors = Const.COLORS

        fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
        # ax.set_title(title)
        for i, (time, label) in enumerate(zip(times, labels)):
            color = colors[i % len(colors)]

            sns.kdeplot(
                time, label=label, ax=ax, color=color
            )

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("PDF")

        if x_right and ax.get_xlim()[-1] > x_right:
            ax.set_xlim(left=0, right=x_right)
        else:
            ax.set_xlim(left=0)

        ax.legend(title=legend_title)
        fig.tight_layout()
        fig.show()
        if save_path:
            fig.savefig(save_path)
        plt.close(fig)

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
    def _load_csv(save_path):
        if save_path:
            file_extension = os.path.splitext(save_path)[1]
            if not file_extension:
                save_path = save_path + ".csv"

            if os.path.isfile(save_path):
                data = pd.read_csv(save_path)
                return data

        return pd.DataFrame([])

    @staticmethod
    def aggregate_by_agent(agent, save_dir, delete_file=False):
        merger = PdfFileMerger()

        agent_name = agent.name.replace(" ", "-").lower()
        agent_files = []

        for timing in [
            "solvers",
            "parts",
            "tolerances",
            "deltas",
            "limits",
            "activations",
            "objectives",
            "warmstart",
            "lambdas",
        ]:
            file = agent_name + "-" + timing + ".pdf"
            if file in os.listdir(save_dir):
                f = open(os.path.join(save_dir, file), "rb")
                agent_files.append((file, f))
                merger.append(f)

        with open(os.path.join(save_dir, "_" + agent_name + "-timing.pdf"), "wb") as f:
            merger.write(f)

        for file, f in agent_files:
            f.close()
            try:
                if delete_file:
                    os.remove(os.path.join(save_dir, file))
            except PermissionError as e:
                print(e)
