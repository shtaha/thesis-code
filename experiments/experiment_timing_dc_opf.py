import itertools
import os
import random
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from lib.action_space import ActionSpaceGenerator
from lib.dc_opf import GridDCOPF, TopologyOptimizationDCOPF
from lib.visualizer import print_info


class ExperimentDCOPFTiming:
    @staticmethod
    def _plot_and_save(
        times, labels, n_bins=25, title=None, legend_title=None, save_path=None
    ):
        fig, ax = plt.subplots()
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
                save_path = save_path + ".png"

            fig.savefig(save_path)

    @staticmethod
    def _save_csv(data_dict, save_path):
        data = pd.DataFrame(columns=["params", "total", "build", "solve", "step"])
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
        case,
        n_measurements=100,
        tol=0.01,
        solver_name="gurobi",
        n_max_line_status_changed=1,
        n_max_sub_changed=1,
        line_disconnection=True,
        symmetry=True,
        switching_limits=True,
        cooldown=True,
        unitary_action=True,
        gen_cost=False,
        lin_line_margins=True,
        quad_line_margins=False,
        lambd=1.0,
        lin_gen_penalty=True,
        quad_gen_penalty=False,
        warm_start=True,
        verbose=False,
    ):
        random.seed(0)
        np.random.seed(0)

        measurements = list()
        env = case.env

        action_generator = ActionSpaceGenerator(env)
        (
            actions_topology_set,
            actions_topology_set_info,
        ) = action_generator.get_all_unitary_topologies_set(
            filter_one_line_disconnections=True
        )
        actions_do_nothing = env.action_space({})

        actions = list(itertools.chain([actions_do_nothing], actions_topology_set))

        _ = env.reset()
        grid = GridDCOPF(
            case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p
        )
        t = 0
        while len(measurements) < n_measurements:
            start_total = timer()

            action = random.choice(actions)

            start_build = timer()
            model = TopologyOptimizationDCOPF(
                f"{case.name} DC OPF Topology Optimization",
                grid=grid,
                grid_backend=case.grid_backend,
                base_unit_p=case.base_unit_p,
                base_unit_v=case.base_unit_v,
                solver_name=solver_name,
                n_max_line_status_changed=n_max_line_status_changed,
                n_max_sub_changed=n_max_sub_changed,
            )
            model.build_model(
                line_disconnection=line_disconnection,
                symmetry=symmetry,
                switching_limits=switching_limits,
                cooldown=cooldown,
                unitary_action=unitary_action,
                gen_cost=gen_cost,
                lin_line_margins=lin_line_margins,
                quad_line_margins=quad_line_margins,
                lambd=lambd,
                lin_gen_penalty=lin_gen_penalty,
                quad_gen_penalty=quad_gen_penalty,
            )
            time_build = timer() - start_build

            start_solve = timer()
            model.solve(tol=tol, verbose=verbose, warm_start=warm_start)
            time_solve = timer() - start_solve

            start_step = timer()
            obs_next, reward, done, info = env.step(action)
            grid.update(obs_next, reset=False)
            time_step = timer() - start_step

            print(f"\n\nSTEP {t}")
            print(action)
            print_info(info, done, reward)

            t = t + 1
            if done:
                print("\n\nDONE")
                _ = env.reset()
                grid.update(obs_next, reset=True)

            time_total = timer() - start_total

            measurements.append(
                {
                    "total": time_total,
                    "build": time_build,
                    "solve": time_solve,
                    "step": time_step,
                }
            )

        return pd.DataFrame(measurements)

    def compare_by_solver(
        self, case, save_dir, solver_names, n_bins, **kwargs,
    ):
        file_name = f"{case.name}_solvers"

        data_dict = dict()
        for idx, solver_name in enumerate(solver_names):
            data_dict[f"{solver_name}-{idx}"] = self._runner_timing(
                case=case, solver_name=solver_name, **kwargs,
            )

        self._plot_and_save(
            times=[data_dict[param]["solve"] for param in data_dict],
            labels=[param.capitalize() for param in data_dict],
            n_bins=n_bins,
            title=f"{case.name} - MIP solver comparison",
            legend_title="Solver",
            save_path=os.path.join(save_dir, file_name + ".png"),
        )

        self._save_csv(
            data_dict=data_dict, save_path=os.path.join(save_dir, file_name + ".csv")
        )

    def compare_by_tolerance(self, case, save_dir, tols, n_bins, **kwargs):
        file_name = f"{case.name}_tols"

        data_dict = dict()
        for tol in tols:
            data_dict[str(tol)] = self._runner_timing(case=case, tol=tol, **kwargs,)

        self._plot_and_save(
            times=[data_dict[param]["solve"] for param in data_dict],
            labels=[param for param in data_dict],
            n_bins=n_bins,
            title=f"{case.name} - Duality gap tolerance comparison",
            legend_title="Tolerance",
            save_path=os.path.join(save_dir, file_name + ".png"),
        )

        self._save_csv(
            data_dict=data_dict, save_path=os.path.join(save_dir, file_name + ".csv")
        )

    def compare_by_switching_limits(
        self, case, save_dir, switch_limits, n_bins, **kwargs,
    ):
        file_name = f"{case.name}_limits"

        data_dict = dict()
        for limits in switch_limits:
            n_max_line_status_changed, n_max_sub_changed = limits
            limits_str = f"{n_max_line_status_changed}-{n_max_sub_changed}"

            data_dict[limits_str] = self._runner_timing(
                case=case,
                n_max_line_status_changed=n_max_line_status_changed,
                n_max_sub_changed=n_max_sub_changed,
                **kwargs,
            )

        self._plot_and_save(
            times=[data_dict[param]["solve"] for param in data_dict],
            labels=[param for param in data_dict],
            n_bins=n_bins,
            title=f"{case.name} - Maximum switching limit comparison",
            legend_title="Line-Substation",
            save_path=os.path.join(save_dir, file_name + ".png"),
        )

        self._save_csv(
            data_dict=data_dict, save_path=os.path.join(save_dir, file_name + ".csv")
        )

    def compare_by_constraint_activations(
        self, case, save_dir, constraint_activations, n_bins=25, **kwargs,
    ):
        file_name = f"{case.name}_activations"

        data_dict = dict()
        for activations in constraint_activations:
            (
                line_disconnection,
                symmmetry,
                switching_limits,
                cooldown,
                unitary_action,
            ) = activations
            activations_str = "-".join(["T" if a else "F" for a in activations])

            data_dict[activations_str] = self._runner_timing(
                case=case,
                line_disconnection=line_disconnection,
                symmetry=symmmetry,
                switching_limits=switching_limits,
                cooldown=cooldown,
                unitary_action=unitary_action,
                **kwargs,
            )

        self._plot_and_save(
            times=[data_dict[param]["solve"] for param in data_dict],
            labels=[param for param in data_dict],
            n_bins=n_bins,
            title=f"{case.name} - Constraint activations comparison",
            legend_title="Line-Symmetry-Switching-Cooldown-Unitary",
            save_path=os.path.join(save_dir, file_name + ".png"),
        )

        self._save_csv(
            data_dict=data_dict, save_path=os.path.join(save_dir, file_name + ".csv")
        )

    def compare_by_objective(
        self, case, save_dir, objectives, n_bins=25, **kwargs,
    ):
        file_name = f"{case.name}_objectives"

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
                case=case,
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
            title=f"{case.name} - Objective function comparison",
            legend_title="GenCost-LinMar-QuadMar-LinGen-QuadGen",
            save_path=os.path.join(save_dir, file_name + ".png"),
        )

        self._save_csv(
            data_dict=data_dict, save_path=os.path.join(save_dir, file_name + ".csv")
        )

    def compare_by_warmstart(
        self, case, save_dir, n_bins=25, **kwargs,
    ):
        file_name = f"{case.name}_warmstart"

        data_dict = dict()
        for warm_start in [True, False]:
            data_dict[str(warm_start)] = self._runner_timing(
                case=case, warm_start=warm_start, **kwargs,
            )

        self._plot_and_save(
            times=[data_dict[param]["solve"] for param in data_dict],
            labels=[param for param in data_dict],
            n_bins=n_bins,
            title=f"{case.name} - Solver warm start comparison",
            legend_title="Warm start",
            save_path=os.path.join(save_dir, file_name + ".png"),
        )

        self._save_csv(
            data_dict=data_dict, save_path=os.path.join(save_dir, file_name + ".csv")
        )

    def compare_by_lambda(self, case, save_dir, lambdas, n_bins, **kwargs):
        file_name = f"{case.name}_lambdas"

        data_dict = dict()
        for lambd in lambdas:
            data_dict[str(lambd)] = self._runner_timing(
                case=case, lambd=lambd, **kwargs,
            )

        self._plot_and_save(
            times=[data_dict[param]["solve"] for param in data_dict],
            labels=[param for param in data_dict],
            n_bins=n_bins,
            title=f"{case.name} - Penalty scaling comparison",
            legend_title="Regularization parameter",
            save_path=os.path.join(save_dir, file_name + ".png"),
        )

        self._save_csv(
            data_dict=data_dict, save_path=os.path.join(save_dir, file_name + ".csv")
        )
