import itertools
import os
import random
from decimal import Decimal
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from lib.action_space import ActionSpaceGenerator
from lib.dc_opf import GridDCOPF, TopologyOptimizationDCOPF
from lib.visualizer import print_info


class ExperimentDCOPFTiming:
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
        gen_cost=False,
        line_margin=False,
        min_rho=True,
        verbose=False,
    ):
        random.seed(0)

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

        obs = env.reset()
        grid = GridDCOPF(
            case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p
        )
        t = 0
        while len(measurements) < n_measurements:
            start_total = timer()

            topo_vect = obs.topo_vect
            line_status = obs.line_status

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
                gen_cost=gen_cost,
                line_margin=line_margin,
                min_rho=min_rho,
            )
            time_build = timer() - start_build

            start_solve = timer()
            model.solve(tol=tol, verbose=verbose)
            time_solve = timer() - start_solve

            start_step = timer()
            obs_next, reward, done, info = env.step(action)
            grid.update(obs_next, reset=False, verbose=verbose)
            time_step = timer() - start_step

            print(f"\n\nSTEP {t}")
            print(action)
            print(
                "{:<35}{}\t{}".format(
                    "ENV", str(topo_vect), str(line_status.astype(int))
                )
            )
            print_info(info, done, reward)

            obs = obs_next
            t = t + 1
            if done:
                print("\n\nDONE")
                obs = env.reset()
                grid.update(obs_next, reset=True, verbose=verbose)

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
        self, case, save_dir, solver_names, n_bins=25, **kwargs,
    ):
        data = dict()

        fig, ax = plt.subplots()
        ax.set_title(f"{case.name} - MIP solver comparison")
        tol_str = "{:.2e}".format(Decimal(kwargs["tol"]))
        for solver_name in solver_names:
            data[solver_name] = self._runner_timing(
                case=case, solver_name=solver_name, **kwargs,
            )

            sns.distplot(
                data[solver_name]["solve"],
                label=solver_name.capitalize(),
                ax=ax,
                hist=True,
                kde=True,
                bins=n_bins,
                norm_hist=True,
            )
            data[solver_name].to_csv(
                os.path.join(save_dir, f"{case.name}_tol-{tol_str}_{solver_name}.csv")
            )

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("PDF")
        ax.legend(title="Solver")
        ax.set_xlim(left=0)

        fig.show()
        fig.savefig(
            os.path.join(
                save_dir,
                f"{case.name}_tol-{tol_str}_" + "-".join(solver_names) + ".png",
            )
        )

    def compare_by_tolerance(self, case, save_dir, tols, n_bins=25, **kwargs):
        data = dict()

        fig, ax = plt.subplots()
        ax.set_title(f"{case.name} - Duality gap tolerance comparison")
        for tol in tols:
            data[str(tol)] = self._runner_timing(case=case, tol=tol, **kwargs,)

            sns.distplot(
                data[str(tol)]["solve"],
                label=str(tol),
                ax=ax,
                hist=True,
                kde=True,
                bins=n_bins,
                norm_hist=True,
            )

            tol_str = "{:.2e}".format(Decimal(tol))
            data[str(tol)].to_csv(
                os.path.join(save_dir, f"{case.name}_tol-{tol_str}.csv")
            )

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("PDF")
        ax.legend(title="Tolerance")
        ax.set_xlim(left=0)

        fig.show()
        fig.savefig(os.path.join(save_dir, f"{case.name}_tols.png"))

    def compare_by_switching_limits(
        self, case, save_dir, switch_limits, n_bins=25, **kwargs,
    ):
        data = dict()

        fig, ax = plt.subplots()
        ax.set_title(f"{case.name} - Maximum switching limit comparison")
        for limits in switch_limits:
            n_max_line_status_changed, n_max_sub_changed = limits
            limits_str = f"{n_max_line_status_changed}-{n_max_sub_changed}"

            data[limits_str] = self._runner_timing(
                case=case,
                n_max_line_status_changed=n_max_line_status_changed,
                n_max_sub_changed=n_max_sub_changed,
                **kwargs,
            )

            sns.distplot(
                data[limits_str]["solve"],
                label=limits_str,
                ax=ax,
                hist=True,
                kde=True,
                bins=n_bins,
                norm_hist=True,
            )
            data[limits_str].to_csv(
                os.path.join(save_dir, f"{case.name}_{limits_str}.csv")
            )

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("PDF")
        ax.legend(title="Line-Substation")
        ax.set_xlim(left=0)

        fig.show()
        fig.savefig(os.path.join(save_dir, f"{case.name}_limits.png"))

    def compare_by_constraint_activations(
        self, case, save_dir, constraint_activations, n_bins=25, **kwargs,
    ):
        data = dict()

        fig, ax = plt.subplots()
        ax.set_title(f"{case.name} - Constraint activations comparison")
        for activations in constraint_activations:
            line_disconnection, symmmetry, switching_limits, cooldown = activations
            activations_str = "-".join(["T" if a else "F" for a in activations])

            data[activations_str] = self._runner_timing(
                case=case,
                line_disconnection=line_disconnection,
                symmetry=symmmetry,
                switching_limits=switching_limits,
                cooldown=cooldown,
                **kwargs,
            )

            sns.distplot(
                data[activations_str]["solve"],
                label=activations_str,
                ax=ax,
                hist=True,
                kde=True,
                bins=n_bins,
                norm_hist=True,
            )

            data[activations_str].to_csv(
                os.path.join(save_dir, f"{case.name}_{activations_str}.csv")
            )

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("PDF")
        ax.legend(title="Line-Symmetry-Switching-Cooldown")
        ax.set_xlim(left=0)

        fig.show()
        fig.savefig(os.path.join(save_dir, f"{case.name}_activations.png"))

    def compare_by_objective(
        self, case, save_dir, objectives, n_bins=25, **kwargs,
    ):
        data = dict()

        fig, ax = plt.subplots()
        ax.set_title(f"{case.name} - Objective function comparison")
        for objective in objectives:
            gen_cost, line_margin, min_rho = objective

            objective_str = "-".join(["T" if obj else "F" for obj in objective])
            data[objective_str] = self._runner_timing(
                case=case,
                gen_cost=gen_cost,
                line_margin=line_margin,
                min_rho=min_rho,
                **kwargs,
            )

            sns.distplot(
                data[objective_str]["solve"],
                label=objective_str,
                ax=ax,
                hist=True,
                kde=True,
                bins=n_bins,
                norm_hist=True,
            )

            data[objective_str].to_csv(
                os.path.join(save_dir, f"{case.name}_{objective_str}.csv")
            )

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("PDF")
        ax.legend(title="Generator-Margins-Rho")
        ax.set_xlim(left=0)

        fig.show()
        fig.savefig(os.path.join(save_dir, f"{case.name}_objectives.png"))
