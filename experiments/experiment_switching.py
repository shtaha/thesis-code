import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyPDF2 import PdfFileMerger

from lib.chronics import get_sorted_chronics
from lib.constants import Constants as Const
from lib.rl_utils import compute_returns
from lib.visualizer import pprint
from .experiment_base import ExperimentBase, ExperimentMixin


class ExperimentSwitching(ExperimentBase, ExperimentMixin):
    def analyse(
        self, case, agent, n_chronics=1, n_steps=500, save_dir=None, verbose=False
    ):
        env = case.env

        self.print_experiment("Switching")
        agent.print_agent(default=verbose)

        file_name = agent.name.replace(" ", "-").lower() + "-chronics"
        chronic_data, done_chronic_indices = self._load_done_chronics_from_pickle(
            file_name=file_name, save_dir=save_dir
        )

        new_chronic_data = self._runner(
            case=case,
            env=env,
            agent=agent,
            n_chronics=n_chronics,
            n_steps=n_steps,
            done_chronic_indices=done_chronic_indices,
        )
        chronic_data = chronic_data.append(new_chronic_data)

        self._save_chronics(
            chronic_data=chronic_data, file_name=file_name, save_dir=save_dir
        )

    def compare_agents(self, case, save_dir=None, delete_file=True):
        case_name = self._get_case_name(case)
        chronic_data = dict()

        agent_names = []
        for file in os.listdir(save_dir):
            if "-chronics.pkl" in file:
                agent_name = file[: -len("-chronics.pkl")]
                agent_name = agent_name.replace("-", " ").capitalize()
                agent_names.append(agent_name)
                chronic_data[agent_name] = pd.read_pickle(os.path.join(save_dir, file))

        chronic_indices_all = pd.Index([], name="chronic_idx")
        for agent_name in agent_names:
            chronic_indices_all = chronic_indices_all.union(
                chronic_data[agent_name].index
            )

        chronic_names_all = pd.DataFrame(
            columns=["chronic_name"], index=chronic_indices_all
        )
        for agent_name in agent_names:
            chronic_names_all["chronic_name"].loc[
                chronic_data[agent_name].index
            ] = chronic_data[agent_name]["chronic_name"]

        for chronic_idx in chronic_indices_all:
            chronic_name = chronic_names_all["chronic_name"].loc[chronic_idx]

            self._plot_rewards(
                chronic_data, case_name, chronic_idx, chronic_name, save_dir
            )

            for dist, ylabel in [
                ("distances", r"$d(\tau, \tau^\mathrm{ref})$"),
                ("distances_status", r"$d_\mathcal{P}(\tau, \tau^\mathrm{ref})$"),
                ("distances_sub", r"$d_\mathcal{S}(\tau, \tau^\mathrm{ref})$"),
            ]:
                self._plot_distances(
                    chronic_data,
                    dist,
                    ylabel,
                    case_name,
                    chronic_idx,
                    chronic_name,
                    save_dir,
                )

            self._plot_gains(
                chronic_data, case_name, chronic_idx, chronic_name, save_dir
            )

            self._plot_objective_parts(
                chronic_data, case_name, chronic_idx, chronic_name, save_dir
            )

        self.aggregate_by_chronics(save_dir, delete_file=delete_file)

    def aggregate_by_chronics(self, save_dir, delete_file=True):
        for plot_name in [
            "rewards",
            "distances",
            "distances_status",
            "distances_sub",
            "gains",
            "objectives",
            "relative-obj",
            "rhos",
        ]:
            merger = PdfFileMerger()
            chronic_files = []
            for file in os.listdir(save_dir):
                if "agents-chronic-" in file and plot_name + ".pdf" in file:
                    f = open(os.path.join(save_dir, file), "rb")
                    chronic_files.append((file, f))
                    merger.append(f)

            if merger.inputs:
                with open(
                    os.path.join(save_dir, "_" + f"agents-chronics-{plot_name}.pdf"),
                    "wb",
                ) as f:
                    merger.write(f)

                # Reset merger
                merger.pages = []
                merger.inputs = []
                merger.output = None

            self.close_files(chronic_files, save_dir, delete_file=delete_file)

        agent_names = np.unique(
            [
                file.split("-chronic-")[0]
                for file in os.listdir(save_dir)
                if "-chronic-" in file
            ]
        )
        for agent_name in agent_names:
            for plot_name in ["mus", "fraction-obj"]:
                merger = PdfFileMerger()
                chronic_files = []
                for file in os.listdir(save_dir):
                    if (
                        "-chronic-" in file
                        and agent_name in file
                        and plot_name + ".pdf" in file
                    ):
                        f = open(os.path.join(save_dir, file), "rb")
                        chronic_files.append((file, f))
                        merger.append(f)

                if merger.inputs:
                    with open(
                        os.path.join(
                            save_dir, "_" + f"{agent_name}-chronics-{plot_name}.pdf"
                        ),
                        "wb",
                    ) as f:
                        merger.write(f)

                    # Reset merger
                    merger.pages = []
                    merger.inputs = []
                    merger.output = None

                self.close_files(chronic_files, save_dir, delete_file=delete_file)

    @staticmethod
    def _plot_gains(chronic_data, case_name, chronic_idx, chronic_name, save_dir):
        colors = Const.COLORS

        fig_obj, ax_obj = plt.subplots(figsize=Const.FIG_SIZE)
        fig_gain, ax_gain = plt.subplots(figsize=Const.FIG_SIZE)
        fig_rel, ax_rel = plt.subplots(figsize=Const.FIG_SIZE)
        for agent_id, (agent_name, agent_data) in enumerate(chronic_data.items()):
            if chronic_idx in agent_data.index and agent_name != "Do nothing agent":
                color_id = agent_id % len(colors)
                color = colors[color_id]

                if "solution_status" in agent_data.columns:
                    agent_data = agent_data[
                        agent_data["solution_status"] != "infeasible"
                    ]

                t = agent_data.loc[chronic_idx]["time_steps"]
                actions = agent_data.loc[chronic_idx]["actions"]

                obj = np.array(agent_data.loc[chronic_idx]["objectives"])
                obj_dn = np.array(agent_data.loc[chronic_idx]["objectives_dn"])
                mu_max = np.array(agent_data.loc[chronic_idx]["mu_max"])

                ax_obj.plot(
                    t, obj, linewidth=0.5, c=color, linestyle="-", label=agent_name
                )
                ax_obj.plot(t, obj_dn, linewidth=0.5, c=color, linestyle="--")
                ax_obj.plot(t, mu_max, linewidth=0.5, c=color, linestyle="-.")

                gain = obj_dn - obj
                markerline, stemlines, _ = ax_gain.stem(
                    t,
                    gain,
                    use_line_collection=True,
                    markerfmt=f"C{color_id}o",
                    basefmt=" ",
                    linefmt=f"C{color_id}",
                    label=agent_name,
                )
                plt.setp(markerline, markersize=1)
                plt.setp(stemlines, linewidth=0.5)

                rel_gain = np.divide(obj, obj_dn + 1e-9)

                # action_mask = [True if action else False for action in actions]
                # rel_gain = rel_gain[action_mask]
                ax_rel.hist(
                    rel_gain, lw=1.0, bins=25, histtype="step", label=agent_name, density=True,
                )

                # for i in range(len(t)):
                #     if isinstance(actions[i], int):
                #         action_id = actions[i]
                #         if action_id != 0:
                #             ax_gain.text(t[i], gain[i], str(action_id), fontsize=2)

        ax_obj.set_xlabel("Time step t")
        ax_obj.set_ylabel("$\mathrm{obj}$")
        ax_obj.legend()
        ax_obj.set_ylim(bottom=0.0)

        ax_gain.set_xlabel("Time step t")
        ax_gain.set_ylabel("Gain of selected action vs. do-nothing action")
        ax_gain.legend()

        ax_rel.set_xlabel(r"$\mathrm{obj}/\mathrm{obj}_{DN}$")
        ax_rel.set_ylabel("Count")
        ax_rel.set_ylabel("PDF")
        # ax_rel.set_yscale('log')
        ax_rel.legend()

        if save_dir:
            file_name = f"agents-chronic-" + "{:05}".format(chronic_idx) + "-"
            fig_obj.savefig(os.path.join(save_dir, file_name + "objectives"))
            fig_gain.savefig(os.path.join(save_dir, file_name + "gains"))
            fig_rel.savefig(os.path.join(save_dir, file_name + "relative-obj"))

        plt.close(fig_obj)
        plt.close(fig_gain)
        plt.close(fig_rel)

    @staticmethod
    def _plot_objective_parts(
        chronic_data, case_name, chronic_idx, chronic_name, save_dir
    ):
        colors = Const.COLORS

        for agent_id, agent_name in enumerate(chronic_data):
            chronic = chronic_data[agent_name].loc[chronic_idx]

            t = np.array(chronic["time_steps"])

            obj = np.array(chronic["objectives"])
            obj_dn = np.array(chronic["objectives_dn"])
            rel_gain = np.divide(obj, obj_dn + 1e-9)

            mu_max = np.array(chronic["mu_max"])
            mu_gen = np.array(chronic["mu_gen"])
            mu_max_dn = np.array(chronic["mu_max_dn"])
            mu_gen_dn = np.array(chronic["mu_gen_dn"])

            fig_mu, ax_mu = plt.subplots(figsize=Const.FIG_SIZE)
            fig_frac, ax_frac = plt.subplots(figsize=Const.FIG_SIZE)

            width = 0.25
            indices_gain = np.less_equal(rel_gain, 0.98)
            x = np.arange(int(indices_gain.sum()))

            t = t[indices_gain]
            mu_max = mu_max[indices_gain]
            mu_gen = mu_gen[indices_gain]
            mu_max_dn = mu_max_dn[indices_gain]
            mu_gen_dn = mu_gen_dn[indices_gain]
            obj_dn = obj_dn[indices_gain]

            horizon = 1
            if len(mu_max.shape) > 1:
                horizon = mu_max.shape[1]
                mu_max = mu_max.sum(axis=1)
                mu_gen = mu_gen.sum(axis=1)
                mu_max_dn = mu_max_dn.sum(axis=1)
                mu_gen_dn = mu_gen_dn.sum(axis=1)
                obj_dn = obj_dn

            ax_mu.bar(
                x, mu_max / horizon, width=width, label=r"$\mu^{max}$", color=colors[0],
            )
            ax_mu.bar(
                x,
                mu_gen / horizon,
                bottom=mu_max / horizon,
                width=width,
                label=r"$\mu^{gen}$",
                color=colors[1],
            )
            ax_frac.bar(
                x,
                100.0 * mu_max / obj_dn / horizon,
                width=width,
                label=r"$\mu^{max}$",
                color=colors[0],
            )
            ax_frac.bar(
                x,
                100.0 * 100.0 * mu_gen / obj_dn / horizon,
                bottom=100.0 * mu_max / obj_dn / horizon,
                width=width,
                label=r"$\lambda^{gen} \cdot \mu^{gen}$",
                color=colors[1],
            )

            # Do-nothing objective
            ax_mu.bar(
                x + width,
                mu_max_dn / horizon,
                width=width,
                label=r"$\mu^{max}_{DN}$",
                color=colors[2],
            )
            ax_mu.bar(
                x + width,
                mu_gen_dn / horizon,
                bottom=mu_max_dn / horizon,
                width=width,
                label=r"$\mu^{gen}_{DN}$",
                color=colors[3],
            )

            ax_frac.bar(
                x + width,
                100.0 * mu_max_dn / obj_dn / horizon,
                width=width,
                label=r"$\mu^{max}_{DN}$",
                color=colors[2],
            )
            ax_frac.bar(
                x + width,
                100.0 * 100.0 * mu_gen_dn / obj_dn / horizon,
                bottom=100.0 * mu_max_dn / obj_dn / horizon,
                width=width,
                label=r"$\lambda^{gen} \cdot \mu^{gen}_{DN}$",
                color=colors[3],
            )

            ax_mu.set_xlabel("Time step t")
            ax_mu.set_ylabel(r"$\mu$")
            ax_mu.set_xticks(x)
            ax_mu.set_xticklabels(t)
            ax_mu.legend()
            # fig_mu.suptitle(f"{case_name} - Chronic {chronic_name}")

            ax_frac.set_xlabel("Time step t")
            ax_frac.set_ylabel(r"Objective value fraction by terms [\\%]")
            ax_frac.set_xticks(x)
            ax_frac.set_xticklabels(t)
            ax_frac.set_yticks(np.arange(0, 111, 10))
            ax_frac.set_yticklabels([f"{tick} \\%" for tick in np.arange(0, 111, 10)])
            ax_frac.legend()

            if save_dir:
                agent_name_ = agent_name.replace(" ", "-").lower()
                file_name = (
                    f"{agent_name_}-chronic-" + "{:05}".format(chronic_idx) + "-"
                )
                fig_mu.savefig(os.path.join(save_dir, file_name + "mus"))
                fig_frac.savefig(os.path.join(save_dir, file_name + "fraction-obj"))

            plt.close(fig_mu)
            plt.close(fig_frac)

    @staticmethod
    def _runner(case, env, agent, n_chronics, n_steps, done_chronic_indices=()):
        chronics_dir, chronics, chronics_sorted = get_sorted_chronics(env=env)
        pprint("Chronics:", chronics_dir)

        np.random.seed(0)
        env.seed(0)

        chronic_data = []
        for chronic_idx, chronic in enumerate(chronics_sorted):
            if len(chronic_data) >= n_chronics > 0:
                break

            if chronic_idx in done_chronic_indices:
                continue

            chronic_org_idx = chronics.index(chronic)
            env.chronics_handler.tell_id(chronic_org_idx - 1)  # Set chronic id

            obs = env.reset()
            agent.reset(obs=obs)
            chronic_len = env.chronics_handler.real_data.data.max_iter

            chronic_name = "/".join(
                os.path.normpath(env.chronics_handler.get_id()).split(os.sep)[-3:]
            )

            pprint("    - Chronic:", chronic_name)

            if case.name == "Case L2RPN 2020 WCCI":
                chronic_name = env.chronics_handler.get_name().split("_")
                chronic_name = "-".join([chronic_name[1][:3], chronic_name[2]])
            else:
                chronic_name = env.chronics_handler.get_name()

            done = False
            reward = 0.0
            t = 0
            actions = []
            rewards = []
            distances = []
            distances_status = []
            distances_sub = []
            time_steps = []
            objectives = []
            objectives_dn = []
            mu_max = []
            mu_max_dn = []
            mu_gen = []
            mu_gen_dn = []
            while not done:
                action, info = agent.act_with_objectives(obs, reward, done)
                obs_next, reward, done, _ = env.step(action)
                t = env.chronics_handler.real_data.data.current_index

                if t % 50 == 0:
                    pprint("Step:", t)

                if t >= n_steps > 0:
                    done = True

                if done:
                    pprint("        - Length:", f"{t}/{chronic_len}")

                action_id = [
                    idx
                    for idx, agent_action in enumerate(agent.actions)
                    if action == agent_action
                ]

                if len(action_id) != 1:
                    action_id = np.nan
                else:
                    assert (
                        len(action_id) == 1
                    )  # Exactly one action should be equivalent
                    action_id = int(action_id[0])

                dist, dist_status, dist_sub = agent.distance_to_ref_topology(
                    obs_next.topo_vect, obs_next.line_status
                )

                obs = obs_next
                actions.append(action_id)
                time_steps.append(t)

                rewards.append(float(reward))
                distances.append(dist)
                distances_status.append(dist_status)
                distances_sub.append(dist_sub)

                if "obj" in info:
                    objectives.append(info["obj"])
                if "obj_dn" in info:
                    objectives_dn.append(info["obj_dn"])

                if "mu_max" in info:
                    mu_max.append(info["mu_max"])
                if "mu_max_dn" in info:
                    mu_max_dn.append(info["mu_max_dn"])

                if "mu_gen" in info:
                    mu_gen.append(info["mu_gen"])
                if "mu_gen_dn" in info:
                    mu_gen_dn.append(info["mu_gen_dn"])

            total_return = compute_returns(rewards)[0]
            chronic_data.append(
                {
                    "chronic_idx": chronic_idx,
                    "chronic_org_idx": chronic_org_idx,
                    "chronic_name": chronic_name,
                    "actions": actions,
                    "time_steps": time_steps,
                    "rewards": rewards,
                    "return": total_return,
                    "chronic_length": chronic_len,
                    "duration": t,
                    "distances": distances,
                    "distances_status": distances_status,
                    "distances_sub": distances_sub,
                    "objectives": objectives,
                    "objectives_dn": objectives_dn,
                    "mu_max": mu_max,
                    "mu_max_dn": mu_max_dn,
                    "mu_gen": mu_gen,
                    "mu_gen_dn": mu_gen_dn,
                }
            )

        if chronic_data:
            chronic_data = pd.DataFrame(chronic_data)
            chronic_data = chronic_data.set_index("chronic_idx")
        else:
            chronic_data = pd.DataFrame()

        return chronic_data
