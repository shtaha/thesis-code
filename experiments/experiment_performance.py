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


class ExperimentPerformance(ExperimentBase, ExperimentMixin):
    def analyse(self, case, agent, save_dir=None, verbose=False):
        env = case.env

        self.print_experiment("Failures and Switching")
        agent.print_agent(default=verbose)

        file_name = agent.name.replace(" ", "-").lower() + "-chronics"
        chronic_data, done_chronic_indices = self._load_done_chronics(
            file_name=file_name, save_dir=save_dir
        )

        new_chronic_data = self._runner(
            case=case, env=env, agent=agent, done_chronic_indices=done_chronic_indices
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

            self._plot_relative_flow(
                chronic_data, case_name, chronic_idx, chronic_name, save_dir
            )

            for dist, ylabel in [
                ("distances", "Unitary action distance to reference topology"),
                ("distances_status", "Line status distance to reference topology"),
                ("distances_sub", "Substation distance distance to reference topology"),
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

            self._plot_generators(
                chronic_data, case_name, chronic_idx, chronic_name, save_dir
            )

        self._plot_durations(
            chronic_data, chronic_names_all, agent_names, case_name, save_dir
        )

        self._plot_returns(
            chronic_data, chronic_names_all, agent_names, case_name, save_dir
        )

        self.aggregate_by_chronics(save_dir, delete_file=delete_file)

    def aggregate_by_chronics(self, save_dir, delete_file=True):
        for plot_name in [
            "rewards",
            "distances",
            "distances_status",
            "distances_sub",
            "advantages",
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
            for plot_name in ["generators"]:
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
    def _plot_durations(
        chronic_data, chronic_names_all, agent_names, case_name, save_dir=None
    ):
        fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
        width = 0.3 / len(agent_names)

        x_all = np.arange(len(chronic_names_all.index))
        for agent_id, agent_name in enumerate(agent_names):
            chronic_names = chronic_data[agent_name]["chronic_name"].values

            x = []
            for x_, name in zip(x_all, chronic_names_all["chronic_name"]):
                if name in chronic_names:
                    x.append(x_)
            x = np.array(x)

            y = chronic_data[agent_name]["duration"]
            ax.barh(x + agent_id * width, y, width, left=0.001, label=agent_name)

            ax.scatter(
                chronic_data[agent_name]["chronic_length"], x, marker="|", c="black"
            )

        ax.set_yticks(x_all)
        ax.set_yticklabels(chronic_names_all["chronic_name"])
        ax.invert_yaxis()
        ax.legend()

        ax.set_ylabel("Chronic")
        ax.set_xlabel("Chronic duration")
        fig.suptitle(f"{case_name} - Chronic durations")

        if save_dir:
            file_name = f"_agents-chronics-"
            fig.savefig(os.path.join(save_dir, file_name + "durations"))
        plt.close(fig)

    @staticmethod
    def _plot_returns(
        chronic_data, chronic_names_all, agent_names, case_name, save_dir=None
    ):
        fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
        width = 0.3 / len(agent_names)

        x_all = np.arange(len(chronic_names_all.index))
        for agent_id, agent_name in enumerate(agent_names):
            chronic_names = chronic_data[agent_name]["chronic_name"].values

            x = []
            for x_, name in zip(x_all, chronic_names_all["chronic_name"]):
                if name in chronic_names:
                    x.append(x_)
            x = np.array(x)

            y = chronic_data[agent_name]["return"]
            ax.barh(x + agent_id * width, y, width, left=0.001)

        ax.set_yticks(x_all)
        ax.set_yticklabels(chronic_names_all["chronic_name"])
        ax.invert_yaxis()
        ax.legend(tuple(agent_names))

        ax.set_ylabel("Chronic")
        ax.set_xlabel("Chronic return")
        fig.suptitle(f"{case_name} - Chronic returns")

        if save_dir:
            file_name = f"_agents-chronics-"
            fig.savefig(os.path.join(save_dir, file_name + "returns"))
        plt.close(fig)

    @staticmethod
    def _plot_generators(
        chronic_data, case_name, chronic_idx, chronic_name, save_dir=None
    ):
        colors = Const.COLORS
        for agent_name in chronic_data:
            if (
                agent_name != "Do nothing agent"
                and chronic_idx in chronic_data[agent_name].index
            ):
                fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
                t = chronic_data[agent_name].loc[chronic_idx]["time_steps"]

                results = chronic_data[agent_name].loc[chronic_idx]["results"]
                observations = chronic_data[agent_name].loc[chronic_idx]["observations"]

                res_gen = np.vstack(
                    [result["res_gen"]["p_pu"].values for result in results if result]
                )
                res_gen_env = np.vstack([obs.prod_p for obs in observations if obs])

                for gen_id in range(res_gen.shape[1]):
                    color = colors[gen_id % len(colors)]
                    ax.plot(
                        t, res_gen[:, gen_id], c=color, linestyle="-", linewidth=0.5,
                    )
                    ax.plot(
                        t,
                        res_gen_env[:, gen_id],
                        c=color,
                        linestyle="--",
                        linewidth=0.5,
                    )

                ax.set_xlabel("Time step t")
                ax.set_ylabel(r"$P_g$ [p.u.]")
                fig.suptitle(f"{case_name} - Chronic {chronic_name}")

                if save_dir:
                    agent_name_ = agent_name.replace(" ", "-").lower()
                    file_name = (
                        f"{agent_name_}-chronic-" + "{:05}".format(chronic_idx) + "-"
                    )
                    fig.savefig(os.path.join(save_dir, file_name + "generators"))
                plt.close(fig)

    @staticmethod
    def _runner(case, env, agent, done_chronic_indices=()):
        chronics_dir, chronics, chronics_sorted = get_sorted_chronics(
            case=case, env=env
        )
        pprint("Chronics:", chronics_dir)

        np.random.seed(0)
        env.seed(0)

        chronic_data = []
        for chronic_idx, chronic in enumerate(chronics_sorted):
            if chronic_idx in done_chronic_indices:
                continue

            if case.name == "Case RTE 5":
                if chronic_idx > 1:
                    continue
            elif case.name == "Case L2RPN 2019":
                if chronic_idx != 10:
                    continue
            elif case.name == "Case L2RPN 2020 WCCI":
                if chronic_idx % 480 != 0:
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
            actions_info = []
            rewards = []
            rewards_sim = []
            rewards_dn = []
            observations = []
            results = []
            distances = []
            distances_status = []
            distances_sub = []
            time_steps = []
            while not done:
                action = agent.act(obs, reward, done)
                obs_next, reward, done, info = env.step(action)

                t = env.chronics_handler.real_data.data.current_index

                if t % 50 == 0:
                    pprint("Step:", t)

                if done:
                    pprint("        - Length:", f"{t}/{chronic_len}")

                action_id = [
                    idx
                    for idx, agent_action in enumerate(agent.actions)
                    if action == agent_action
                ]

                if "unitary_action" in agent.model_kwargs:
                    if not agent.model_kwargs["unitary_action"] and len(action_id) != 1:
                        action_id = np.nan
                    else:
                        assert (
                            len(action_id) == 1
                        )  # Exactly one action should be equivalent
                        action_id = int(action_id[0])
                else:
                    assert (
                        len(action_id) == 1
                    )  # Exactly one action should be equivalent
                    action_id = int(action_id[0])

                # Compare to DN action
                if action != agent.actions[0]:
                    # obs_sim, reward_sim, done_sim, info_sim = obs.simulate(action)
                    reward_sim = reward
                    obs_dn, reward_dn, done_dn, info_dn = obs.simulate(agent.actions[0])
                else:
                    reward_sim, reward_dn = 0.0, 0.0

                dist, dist_status, dist_sub = agent.distance_to_ref_topology(
                    obs_next.topo_vect, obs_next.line_status
                )

                obs = obs_next
                actions.append(action_id)
                actions_info.append(action.as_dict())
                time_steps.append(t)

                rewards.append(float(reward))
                rewards_sim.append(float(reward_sim))
                rewards_dn.append(float(reward_dn))

                observations.append(obs)
                results.append(agent.result)

                distances.append(dist)
                distances_status.append(dist_status)
                distances_sub.append(dist_sub)

            total_return = compute_returns(rewards)[0]
            chronic_data.append(
                {
                    "chronic_idx": chronic_idx,
                    "chronic_org_idx": chronic_org_idx,
                    "chronic_name": chronic_name,
                    "actions": actions,
                    "actions_info": actions_info,
                    "time_steps": time_steps,
                    "rewards": rewards,
                    "return": total_return,
                    "chronic_length": chronic_len,
                    "duration": t,
                    "distances": distances,
                    "distances_status": distances_status,
                    "distances_sub": distances_sub,
                    "observations": observations,
                    "results": results,
                }
            )

        if chronic_data:
            chronic_data = pd.DataFrame(chronic_data)
            chronic_data = chronic_data.set_index("chronic_idx")
        else:
            chronic_data = pd.DataFrame()

        return chronic_data
