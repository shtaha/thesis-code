import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lib.constants import Constants as Const
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

    @staticmethod
    def close_files(files, save_dir, delete_file=True):
        # delete_file = False
        for file, f in files:
            f.close()
            try:
                if delete_file:
                    os.remove(os.path.join(save_dir, file))
            except PermissionError as e:
                print(e)

    @staticmethod
    def _load_done_chronics_from_pickle(file_name, save_dir=None):
        if save_dir and file_name + ".pkl" in os.listdir(save_dir):
            chronic_data = pd.read_pickle(os.path.join(save_dir, file_name + ".pkl"))
            done_chronic_indices = chronic_data.index
        else:
            chronic_data = pd.DataFrame()
            done_chronic_indices = []

        return chronic_data, done_chronic_indices

    @staticmethod
    def _save_chronics(chronic_data, file_name, save_dir=None):
        if save_dir:
            chronic_data.to_pickle(os.path.join(save_dir, file_name + ".pkl"))


class ExperimentMixin:
    @staticmethod
    def _plot_rewards(
        chronic_data, case_name, chronic_idx, chronic_name, save_dir=None
    ):
        fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
        for agent_name in chronic_data:
            if chronic_idx in chronic_data[agent_name].index:
                t = chronic_data[agent_name].loc[chronic_idx]["time_steps"]
                rewards = chronic_data[agent_name].loc[chronic_idx]["rewards"]

                ax.plot(
                    t, rewards, linewidth=1, label=agent_name,
                )

        ax.set_xlabel("Time step t")
        ax.set_ylabel("Reward")
        ax.legend()
        fig.suptitle(f"{case_name} - Chronic {chronic_name}")
        fig.tight_layout()
        if save_dir:
            file_name = f"agents-chronic-" + "{:05}".format(chronic_idx) + "-"
            fig.savefig(os.path.join(save_dir, file_name + "rewards"))
        plt.close(fig)

    @staticmethod
    def _plot_relative_flow(
        chronic_data, case_name, chronic_idx, chronic_name, save_dir=None
    ):
        fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
        for agent_name in chronic_data:
            if chronic_idx in chronic_data[agent_name].index:
                chronic = chronic_data[agent_name].loc[chronic_idx]
                t = chronic["time_steps"]

                if "observation" in chronic:
                    rho = [np.max(obs.rho) for obs in chronic["observations"]]
                else:
                    rho = np.array(chronic["rhos"])

                ax.plot(
                    t, rho, linewidth=1, label=agent_name,
                )

        ax.set_xlabel("Time step t")
        ax.set_ylabel(r"Relative flow $\rho$")
        ax.legend()
        ax.set_ylim([0.0, 2.0])
        fig.suptitle(f"{case_name} - Chronic {chronic_name}")
        fig.tight_layout()
        if save_dir:
            file_name = f"agents-chronic-" + "{:05}".format(chronic_idx) + "-"
            fig.savefig(os.path.join(save_dir, file_name + "rhos"))
        plt.close(fig)

    @staticmethod
    def _plot_distances(
        chronic_data, dist, ylabel, case_name, chronic_idx, chronic_name, save_dir
    ):
        fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
        for agent_name in chronic_data:
            if chronic_idx in chronic_data[agent_name].index:
                t = chronic_data[agent_name].loc[chronic_idx]["time_steps"]
                distances = chronic_data[agent_name].loc[chronic_idx][dist]
                actions = chronic_data[agent_name].loc[chronic_idx]["actions"]

                ax.plot(
                    t, distances, linewidth=0.5, label=agent_name,
                )

                # for i in range(len(t)):
                #     if isinstance(actions[i], int):
                #         action_id = actions[i]
                #         if action_id != 0:
                #             ax.text(t[i], distances[i], str(action_id), fontsize=2)

        ax.set_xlabel("Time step t")
        ax.set_ylabel(ylabel)
        ax.legend()
        fig.suptitle(f"{case_name} - Chronic {chronic_name}")
        fig.tight_layout()
        if save_dir:
            file_name = f"agents-chronic-" + "{:05}".format(chronic_idx) + "-"
            fig.savefig(os.path.join(save_dir, file_name + dist))
        plt.close(fig)
