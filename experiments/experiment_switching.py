import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyPDF2 import PdfFileMerger

from lib.constants import Constants as Const
from .experiment_base import ExperimentBase, ExperimentFailureSwitchingMixin


class ExperimentSwitching(ExperimentBase, ExperimentFailureSwitchingMixin):
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

        for chronic_idx in chronic_data[agent_names[0]].index:
            chronic_name = chronic_data[agent_names[0]].iloc[chronic_idx][
                "chronic_name"
            ]

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

            fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
            for agent_name in chronic_data:
                t = chronic_data[agent_name].iloc[chronic_idx]["time_steps"]
                rewards_sim = chronic_data[agent_name].iloc[chronic_idx]["rewards_sim"]
                rewards_dn = chronic_data[agent_name].iloc[chronic_idx]["rewards_dn"]
                actions = chronic_data[agent_name].iloc[chronic_idx]["actions"]

                delta = np.array(rewards_sim) - np.array(rewards_dn)

                ax.plot(
                    t, delta, linewidth=0.5, label=agent_name,
                )

                for i in range(len(t)):
                    action_id = actions[i]
                    if action_id != 0:
                        ax.text(t[i], delta[i], str(action_id), fontsize=2)

            ax.set_xlabel("Time step t")
            ax.set_ylabel("Advantage of selected action vs. do-nothing action")
            ax.legend()
            fig.suptitle(f"{case_name} - Chronic {chronic_name}")

            if save_dir:
                file_name = f"agents-chronic-{chronic_name}-"
                fig.savefig(os.path.join(save_dir, file_name + "advantages"))
            plt.close(fig)

        self.aggregate_by_chronics(save_dir, delete_file=delete_file)

    def aggregate_by_chronics(self, save_dir, delete_file=True):
        for plot_name in [
            "distances",
            "distances_status",
            "distances_sub",
            "advantages",
        ]:
            merger = PdfFileMerger()
            chronic_files = []
            for file in os.listdir(save_dir):
                if "agents-chronic-" in file and plot_name + ".pdf" in file:
                    f = open(os.path.join(save_dir, file), "rb")
                    chronic_files.append((file, f))
                    merger.append(f)

            with open(
                os.path.join(save_dir, "_" + f"agents-chronics-{plot_name}.pdf"), "wb"
            ) as f:
                merger.write(f)

            # Reset merger
            merger.pages = []
            merger.inputs = []
            merger.output = None

            self.close_files(chronic_files, save_dir, delete_file=delete_file)

    @staticmethod
    def _plot_distances(
        chronic_data, dist, ylabel, case_name, chronic_idx, chronic_name, save_dir
    ):
        fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
        for agent_name in chronic_data:
            t = chronic_data[agent_name].iloc[chronic_idx]["time_steps"]
            distances = chronic_data[agent_name].iloc[chronic_idx][dist]
            actions = chronic_data[agent_name].iloc[chronic_idx]["actions"]

            ax.plot(
                t, distances, linewidth=0.5, label=agent_name,
            )

            for i in range(len(t)):
                action_id = actions[i]
                if action_id != 0:
                    ax.text(t[i], distances[i], str(action_id), fontsize=2)

        ax.set_xlabel("Time step t")
        ax.set_ylabel(ylabel)
        ax.legend()
        fig.suptitle(f"{case_name} - Chronic {chronic_name}")

        if save_dir:
            file_name = f"agents-chronic-{chronic_name}-"
            fig.savefig(os.path.join(save_dir, file_name + dist))
        plt.close(fig)
