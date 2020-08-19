import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyPDF2 import PdfFileMerger

from lib.constants import Constants as Const
from .experiment_base import ExperimentBase, ExperimentFailureSwitchingMixin


class ExperimentFailure(ExperimentBase, ExperimentFailureSwitchingMixin):
    def analyze_failures(self, case, agent, save_dir=None, verbose=False):
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

        for chronic_idx in chronic_data[agent_names[0]].index:
            chronic_name = chronic_data[agent_names[0]].iloc[chronic_idx][
                "chronic_name"
            ]

            fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
            for agent_name in chronic_data:
                if chronic_idx in chronic_data[agent_name].index:
                    t = chronic_data[agent_name].iloc[chronic_idx]["time_steps"]
                    rewards = chronic_data[agent_name].iloc[chronic_idx]["rewards"]

                    ax.plot(
                        t, rewards, linewidth=1, label=agent_name,
                    )

            ax.set_xlabel("Time step t")
            ax.set_ylabel("Reward")
            ax.legend()
            fig.suptitle(f"{case_name} - Chronic {chronic_name}")

            if save_dir:
                file_name = f"agents-chronic-{chronic_name}-"
                fig.savefig(os.path.join(save_dir, file_name + "rewards"))
            plt.close(fig)

        fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
        chronic_names = chronic_data[agent_names[0]]["chronic_name"]

        x = np.arange(len(chronic_names))
        width = 0.3 / len(agent_names)

        for agent_id, agent_name in enumerate(agent_names):
            y = chronic_data[agent_name]["duration"]
            ax.barh(x + agent_id * width, y, width, left=0.001)

        ax.set_yticks(x)
        ax.set_yticklabels(chronic_names)
        ax.invert_yaxis()
        ax.legend(tuple(agent_names))

        ax.set_ylabel("Chronic")
        ax.set_xlabel("Chronic duration")
        fig.suptitle(f"{case_name} - Chronic durations")

        if save_dir:
            file_name = f"_agents-chronics-"
            fig.savefig(os.path.join(save_dir, file_name + "durations"))
        plt.close(fig)

        self.aggregate_by_chronics(save_dir, delete_file=delete_file)

    def aggregate_by_chronics(self, save_dir, delete_file=True):
        merger = PdfFileMerger()
        chronic_files = []

        plot_name = "rewards"
        for file in os.listdir(save_dir):
            if "agents-chronic-" in file and plot_name in file:
                f = open(os.path.join(save_dir, file), "rb")
                chronic_files.append((file, f))
                merger.append(f)

        with open(
            os.path.join(save_dir, "_" + f"agents-chronics-{plot_name}.pdf"), "wb"
        ) as f:
            merger.write(f)

        self.close_files(chronic_files, save_dir, delete_file=delete_file)
