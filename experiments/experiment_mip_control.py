import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from lib.constants import Constants as Const
from lib.visualizer import print_action, pprint


class ExperimentMIPControl:
    @staticmethod
    def _plot_and_save(
        measurements, env, agent, title=None, save_dir=None, fig_format=Const.OUT_FORMAT
    ):
        colors = Const.COLORS
        t = measurements["t"]

        fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
        ax.plot(t, measurements["reward"], label="Reward - ENV")
        ax.plot(t, measurements["reward-est"], label="Reward - EST")
        ax.set_xlabel("Time step t")
        ax.set_ylabel("Reward")
        ax.legend()
        fig.suptitle(title)

        fig.show()
        if save_dir:
            fig.savefig(os.path.join(save_dir, "rewards" + fig_format))

        fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
        ax.plot(t, measurements["rho"], label="Rho - EST")
        ax.plot(t, measurements["env-rho"], label="Rho - ENV")
        ax.plot(
            t, np.ones_like(t), c="tab:red", linestyle="-", linewidth=2,
        )
        ax.set_xlabel("Time step t")
        ax.set_ylabel("Maximum relative power flow - Rho")
        ax.set_ylim((0.0, 2.0))
        ax.legend()
        fig.suptitle(title)

        fig.show()
        if save_dir:
            fig.savefig(os.path.join(save_dir, "rho" + fig_format))

        fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
        for gen_id in range(env.n_gen):
            color = colors[gen_id % len(colors)]
            ax.plot(
                t,
                measurements[f"gen-{gen_id}"],
                label=f"Gen-{gen_id}",
                c=color,
                linestyle="-",
                linewidth=1,
            )
            ax.plot(
                t,
                measurements[f"env-gen-{gen_id}"],
                label=f"Gen-{gen_id} - ENV",
                c=color,
                linestyle="--",
                linewidth=1,
            )

        ax.set_xlabel("Time step t")
        ax.set_ylabel("P [p.u.]")
        fig.suptitle(title)
        if env.n_gen < 3:
            ax.legend()

        fig.show()
        if save_dir:
            fig.savefig(os.path.join(save_dir, "generators_p" + fig_format))

        fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
        for gen_id in range(env.n_gen):
            color = colors[gen_id % len(colors)]
            ax.plot(
                t,
                measurements[f"env-gen-{gen_id}-q"],
                label=f"Gen-{gen_id} - ENV",
                c=color,
                linestyle="--",
                linewidth=1,
            )

        ax.set_xlabel("Time step t")
        ax.set_ylabel("Q [p.u.]")
        fig.suptitle(title)
        if env.n_gen < 3:
            ax.legend()

        fig.show()
        if save_dir:
            fig.savefig(os.path.join(save_dir, "generators_q" + fig_format))

        fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
        for gen_id in range(env.n_gen):
            color = colors[gen_id % len(colors)]
            ax.plot(
                t,
                measurements[f"env-q-p-{gen_id}"],
                label=f"Gen-{gen_id} - ENV",
                c=color,
                linestyle="--",
                linewidth=1,
            )

        ax.set_xlabel("Time step t")
        ax.set_ylabel("Q/P [p.u.]")
        fig.suptitle(title)
        if env.n_gen < 3:
            ax.legend()

        fig.show()
        if save_dir:
            fig.savefig(os.path.join(save_dir, "generators_q-p" + fig_format))

        fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
        for line_id in range(env.n_line):
            color = colors[line_id % len(colors)]
            ax.plot(
                t,
                measurements[f"line-{line_id}"],
                label=f"Line-{line_id}",
                c=color,
                linestyle="-",
                linewidth=1,
            )
            ax.plot(
                t,
                measurements[f"env-line-{line_id}"],
                label=f"Line-{line_id} - ENV",
                c=color,
                linestyle="--",
                linewidth=1,
            )

        ax.set_xlabel("Time step t")
        ax.set_ylabel("P [p.u.]")
        fig.suptitle(title)

        fig.show()
        if save_dir:
            fig.savefig(os.path.join(save_dir, "power-flows" + fig_format))

        fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
        ax.plot(
            t,
            measurements["env-gens-p"],
            label="Total production",
            linestyle="--",
            linewidth=1,
        )
        ax.plot(
            t,
            measurements["env-loads-p"],
            label="Total demand",
            linestyle="--",
            linewidth=1,
        )
        ax.set_xlabel("Time step t")
        ax.set_ylabel("P [p.u.]")
        ax.set_ylim(bottom=0.0)
        ax.legend()
        fig.suptitle(title)

        fig.show()
        if save_dir:
            fig.savefig(os.path.join(save_dir, "production-demand" + fig_format))

        fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
        sns.distplot(
            measurements["action-id"],
            ax=ax,
            bins=len(agent.actions),
            hist=True,
            kde=False,
        )
        ax.set_xlabel("Action Id")
        ax.set_ylabel("Count")
        ax.set_xlim([0, len(agent.actions)])
        fig.suptitle(title)

        fig.show()
        if save_dir:
            fig.savefig(os.path.join(save_dir, "action-ids" + fig_format))

    @staticmethod
    def _save_csv(data, save_dir):
        data.to_csv(os.path.join(save_dir, "measurements.csv"))

    @staticmethod
    def _runner_mip_control(
        env, agent, n_steps=100, verbose=False, **kwargs,
    ):
        np.random.seed(0)
        env.seed(0)

        measurements = []

        e = 0  # Episode counter
        done = False
        obs = env.reset()
        for t in range(n_steps):
            action = agent.act(obs, done, **kwargs)
            obs_next, reward, done, info = env.step(action)

            pprint("Step:", t)
            if verbose:
                print_action(action)

            reward_est = agent.get_reward()
            res_line, res_gen = agent.compare_with_observation(obs_next, verbose=False)
            action_id = [
                idx
                for idx, agent_action in enumerate(agent.actions)
                if action == agent_action
            ]
            if len(action_id) != 1:
                print(action_id)
                print(action)

            assert len(action_id) == 1  # Exactly one action should be equivalent
            action_id = int(action_id[0])

            measurement = dict()
            measurement["t"] = t
            measurement["e"] = e
            measurement["reward"] = reward
            measurement["reward-est"] = reward_est
            measurement["action-id"] = action_id
            measurement["rho"] = res_line["rho"].max()
            measurement["env-rho"] = res_line["env_rho"].max()
            measurement["env-gens-p"] = obs.prod_p.sum()
            measurement["env-loads-p"] = obs.load_p.sum()

            for gen_id in res_gen.index:
                measurement[f"gen-{gen_id}"] = res_gen["p_pu"][gen_id]
                measurement[f"env-gen-{gen_id}"] = res_gen["env_p_pu"][gen_id]
                measurement[f"env-gen-{gen_id}-q"] = res_gen["env_q_pu"][gen_id]
                measurement[f"env-q-p-{gen_id}"] = res_gen["env_gen_q_p"][gen_id]

            for line_id in res_line.index:
                measurement[f"line-{line_id}"] = res_line["p_pu"][line_id]
                measurement[f"env-line-{line_id}"] = res_line["env_p_pu"][line_id]

            measurements.append(measurement)

            obs = obs_next
            if done:
                print("DONE\n")
                obs = env.reset()
                e = e + 1

        measurements = pd.DataFrame(measurements)
        return measurements

    def evaluate_performance(
        self, case, agent, save_dir=None, n_steps=100, verbose=False, **kwargs,
    ):
        env = case.env
        case_name = self._get_case_name(case)

        measurements = self._runner_mip_control(
            env, agent, n_steps=n_steps, verbose=verbose, **kwargs
        )

        if verbose:
            print(
                "MEASUREMENTS:\n"
                + measurements[
                    [
                        "t",
                        "e",
                        "reward",
                        "reward-est",
                        "action-id",
                        "rho",
                        "env-rho",
                        "env-gens-p",
                        "env-loads-p",
                    ]
                ].to_string()
            )

        self._plot_and_save(
            measurements,
            env=env,
            agent=agent,
            title=f"{case_name} - {agent.name}",
            save_dir=save_dir,
        )
        self._save_csv(data=measurements, save_dir=save_dir)

    @staticmethod
    def _get_case_name(case):
        env_pf = "AC"
        if case.env.parameters.ENV_DC:
            env_pf = "DC"

        return f"{case.name} ({env_pf})"
