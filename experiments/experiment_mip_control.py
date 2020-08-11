import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from lib.constants import Constants as Const
from lib.visualizer import print_action, pprint
from .experiment_base import ExperimentBase


class ExperimentMIPControl(ExperimentBase):
    def evaluate_performance(
        self, case, agent, save_dir=None, n_steps=100, verbose=False,
    ):
        env = case.env
        case_name = self._get_case_name(case)

        self.print_experiment("Control Performance")

        agent.set_kwargs()
        agent.print_agent(default=True)

        measurements = self._runner_mip_control(
            env, agent, n_steps=n_steps, verbose=verbose
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
            prefix=agent.name.replace(" ", "-").lower(),
        )
        self._save_csv(
            data=measurements,
            save_dir=save_dir,
            prefix=agent.name.replace(" ", "-").lower(),
        )

    @staticmethod
    def _runner_mip_control(env, agent, n_steps=100, verbose=False):
        np.random.seed(1)
        env.seed(1)

        measurements = []

        e = 0  # Episode counter
        done = False
        obs = env.reset()
        agent.reset(obs=obs)
        for t in range(n_steps):
            action = agent.act(obs, done)
            obs_next, reward, done, info = env.step(action)

            pprint("Step:", t)
            if verbose:
                print_action(action)

            reward_est = agent.get_reward()
            res_line, res_gen = agent.compare_with_observation(
                obs_next, verbose=verbose
            )
            dist, dist_status, dist_sub = agent.distance_to_ref_topology(
                obs_next.topo_vect, obs_next.line_status
            )

            action_id = [
                idx
                for idx, agent_action in enumerate(agent.actions)
                if action == agent_action
            ]

            if "unitary_action" in agent.model_kwargs:
                if not agent.model_kwargs["unitary_action"] and len(action_id) != 1:
                    print_action(action)
                    action_id = np.nan
                else:
                    assert (
                        len(action_id) == 1
                    )  # Exactly one action should be equivalent
                    action_id = int(action_id[0])
            else:
                assert len(action_id) == 1  # Exactly one action should be equivalent
                action_id = int(action_id[0])

            measurement = dict()
            measurement["t"] = t
            measurement["e"] = e
            measurement["reward"] = reward
            measurement["reward-est"] = reward_est
            measurement["dist"] = dist
            measurement["dist_status"] = dist_status
            measurement["dist_sub"] = dist_sub
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
                agent.reset(obs=obs)
                e = e + 1

        measurements = pd.DataFrame(measurements)
        return measurements

    @staticmethod
    def _plot_and_save(
        measurements, env, agent, title=None, save_dir=None, prefix=None,
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
            file_name = "rewards"
            if prefix:
                file_name = prefix + "-" + file_name

            fig.savefig(os.path.join(save_dir, file_name))

        fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
        ax.plot(t, measurements["dist"], label="Overall")
        ax.plot(t, measurements["dist_status"], label="Line")
        ax.plot(t, measurements["dist_sub"], label="Substation")
        ax.set_xlabel("Time step t")
        ax.set_ylabel("Unitary action distance to reference topology")
        ax.legend()
        fig.suptitle(title)

        fig.show()
        if save_dir:
            file_name = "distance"
            if prefix:
                file_name = prefix + "-" + file_name

            fig.savefig(os.path.join(save_dir, file_name))

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
            file_name = "rho"
            if prefix:
                file_name = prefix + "-" + file_name

            fig.savefig(os.path.join(save_dir, file_name))

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
            file_name = "generators_p"
            if prefix:
                file_name = prefix + "-" + file_name

            fig.savefig(os.path.join(save_dir, file_name))

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
            file_name = "generators_q"
            if prefix:
                file_name = prefix + "-" + file_name

            fig.savefig(os.path.join(save_dir, file_name))

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
            file_name = "generators_q-p"
            if prefix:
                file_name = prefix + "-" + file_name

            fig.savefig(os.path.join(save_dir, file_name))

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
            file_name = "power-flows"
            if prefix:
                file_name = prefix + "-" + file_name

            fig.savefig(os.path.join(save_dir, file_name))

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
            file_name = "production-demand"
            if prefix:
                file_name = prefix + "-" + file_name

            fig.savefig(os.path.join(save_dir, file_name))

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
            file_name = "action-ids"
            if prefix:
                file_name = prefix + "-" + file_name

            fig.savefig(os.path.join(save_dir, file_name))

    @staticmethod
    def _save_csv(data, save_dir, prefix=None):
        file_name = "measurements"
        if prefix:
            file_name = prefix + "-" + file_name

        data.to_csv(os.path.join(save_dir, file_name + ".csv"))

    @staticmethod
    def compare_agents(save_dir):
        measurements = dict()

        for file in os.listdir(save_dir):
            if "measurements.csv" in file:
                agent_name = file[: -len("-measurements.csv")]
                measurements[agent_name] = pd.read_csv(os.path.join(save_dir, file))

        fig_env, ax_env = plt.subplots(figsize=Const.FIG_SIZE)
        fig_est, ax_est = plt.subplots(figsize=Const.FIG_SIZE)
        for agent_name in measurements:
            name = agent_name.replace("-", " ").capitalize()
            t = measurements[agent_name]["t"]
            ax_env.plot(t, measurements[agent_name]["reward"], label=name)
            ax_est.plot(t, measurements[agent_name]["reward-est"], label=name)

        ax_env.set_xlabel("Time step t")
        ax_env.set_ylabel("Reward")
        ax_est.set_xlabel("Time step t")
        ax_est.set_ylabel("Reward")
        ax_env.legend()
        ax_est.legend()
        fig_env.suptitle("Agent Comparison - Reward ENV")
        fig_est.suptitle("Agent Comparison - Reward EST")

        fig_env.show()
        fig_est.show()
        if save_dir:
            file_name = "agents-rewards"
            fig_env.savefig(os.path.join(save_dir, file_name + "-env"))
            fig_est.savefig(os.path.join(save_dir, file_name + "-est"))
