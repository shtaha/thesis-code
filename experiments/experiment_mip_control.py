import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class ExperimentMIPControl:
    @staticmethod
    def _plot_and_save(measurements, env, agent, title=None, save_dir=None):
        colors = list(mcolors.TABLEAU_COLORS)
        t = measurements["t"]

        fig, ax = plt.subplots()
        ax.plot(t, measurements["reward"], label="Reward - ENV")
        ax.plot(t, measurements["reward_est"], label="Reward - EST")
        ax.set_xlabel("Time step t")
        ax.set_ylabel("Reward")
        ax.legend()
        ax.set_title(title)
        fig.show()
        if save_dir:
            fig.savefig(os.path.join(save_dir, "rewards.png"))

        fig, ax = plt.subplots()
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
        ax.set_title(title)
        fig.show()
        if save_dir:
            fig.savefig(os.path.join(save_dir, "generator-productions.png"))

        fig, ax = plt.subplots()
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
        ax.set_title(title)
        fig.show()
        if save_dir:
            fig.savefig(os.path.join(save_dir, "power-flows.png"))

        fig, ax = plt.subplots()
        sns.distplot(
            measurements["action_id"],
            ax=ax,
            bins=len(agent.actions),
            hist=True,
            kde=False,
        )
        ax.set_xlabel("Action Id")
        ax.set_ylabel("Count")
        ax.set_xlim([0, len(agent.actions)])
        ax.set_title(title)
        fig.show()
        if save_dir:
            fig.savefig(os.path.join(save_dir, "action-ids.png"))

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

            print(f"STEP {t}")
            if verbose:
                print(action, "\n")

            reward_est = agent.get_reward()
            res_line, res_gen = agent.compare_with_observation(obs_next, verbose=False)
            action_id = [
                idx
                for idx, agent_action in enumerate(agent.actions)
                if action == agent_action
            ]
            assert len(action_id) == 1  # Exactly one action should be equivalent
            action_id = int(action_id[0])

            measurement = dict()
            measurement["t"] = t
            measurement["e"] = e
            measurement["reward"] = reward
            measurement["reward_est"] = reward_est
            measurement["action_id"] = action_id

            for gen_id in res_gen.index:
                measurement[f"gen-{gen_id}"] = res_gen["p_pu"][gen_id]
                measurement[f"env-gen-{gen_id}"] = res_gen["env_p_pu"][gen_id]

            for line_id in res_line.index:
                measurement[f"line-{line_id}"] = res_line["p_pu"][line_id]
                measurement[f"env-line-{line_id}"] = res_line["env_p_pu"][line_id]

            measurements.append(measurement)

            obs = obs_next
            if done:
                print("DONE", "\n")
                obs = env.reset()
                e = e + 1

        measurements = pd.DataFrame(measurements)
        return measurements

    def evaluate_performance(
        self, case, agent, save_dir=None, n_steps=100, verbose=False, **kwargs,
    ):
        env = case.env

        measurements = self._runner_mip_control(
            env, agent, n_steps=n_steps, verbose=verbose, **kwargs
        )

        if verbose:
            print("MEASUREMENTS:\n" + measurements.to_string())

        self._plot_and_save(
            measurements, env=env, agent=agent, title=env.name, save_dir=save_dir
        )
        self._save_csv(data=measurements, save_dir=save_dir)
