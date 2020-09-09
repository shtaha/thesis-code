import os

import numpy as np
from grid2op.Converter import ToVect

from lib.chronics import get_sorted_chronics
from lib.visualizer import pprint


class ExperienceCollector(object):
    def __init__(self, save_dir):
        self.save_dir = save_dir

        self.chronic_files = []
        self.chronic_ids = []

        self.data = None

        self.obses = None
        self.actions = None
        self.rewards = None
        self.dones = None
        self.reset()

    def reset(self):
        self.obses = []
        self.actions = []
        self.rewards = []
        self.dones = []

    @staticmethod
    def print_collector(phase):
        print("\n" + "-" * 80)
        pprint("", f"{phase} Experience")
        print("-" * 80)

    def collect(self, env, agent, n_chronics=10, n_steps=-1, verbose=False):
        self.print_collector("Collecting")
        agent.print_agent(default=verbose)

        agent_name = agent.name.replace(" ", "-").lower()
        self._load_chronics(agent_name=agent_name)

        chronics_dir, chronics, chronics_sorted = get_sorted_chronics(env=env)
        pprint("Chronics:", chronics_dir)

        done_chronic_ids = []
        for chronic_idx, chronic_name in enumerate(chronics_sorted):
            if len(done_chronic_ids) >= n_chronics:
                break

            # If chronic already done
            if chronic_idx in self.chronic_ids:
                continue

            # Environment specific filtering
            if env.name == "rte_case5_example":
                if chronic_idx not in [13, 14, 15, 16, 17, 16, 18, 19]:
                    continue
                pass
            elif env.name == "l2rpn_2019":
                if chronic_idx not in [0, 10, 100, 200, 196]:
                    continue
            elif env.name == "l2rpn_wcci_2020":
                if (chronic_idx % 240) not in [0, 1]:
                    continue

            chronic_org_idx = chronics.index(chronic_name)
            env.chronics_handler.tell_id(chronic_org_idx - 1)  # Set chronic id

            obs = env.reset()
            agent.reset(obs=obs)

            chronic_len = env.chronics_handler.real_data.data.max_iter
            chronic_path_name = "/".join(
                os.path.normpath(env.chronics_handler.get_id()).split(os.sep)[-3:]
            )
            pprint("    - Chronic:", chronic_path_name)

            t = 0
            done = False
            reward = np.nan

            """
                Collect data.
            """
            while not done and not (t >= n_steps > 0):
                action = agent.act(obs, reward=reward, done=done)
                obs_next, reward, done, info = env.step(action)
                self._add(obs, action, reward, done)

                t = env.chronics_handler.real_data.data.current_index

                if t % 200 == 0:
                    pprint("Step:", t)

                if done:
                    pprint("        - Length:", f"{t}/{chronic_len}")

                obs = obs_next

            self.obses.append(obs.to_vect())
            done_chronic_ids.append(chronic_idx)

            self._save_chronic(agent, chronic_idx, verbose)
            self.reset()

    def _add(self, obs, action, reward, done):
        self.obses.append(obs.to_vect())
        self.actions.append(action.to_vect())
        self.rewards.append(reward)
        self.dones.append(done)

    def _save_chronic(self, agent, chronic_idx, verbose=False):
        obses = np.array(self.obses)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)

        agent_name = agent.name.replace(" ", "-").lower()
        chronic_file = "{}-chronic-{:05}.npz".format(agent_name, chronic_idx)
        pprint("        - Experience saved to:", chronic_file)

        if verbose:
            pprint("        - Observations:", obses.shape)
            pprint("        - Actions:", actions.shape)
            pprint("        - Rewards:", rewards.shape)
            pprint("        - Dones:", dones.shape)
        else:
            pprint(
                "        - O A R D:",
                obses.shape,
                actions.shape,
                rewards.shape,
                dones.shape,
            )

        np.savez_compressed(
            os.path.join(self.save_dir, chronic_file),
            obses=obses,
            actions=actions,
            rewards=rewards,
            dones=dones,
        )

        self.chronic_files.append(chronic_file)
        self.chronic_ids.append(chronic_idx)

    def _load_chronics(self, agent_name):
        for chronic_file in os.listdir(self.save_dir):
            if "chronic-" in chronic_file and agent_name in chronic_file:
                chronic_idx = int(os.path.splitext(chronic_file)[0].split("-")[-1])
                self.chronic_files.append(chronic_file)
                self.chronic_ids.append(chronic_idx)

        if self.chronic_ids:
            pprint(
                "    - Done chronics:",
                ", ".join(map(lambda x: str(x), self.chronic_ids)),
            )

    def load_data(self, agent_name, env):
        self.print_collector("Loading")

        self._load_chronics(agent_name=agent_name)

        chronic_data = dict()
        for chronic_idx, chronic_file in zip(self.chronic_ids, self.chronic_files):
            chronic_data[chronic_idx] = dict()

            npz_file = np.load(os.path.join(self.save_dir, chronic_file))
            for key in npz_file.keys():
                chronic_data[chronic_idx][key] = npz_file[key]

        self.data = chronic_data
        self.transform_data(env=env)

    def transform_data(self, env):
        converter = ToVect(action_space=env.action_space)

        for chronic_idx in self.data:
            data_chronic = self.data[chronic_idx]

            obses = []
            actions = []
            for obs_vect in data_chronic["obses"]:
                obs = env.observation_space.from_vect(obs_vect)
                obses.append(obs)

            for action_vect in data_chronic["actions"]:
                action = converter.convert_act(action_vect)
                actions.append(action)

            self.data[chronic_idx]["obses"] = obses
            self.data[chronic_idx]["actions"] = actions

    def aggregate_data(self):
        observations = []
        actions = []
        rewards = []
        dones = []
        for chronic_idx in self.data:
            data_chronic = self.data[chronic_idx]
            obses_chronic = data_chronic["obses"]
            actions_chronic = data_chronic["actions"]
            rewards_chronic = data_chronic["rewards"]
            dones_chronic = data_chronic["dones"]

            pprint("Chronic:", chronic_idx)
            pprint(
                "        - O A R D:",
                len(obses_chronic),
                len(actions_chronic),
                rewards_chronic.shape,
                dones_chronic.shape,
            )

            observations.extend(obses_chronic[:-1])
            actions.extend(actions_chronic)
            rewards.extend(rewards_chronic.tolist())
            dones.extend(dones_chronic.tolist())

        return observations, actions, np.array(rewards), np.array(dones)
