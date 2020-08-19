import os

import numpy as np
import pandas as pd

from lib.chronics import get_sorted_chronics
from lib.rl_utils import compute_returns
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
        for file, f in files:
            f.close()
            try:
                if delete_file:
                    os.remove(os.path.join(save_dir, file))
            except PermissionError as e:
                print(e)


class ExperimentFailureSwitchingMixin:
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

            chronic_org_idx = chronics.index(chronic)
            env.chronics_handler.tell_id(chronic_org_idx - 1)  # Set chronic id

            obs = env.reset()
            chronic_len = env.chronics_handler.real_data.data.max_iter + 1

            chronic_name = "/".join(
                os.path.normpath(env.chronics_handler.get_id()).split(os.sep)[-3:]
            )

            pprint("    - Chronic:", chronic_name)

            if case.name == "Case L2RPN 2020 WCCI":
                chronic_name = env.chronics_handler.get_name()[
                    -len("Scenario_") :
                ].split("_")
            else:
                chronic_name = env.chronics_handler.get_name()

            done = False
            t = 0
            actions = []
            actions_info = []
            rewards = []
            rewards_sim = []
            rewards_dn = []
            observations = []
            distances = []
            distances_status = []
            distances_sub = []
            time_steps = []
            while not done:
                action = agent.act(obs, done)
                obs_next, reward, done, info = env.step(action)

                t = env.chronics_handler.real_data.data.current_index

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
                    "rewards_sim": rewards_sim,
                    "rewards_dn": rewards_dn,
                    "return": total_return,
                    "chronic_length": chronic_len,
                    "duration": t,
                    "observations": observations,
                    "distances": distances,
                    "distances_status": distances_status,
                    "distances_sub": distances_sub,
                }
            )

            # TODO: Delete break
            # if chronic_idx > 0:
            #     break

        if chronic_data:
            chronic_data = pd.DataFrame(chronic_data)
            chronic_data = chronic_data.set_index("chronic_idx")
        else:
            chronic_data = pd.DataFrame()

        return chronic_data

    @staticmethod
    def _load_done_chronics(file_name, save_dir=None):
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

            if "observations" in chronic_data.columns:
                chronic_data = chronic_data.drop("observations", axis=1)

            chronic_data.to_csv(os.path.join(save_dir, file_name + ".csv"))
