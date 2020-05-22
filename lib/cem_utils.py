import argparse
import time

import numpy as np
import tensorflow as tf
from grid2op.Agent import BaseAgent, AgentWithConverter

from lib.rl_utils import compute_returns
from lib.visualizer import print_trainable_variables


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name", default="l2rpn_case14_sandbox", type=str, help="Environment name."
    )
    parser.add_argument(
        "--n_iters", default=100, type=int, help="Number of training iterations."
    )
    parser.add_argument(
        "--n_samples",
        default=200,
        type=int,
        help="Number of samples CEM algorithm chooses from on each iteration.",
    )
    parser.add_argument(
        "--max_timesteps", default=100, type=int, help="Maximum episode length."
    )
    parser.add_argument(
        "--best_frac",
        default=0.1,
        type=float,
        help="Fraction of top samples used to calculate mean and variance of next iteration",
    )
    return parser.parse_args()


def episode_rollout(env, action_set, actor, max_timesteps=100, render=False, fps=100):
    state = env.reset()
    if render:
        env.render(mode="human")

    states = []
    actions = []
    rewards = []

    for t in range(max_timesteps):  # t = 0, 1, 2, ..., T-1
        states.append(state.rho)  # s_t

        action = actor.act(state)

        actions.append(action_set.index(action))  # a_t

        next_state, reward, done, _ = env.step(action)  # s_t+1, r_t+1

        rewards.append(reward)  # r_t+1

        # (state, action, reward, next_states)  (s_t, a_t, r_t+1, s_t+1)
        state = next_state

        if render:
            time.sleep(1.0 / fps)
            env.render()

        if done:
            break

    # states: s_0, s_1, s_2, ..., s_T-1
    # actions: a_0, a_1, ..., a_T-1
    # rewards: r_1, r_2, ..., r_T
    return np.array(states), np.array(actions), np.array(rewards)


class AgentCEMG(AgentWithConverter, tf.keras.Model):
    def __init__(self, env, n_states, action_set, n_hidden, alpha, rho_threshold=0.8):
        self.n_states = n_states
        self.n_actions = len(action_set)
        self.action_set = action_set
        self.rho_threshold = rho_threshold

        tf.keras.Model.__init__(self)
        AgentWithConverter.__init__(self, action_space=env.action_space)

        # Actor
        self.layer_input = tf.keras.layers.Dense(
            n_hidden[0],
            input_shape=(self.n_states,),
            activation=tf.nn.relu,
            name="actor_layer_input",
            trainable=True,
        )

        self.layers_hidden = []
        for layer, layer_n_hidden in enumerate(n_hidden[1:]):
            self.layers_hidden.append(
                tf.keras.layers.Dense(
                    layer_n_hidden, activation=tf.nn.relu, name=f"actor_layer_hidden_{layer}", trainable=True
                )
            )

        self.layer_output = tf.keras.layers.Dense(
            self.n_actions, activation=None, name="actor_layer_output", trainable=True
        )

        # Training
        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=alpha),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )

        # Initialize variables
        observation_encoded = np.random.rand(10, self.n_states).astype(
            np.float32
        )  # (None, n_states)
        _ = self.my_act(observation_encoded)  # (None, n_actions) logits

        print_trainable_variables(self)

    @tf.function(autograph=False)
    def call(self, state):
        a = self.layer_input(state)  # (None , n_hidden_l1)
        for layer in self.layers_hidden:
            a = layer(a)  # (None, n_hidden)

        logits = self.layer_output(a)  # (None, n_actions)
        return logits

    def convert_obs(self, observation):
        # return np.reshape(
        #     observation.to_vect(), (-1, self.n_states)
        # )

        return np.reshape(
            observation.rho, (-1, self.n_states)
        )

    def convert_act(self, encoded_act: np.ndarray):
        if encoded_act.size > 1:
            return [self.action_set[action_id] for action_id in encoded_act]
        else:
            return self.action_set[encoded_act]

    def my_act(self, observation_encoded, reward=0.0, done=False):
        logits = self(observation_encoded)  # (None, n_actions)
        action_ids = tf.argmax(logits, axis=-1).numpy()  # (None, )
        return action_ids  # (None, )

    def act(self, observation, reward=0.0, done=False):
        if np.greater(observation.rho, self.rho_threshold).any():
            observation_encoded = self.convert_obs(observation)  # (1, n_states)
            # print("observation_encoded", observation_encoded.shape, observation_encoded)
            encoded_act = self.my_act(observation_encoded, reward, done)  # (1, )
            # print("encoded act", encoded_act.shape, encoded_act)
            encoded_act = np.squeeze(encoded_act)
            return self.convert_act(encoded_act)
        else:
            return self.action_set[0]


class RandomAgent(AgentWithConverter):
    def __init__(self, env, n_states, actions, seed=0):
        AgentWithConverter.__init__(self, env.action_space)
        np.random.seed(seed)

        self.actions = actions
        self.n_states = n_states
        self.n_actions = len(self.actions)

        observation_encoded = np.random.rand(3, self.n_states).astype(
            np.float32
        )  # (None, n_states)
        _ = self.my_act(observation_encoded)  # (None, n_actions)

    def my_act(self, transformed_observation: np.ndarray, reward=0.0, done=False):
        action_ids = np.random.randint(
            0, self.n_actions, transformed_observation.shape[0]
        )  # (None, )
        return action_ids

    def convert_obs(self, observation):
        return np.reshape(
            observation.to_vect(), (-1, self.n_states)
        )  # (1, n_states)

    def convert_act(self, encoded_act: np.ndarray):
        if encoded_act.size > 1:
            return [self.actions[action_id] for action_id in encoded_act]
        else:
            return self.actions[encoded_act]

    def act(self, observation, reward=0.0, done=False):
        observation_encoded = self.convert_obs(observation)  # (1, n_states)
        encoded_act = self.my_act(observation_encoded, reward, done)  # (1, n_actions)
        encoded_act = np.squeeze(encoded_act)
        return self.convert_act(encoded_act)


class DoNothingAgent(BaseAgent):
    def __init__(self, action_space):
        BaseAgent.__init__(self, action_space)

    def act(self, observation, reward=0.0, done=False):
        res = self.action_space({})
        return res


def main_cem(args, env, agent, sampling_agent):
    best_n_samples = int(np.ceil(args.best_frac * args.n_samples))

    test_returns = []
    print("TRAINING")
    for i in range(args.n_iters):
        sample_returns = []
        sample_states = []
        sample_actions = []
        for j in range(args.n_samples):
            states, actions, rewards = episode_rollout(env, agent.action_set, sampling_agent, args.max_timesteps)
            total_return, returns = compute_returns(rewards)

            sample_states.append(states)
            sample_actions.append(actions)
            sample_returns.append(total_return)

        sample_returns = np.array(sample_returns)
        best_indices = np.argsort(sample_returns)[::-1][:best_n_samples]

        data = np.vstack([sample_states[k] for k in best_indices])
        targets = np.vstack([np.reshape(sample_actions[k], (-1, 1)) for k in best_indices])

        # Learn policy
        agent.fit(data, targets, verbose=1, epochs=3)

        # Test actor
        if i % 5 == 0:
            _, actions, rewards = episode_rollout(env, agent.action_set, agent, args.max_timesteps, render=False)
            print(f"unique: R {np.unique(targets)} CEM {np.unique(actions)}")
            print_trainable_variables(agent)

            total_return, _ = compute_returns(rewards)
            print(f"{i}: {sample_returns.mean()} {total_return}")
            test_returns.append(total_return)
        else:
            print(f"{i} {sample_returns.mean()}")

    return test_returns
