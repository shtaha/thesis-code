import numpy as np


class ExperimentMIPControl:
    @staticmethod
    def _runner_mip_control(
        env, agent, n_steps=100, verbose=False, **kwargs,
    ):
        np.random.seed(1)

        if verbose:
            agent.grid.print_grid()

        obs = env.reset()
        for t in range(n_steps):
            _, _, action = agent.act(obs, **kwargs)

            print(f"STEP {t}")
            print(action, "\n")

            obs_next, reward, done, info = env.step(action)

            if verbose:
                agent.compare_with_observation(obs_next, verbose=verbose)

            obs = obs_next
            if done:
                print("DONE", "\n")
                obs = env.reset()

            agent.update(obs, reset=done, verbose=verbose)

    def evaluate_performance(self, env, agent, n_steps=100, verbose=False, **kwargs):
        self._runner_mip_control(env, agent, n_steps=n_steps, verbose=verbose, **kwargs)
