from grid2op.Agent import BaseAgent

from grid2op.Agent.AgentWithConverter import AgentWithConverter


class MLAgent(BaseAgent):
    def __init__(self, action_space):
        BaseAgent.__init__(self, action_space)

    def reset(self, obs):
        pass

    def seed(self, seed):
        return super().seed(seed)

    def act(self, observation, reward, done=False):
        pass
