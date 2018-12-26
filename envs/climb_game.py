from numpy import *
from .env import Env

# reward_table = array([
#     [[11, 11], [-30, -30], [0, 0]],
#     [[-30, -30, [7, 7], [6, 6]]],
#     [[0, 0], [0, 0], [5, 5]]
# ])
reward_table = array([
    [11, -30, 0],
    [-30, 7, 6],
    [0, 0, 5]
])


class ClimbGame(Env):
    def __init__(self, *args, **kwargs):
        self.n_agent = 2
        self.episode_length = kwargs.get('episode_length') or 40
        self.n_action = 3
        self.obs_size = 1
    
    def reset(self):
        self.step_count = 0
        return self.const_obs

    def step(self, actions):
        # assert len(actions) == 2
        self.step_count += 1
        a, b = actions
        return self.const_obs, reward_table[a][b], self.step_count > self.episode_length
    
    
    @property
    def const_obs(self):
        return zeros((self.n_agent,1))

    