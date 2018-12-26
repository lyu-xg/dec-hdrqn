import numpy as np
from numpy.random import randint

NORTH = np.array([0, 1])
WEST = np.array([-1, 0])
SOUTH = - NORTH
EAST = - WEST
STAY = np.array([0, 0])

TRANSLATION_TABLE = [
    # [left, intended_direction, right]
    [WEST,  NORTH, EAST],
    [NORTH, EAST,  SOUTH],
    [EAST,  SOUTH, WEST],
    [SOUTH, WEST,  NORTH],
    [STAY,  STAY,  STAY]
]

class CaptureTarget:
    def __init__(self, n_target, n_agent, grid_dim, intermediate_r=False,
                 target_flick_prob=0.3, verbose=False):
        self.n_target, self.n_agent = n_target, n_agent
        self.multi_task = self.n_target != 1
        self.intermediate_reward = intermediate_r
        
        self.x_len, self.y_len = grid_dim
        self.half_x_len, self.half_y_len = self.x_len / 2, self.y_len / 2
        self.target_flick_prob = target_flick_prob
        self.verbose = verbose

        self.target_directions = [1] * n_target

        self.n_action = len(TRANSLATION_TABLE)
        self.obs_size = len(grid_dim) * 2 # agent position and target position
        
        assert  self.n_target == 1 or self.multi_task and self.n_agent > 1

    def reset(self):
        self.step_n = 0
        self.visited = np.zeros(self.n_target)

        # "game state" is really just the positions of all players and targets
        self.target_positions = np.stack([self.rand_position() for _ in range(self.n_target)])
        self.agent_positions  = np.stack([self.rand_position() for _ in range(self.n_agent)])
        assert self.target_positions.shape == (self.n_target, 2) 

        if self.target_captured():
            return self.reset()

        return self.get_obs()

    def get_obs(self):
        agt_pos_obs = self.normalize_positions(self.agent_positions)
        tgt_pos_obs = self.normalize_positions(self.target_positions)

        if self.n_agent > 1 and self.n_target == 1:
            tgt_pos_obs = np.tile(tgt_pos_obs, (self.n_agent, 1))

        tgt_pos_obs = self.flick(tgt_pos_obs, prob=self.target_flick_prob)
        return np.concatenate([agt_pos_obs, tgt_pos_obs], axis=1)

    def step(self, actions):
        # returns: observatioins, reward, is_terminal
        self.step_n += 1

        assert len(actions) == self.n_agent

        # latent state transition
        self.target_positions = self.move(self.target_positions, self.target_directions, noise=0)
        self.agent_positions = self.move(self.agent_positions, actions, noise=0.05)
        won = self.target_captured()
        if won and self.verbose: print('target captured at step', self.step_n)
        
        first_encounters = self.update_visited() if self.intermediate_reward else 0
        r = (float(won) + first_encounters / self.n_agent) / 2 if self.intermediate_reward else float(won)

        end = self.step_n >= 40
        if end and self.verbose: print('target not captured')

        return self.get_obs(), r, int(won or end)

    def update_visited(self):
        stepping_on = (np.array_equal(a, t) for a,t in zip(self.agent_positions, self.respective_target_positions))
        first_encounter = np.array([int(s and not v) for v, s in zip(self.visited, stepping_on)])

        self.visited += np.array(first_encounter)
        return np.sum(first_encounter)

    def action_space_sample(self):
        return np.random.randint(self.n_action)

    def action_space_batch_sample(self):
        return np.random.randint(self.n_action, size=self.n_agent)

    #####################################################################################
    # Helper methods

    def move(self, positions, directions, noise=0):
        translations = np.stack([self.translation(d, noise=noise) for d in directions])
        positions += translations
        return self.wrap_positions(positions)

    def target_captured(self):
        return all(np.array_equal(a, t) for a, t in zip(self.agent_positions, self.respective_target_positions))

    def rand_position(self):
        return np.array([randint(self.x_len), randint(self.y_len)])

    @staticmethod
    def translation(direction, noise=0.1):
        return TRANSLATION_TABLE[direction][np.random.choice(3, p=[noise/2, 1-noise, noise/2])]

    @staticmethod
    def flick(N, prob=0.3):
        mask = np.random.random(N.shape) > prob
        return N * mask

    @property
    def respective_target_positions(self):
        if self.multi_task:
            return self.target_positions
        else:
            return (self.target_positions[0] for _ in range(self.n_agent))

    def wrap_positions(self, positions):
        # fix translations which moved positions out of bound.
        X, Y = np.split(positions, 2, axis=1)
        return np.concatenate([X % self.x_len, Y % self.y_len], axis=1)

    def normalize_positions(self, positions):
        X, Y = np.split(positions, 2, axis=1)
        return np.concatenate([
                (X - self.half_x_len) / self.half_x_len,
                (Y - self.half_y_len) / self.half_y_len
            ],
            axis=1)