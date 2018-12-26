from numpy import random

class Env:
    def action_space_sample(self):
        return random.randint(self.n_action)