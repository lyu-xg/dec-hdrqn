import numpy as np
from collections.abc import Iterable

class RingBuf:
    '''
    old-school Ring Buffer
    which, oddly, does not come with standard python library
    '''
    def __init__(self, size=1000000):
        # Pro-tip: when implementing a ring buffer, always allocate one extra element,
        # this way, self.start == self.end always means the buffer is EMPTY, whereas
        # if you allocate exactly the right number of elements, it could also mean
        # the buffer is full. This greatly simplifies the rest of the code.
        self.data = [None] * (size + 1)
        self.start = 0
        self.end = 0

    def append(self, element):
        self.data[self.end] = element
        self.end = (self.end + 1) % len(self.data)
        # end == start and yet we just added one element. This means the buffer has one
        # too many element. Remove the first element by incrementing start.
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)

    def __getitem__(self, idx):
        return self.data[(self.start + idx) % len(self.data)]

    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def sample_batch(self, size):
        return [self[i] for i in np.random.choice(len(self), size)]


class ExperienceTrajectories:
    def __init__(self, n_agent, obs_size, trace_len, batch_size, size=16000):
        self.trace_len, self.batch_size = trace_len, batch_size

        obs_shape = (n_agent, *obs_size) if is_iter(obs_size) else (n_agent, obs_size)
        self.ZERO_JOINT_OBS = np.zeros(obs_shape)
        self.ZERO_JOINT_ACT = [0] * n_agent

        # transition: (joint_o, joint_a, r, joint_o', t)
        self.ZERO_PADDING = [[self.ZERO_JOINT_OBS, self.ZERO_JOINT_ACT, 0.0, self.ZERO_JOINT_OBS, 1]]

        self.buf = RingBuf(size=size)
        self.scenario_cache_reset()
        self.total_scenario_count = 0

    def append(self, transition):
        # transition: (joint_o, joint_a, r, joint_o', t)
        self.total_reward += transition[2]
        self.scenario_cache.append(transition)

    def flush_scenario_cache(self):
        self.total_scenario_count += 1
        for i in range(len(self.scenario_cache)):
            trace = self.scenario_cache[i:i+self.trace_len]
            # end-of-episode padding
            trace = self.ZERO_PADDING * (self.trace_len - len(trace)) + trace
            self.buf.append(trace)

        R, L = self.total_reward, len(self.scenario_cache) - self.trace_len + 1
        self.scenario_cache_reset()
        return R, L

    def scenario_cache_reset(self):
        self.total_reward = 0
        # beginning-of-episode padding
        self.scenario_cache = self.ZERO_PADDING * (self.trace_len - 1)

    def sample(self):
        return self.buf.sample_batch(self.batch_size)

def is_iter(obj):
    return isinstance(obj, Iterable)