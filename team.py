import os
import tensorflow as tf
import numpy as np
from qnet import Qnetwork
from itertools import chain

K = 1000

HYSTERESIS_STABLE_AT = 200 * K
MIN_EPSILON = 0.1
EPSILON_ANNEAL_END = 200 * K

class Team:
    '''
    Team consists of n Qnetworks which represent n agents.
    Agents interact with environment and learn as a Team.
    '''
    def __init__(self, env, memory, n_agent, identity, hysteretic, dynamic_h, agent_args={}):
        self.env, self.memory = env, memory
        self.identity = identity
        self.last_obs = self.env.reset()

        self.n_agent, (self.init_hysteretic, self.end_hysteretic) = n_agent, hysteretic
        self.dynamic_hysteretic = dynamic_h
        self.hysteretic_follow_epsilon = not hysteretic

        self.create_sess()

        self.agents = [
            Qnetwork(self.sess, str(i), env.obs_size, env.n_action, **agent_args)
            for i in range(n_agent)
        ]

        self.sess.run(tf.global_variables_initializer())
        self.step_count, self.game_count = 0, 0
        self.zero_state =  [a.ZERO_STATE for a in self.agents]
        self.hidden_states = self.zero_state
        self.construct_summary_ops()

        self.eval_results = []
        self.epsilon = 1.0

    def create_sess(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def do_target_update(self):
        self.sess.run([a.target_update_ops for a in self.agents])

    def step(self):
        self.step_count += 1
        a, self.hidden_states = self.get_actions_and_states(self.last_obs, self.hidden_states)
        obs, r, t = self.env.step(a)
        self.memory.append((self.last_obs, a, r, obs, t))
        self.last_obs = obs
        if t:
            self.game_count += 1
            self.last_obs = self.env.reset()
            self.online_summary(*self.memory.flush_scenario_cache())
            self.hidden_states = self.zero_state
            # if not self.game_count % 10:
            #     self.evaluate()
        
    def get_actions_and_states(self, obss, hid_state):
        res = self.run_ops_with_feeds(
            a.get_action_ops(o, s, self.epsilon)
            for a, o, s in zip(self.agents, obss, hid_state)
        )

        actions, hidden_states = zip(*res)
        actions = [a if np.random.random() > self.epsilon else self.env.action_space_sample() for a in actions]

        return actions, hidden_states

    def update_hysteretic(self):
        if self.hysteretic_follow_epsilon:
            self.hysteretic = 1 - self.epsilon
        elif self.dynamic_hysteretic:
            self.hysteretic = min(
                self.end_hysteretic,
                ((self.end_hysteretic - self.init_hysteretic) / HYSTERESIS_STABLE_AT) * self.step_count + self.init_hysteretic
            )

    def train(self):
        self.update_hysteretic()
        if self.memory.total_scenario_count < self.memory.batch_size: return
        batch = self.inverse_joint(self.memory.sample())

        train_result = self.run_ops_with_feeds(
            (a.get_training_ops(*zip(*trace), self.epsilon)
             for a, trace in zip(self.agents, batch)),
            adtl_ops=[self.summaries],
            adtl_feeds={a.hysteretic: self.hysteretic for a in self.agents} # TODO merge this into agent.get_training_ops
        )
        summary = train_result[0]
        self.summary_writer.add_summary(train_result[0], self.step_count)
        self.epsilon = max(self.epsilon*0.9999, MIN_EPSILON)


    def construct_summary_ops(self):
        self.summaries = tf.summary.merge_all()
        if os.path.isdir('./log/' + self.identity):
            os.system('rm -rf ./log/' + self.identity)
        self.summary_writer = tf.summary.FileWriter('./log/' + self.identity, self.sess.graph)
        self.online_summary_ph = tf.placeholder(tf.float32, shape=(2,), name='online')
        self.online_summary_ops = (tf.summary.scalar('online_performance', self.online_summary_ph[0]),
                                   tf.summary.scalar('online_scenario_length', self.online_summary_ph[1]))
        self.eval_summary_ph = tf.placeholder(tf.float32, shape=(2,), name='eval')
        self.eval_summary_ops = (tf.summary.scalar('eval_performance', self.eval_summary_ph[0]),
                                 tf.summary.scalar('eval_scenario_length', self.eval_summary_ph[1]))

    def online_summary(self, R, L):
        return
        # self.online_results.append([R,L])
        # print('[episode{}] R: {}, L: {}'.format(self.game_count, R, L), flush=True)
        # online_perf, online_episode_count = self.sess.run(
        #     self.online_summary_ops,
        #     feed_dict={self.online_summary_ph: np.array([R, L])})
        # self.summary_writer.add_summary(online_perf, self.step_count)
        # self.summary_writer.add_summary(online_episode_count, self.step_count)

    def eval_summary(self, R, L):
        print('eval return', R, flush=True)
        self.eval_results.append([R,L])
        eval_perf, eval_episode_count = self.sess.run(
            self.eval_summary_ops,
            feed_dict={self.eval_summary_ph: np.array([R, L])})
        self.summary_writer.add_summary(eval_perf, self.step_count)
        self.summary_writer.add_summary(eval_episode_count, self.step_count)
        

    def evaluate(self, n_episode=10):
        R, L = 0, 0
        
        for _ in range(n_episode):
            last_obs, t = self.env.reset(), 0
            hidden_states = [a.ZERO_STATE for a in self.agents]
            while not t:
                a, hidden_states = self.get_actions_and_states(last_obs, hidden_states)
                last_obs, r, t = self.env.step(a)
                R += r
                L += 1
        self.eval_summary(R / n_episode, L / n_episode)


    ######################################################################################
    # Helper methods
    def run_ops_with_feeds(self, ops_and_feeds, adtl_ops=[], adtl_feeds={}):
        O, F = [], {}
        O.extend(adtl_ops)
        F.update(adtl_feeds)
        for ops, feeds in ops_and_feeds:
            O.extend(ops)
            F.update(feeds)
        return self.sess.run(O, F)

    def reset_hidden_states(self):
        self.hidden_states = [a.ZERO_STATE for a in self.agents]

    def inverse_joint(self, joint_transitions):
        # generate transitions for individual agents
        trans = [[] for _ in range(self.n_agent)]
        # for trace in joint_transitions:
        for o, a, r, o_, t in chain(*joint_transitions):
            for i in range(self.n_agent):
                # the last entry is a "valid bit", indicates whether the transition is a padding or not
                trans[i].append([o[i], a[i], r, o_[i], t, float(o[i] is not self.memory.ZERO_JOINT_OBS)])
        return trans

    # @property
    # def epsilon(self):
    #     return (-0.9/EPSILON_ANNEAL_END) * self.step_count + 1 if self.step_count < EPSILON_ANNEAL_END else MIN_EPSILON

    # def epsilon_at(self, i, anneal_bottom=200000):
    #     return (-0.9/anneal_bottom) * i + 1 if i < anneal_bottom else .1


    def checkpoint(*args):
        pass

    def load_checkpoint():
        pass
