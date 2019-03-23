import argparse
import tensorflow as tf
import numpy as np
from termcolor import colored
from time import time as current_time
from time import sleep
from envs.capture_target import CaptureTarget
from envs.cmotp import CMOTP
from envs.climb_game import ClimbGame
from envs.env_CatchPigs import RandomPigEnv
from team import Team
from memory import ExperienceTrajectories
from params import CNN_Feature_Extractors
cnn_params = CNN_Feature_Extractors()

ENVIRONMENTS = {
    'capture_target': CaptureTarget,
    'cmotp1': lambda: CMOTP(1),
    'cmotp2': lambda: CMOTP(2),
    'cmotp3': lambda: CMOTP(3),
    'climb_game': ClimbGame,
    'pig': RandomPigEnv,
}

DEFAULT_PARAMETERS = {
    'tracelen': 4,
    'h_size': 64,
    's_sleep': 0,
    'init_hysteretic': .2,
    'end_hysteretic': .4,
    'quantile_init_w': .5,
    'epsilon_hysteretic': 0,
    'total_step': 500 * 1000,
    'env_name': 'capture_target',
    'implicit_quant': 0,
    'intermediate_reward': 0,
    'dynamic_h': 1,
    'gridx': 3,
    'gridy': 3,
    'n_quant': 0,
    'quant_mean_loss': 0,
    'likely': 0,
    'n_agent': 2,
    'n_target': 1,
    'verbose': 0,
    'discount': 0.99,
    'learning_rate': 0.001,
    'huber_delta': 1.0,
    'target_update_freq': 5000,
    'run_id': 99999,
    'distort_type': 'identity',
    'distort_param': 0.0,
    'batch_size': 32,
    'buffer_size': 10000,
    'result_dir': 'results',
}

def train(params):
    P = vars(params)
    # Adding sleeping option to be used by batch runner
    # Sleeping random seconds helps to avoid massive collisions between
    # CPU intensive tasks (e.g. env interaction) and GPU intensive tasks (gradient update)
    # when running multiple processes simultaneously 
    print('sleeping for', params.s_sleep, flush=True)
    sleep(params.s_sleep)
    print('run {} woke up:'.format(params.run_id), flush=True)

    # unique-identifier-string which intend to be readable somehow
    P['s_sleep'] = 0
    identity = ','.join(['{}={}'.format(p,v) for p,v in P.items() if v != DEFAULT_PARAMETERS[p]] or ['default'])
    print('\n', colored(identity, 'blue'), '\n', flush=True)

    Env = ENVIRONMENTS[params.env_name]

    # some environments have graphical observations, hence we use CNN as the front layers
    if params.env_name.startswith('cmotp') or  params.env_name.startswith('pig'):
        env = Env()
        assert params.n_target == 1 and params.n_agent == 2
        P['h_size'] = cnn_params.fc
        P['conv'] = True
    else:
        P['conv'] = False
        env = Env(params.n_target, params.n_agent, (params.gridx, params.gridy),
                  verbose=params.verbose, intermediate_r=params.intermediate_reward)
    
    mem = ExperienceTrajectories(params.n_agent, env.obs_size, params.tracelen, 
                                 params.batch_size, size=params.buffer_size)
    team = Team(env, mem, params.n_agent, identity, 
                (0 if params.epsilon_hysteretic else params.init_hysteretic, params.end_hysteretic),
                params.dynamic_h, agent_args=P)

    t = current_time()
    for i in range(params.total_step):
        # Main Loop
        team.step()

        if not i % 5:
            team.train()

        if not i % params.target_update_freq:
            team.evaluate()
            team.do_target_update()
            print('[{:.1f}K]took {:.1f} seconds to do {:.1f}K steps (eps={})'.format(i/1000, current_time()-t, params.target_update_freq/1000, team.epsilon), flush=True)
            t = current_time()
    
    np.save(open('{}/{}.npy'.format(params.result_dir, identity), 'wb'), np.array(team.eval_results))

def main():
    parser = argparse.ArgumentParser()
    for p, v in DEFAULT_PARAMETERS.items():
        parser.add_argument('--{}'.format(p), action='store', type=type(v), default=v)

    train(parser.parse_args())
    
    
if __name__ == '__main__':
    main()