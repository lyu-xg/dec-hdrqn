import argparse
import tensorflow as tf
import numpy as np
from termcolor import colored
from time import time as current_time
from time import sleep
from envs.capture_target import CaptureTarget
from envs.cmotp import CMOTP
from team import Team
from memory import ExperienceTrajectories
from params import CNN_Feature_Extractors
cnn_params = CNN_Feature_Extractors()

ENVIRONMENTS = {
    'capture_target': CaptureTarget,
    'cmotp1': lambda: CMOTP(1),
    'cmotp2': lambda: CMOTP(2),
    'cmotp3': lambda: CMOTP(3),
}

def train(tracelen, h_size, init_hysteretic, end_hysteretic, gridx, gridy, n_quant, magic, discount,
          n_target, n_agent, verbose, learning_rate, target_update_freq, intermediate_reward,
          huber_delta, dynamic_h, s_sleep, quant_mean_loss, epsilon_hysteretic,
          quantile_init_w, implicit_quant, env_name, run_id, batch_size=32):

    # Adding sleeping option to be used by batch runner
    # Sleeping random seconds helps to avoid massive collisions between
    # CPU intensive tasks (e.g. env interaction) and GPU intensive tasks (gradient update)
    # when running multiple processes simultaneously 
    print('sleeping for', s_sleep)
    sleep(s_sleep)

    # unique-identifier-string which intend to be readable somehow
    identity = 'n_agent={},env={},grid={}x{},n_quant={},hysteretic={}{}-{},magic={},init_bias={},run={}'.format(
        n_agent, env_name, gridx, gridy, n_quant, 
        'follow_epsilon' if epsilon_hysteretic else ('dynamic' if dynamic_h else ''), 
        init_hysteretic, end_hysteretic, magic, quantile_init_w, run_id)
    if target_update_freq != 5000:
        identity += ',target_update={}'.format(target_update_freq)
    if learning_rate != 0.001:
        identity += ',lr={}'.format(learning_rate)
    if quant_mean_loss:
        identity += ',quant_mean_loss=1'
    if huber_delta != 1.0:
        identity += ',huber_delta={}'.format(huber_delta)
    if intermediate_reward:
        identity += ',intermediate_reward=1'
    if implicit_quant:
        identity += ',implicit=1'

    print('\n', colored(identity, 'blue'), '\n')

    if epsilon_hysteretic:
        init_hysteretic = 0

    Env = ENVIRONMENTS[env_name]
    conv = env_name.startswith('cmotp')

    if conv:
        env = Env()
        assert n_target == 1 and n_agent == 2
        h_size = cnn_params.fc
    else:
        env = Env(n_target, n_agent, (gridx, gridy), verbose=verbose, intermediate_r=intermediate_reward)
    mem = ExperienceTrajectories(n_agent, env.obs_size, tracelen, batch_size)
    team = Team(env, mem, n_agent, n_quant, identity, (init_hysteretic, end_hysteretic), dynamic_h, agent_args={
        'train_batch_size': batch_size,
        'train_tracelen': tracelen,
        'h_size': h_size,
        'learning_rate': learning_rate,
        'huber_delta': huber_delta,
        'quant_mean_loss': quant_mean_loss,
        'discount': discount,
        'magic': magic,
        'quantile_init_w': quantile_init_w,
        'implicit_quant': implicit_quant,
        'conv': conv
    })

    t = current_time()
    for i in range(10000*10000):
        
        team.step()
        if team.game_count > 5000:
            break

        # if not i % (1 if conv else 5):
        team.train()

        if not i % target_update_freq:
            # team.evaluate()
            team.do_target_update()
            print('[{:.1f}K]took {:.1f} seconds to do {:.1f}K steps'.format(i/1000, current_time()-t, target_update_freq/1000))
            t = current_time()
    
    np.save(open('results/{}.npy'.format(identity), 'wb'), np.array(team.eval_results))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--tracelen', action='store', type=int, default=4)
    parser.add_argument('-s', '--h_size', action='store', type=int, default=64)
    parser.add_argument('-p', '--s_sleep', action='store', type=float, default=0.0)
    parser.add_argument('-e', '--init_hysteretic', action='store', type=float, default=.2)
    parser.add_argument('--end_hysteretic', action='store', type=float, default=.4)
    parser.add_argument('--quantile_init_w', action='store', type=float, default=.5)
    parser.add_argument('--epsilon_hysteretic', action='store', type=int, default=0)
    # parser.add_argument('--total_step', action='store', type=int, default=5000 * 1000)
    parser.add_argument('--env_name', action='store', type=str, default='cmotp1')
    parser.add_argument('-i', '--implicit_quant', action='store', type=int, default=0)
    parser.add_argument('--intermediate_reward', action='store', type=int, default=0)
    parser.add_argument('-d', '--dynamic_h', action='store', type=int, default=1)
    parser.add_argument('-x', '--gridx', action='store', type=int, default=3)
    parser.add_argument('-y', '--gridy', action='store', type=int, default=3)
    parser.add_argument('-q', '--n_quant', action='store', type=int, default=0)
    parser.add_argument('--quant_mean_loss', action='store', type=int, default=0)
    parser.add_argument('-m', '--magic', action='store', type=int, default=0)
    parser.add_argument('-a', '--n_agent', action='store', type=int, default=2)
    parser.add_argument('-t', '--n_target', action='store', type=int, default=1)
    parser.add_argument('-v', '--verbose', action='store', type=int, default=0)
    parser.add_argument('--discount', action='store', type=float, default=0.99)
    parser.add_argument('-r', '--learning_rate', action='store', type=float, default=0.001)
    parser.add_argument('-b', '--huber_delta', action='store', type=float, default=1.0)
    parser.add_argument('-u', '--target_update_freq', action='store', type=int, default=5000)
    parser.add_argument('--run_id', action='store', type=int, default=0)

    train(**vars(parser.parse_args()))
    
    
if __name__ == '__main__':
    main()