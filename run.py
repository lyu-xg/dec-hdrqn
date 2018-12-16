from numpy import random
import os
import subprocess

NUM_GPU = str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')


params = [
    # '-x 3 -y 3 --epsilon_hysteretic',
    # '-x 4 -y 4 --epsilon_hysteretic',
    # '-x 5 -y 5 --epsilon_hysteretic',
    # '-x 6 -y 6 --epsilon_hysteretic',
    # '-x 3 -y 3 -d 1 -e 0.2',
    # '-x 4 -y 4 -d 1 -e 0.2',
    # '-x 5 -y 5 -d 1 -e 0.2',
    # '-x 6 -y 6 -d 1 -e 0.2',


    # '-x 3 -y 3 --epsilon_hysteretic 1',
    # '-x 4 -y 4 --epsilon_hysteretic 1',
    # '-x 5 -y 5 --epsilon_hysteretic 1',
    # '-x 6 -y 6 --epsilon_hysteretic 1',

    # '-x 3 -y 3 --epsilon_hysteretic 1',
    # '-x 4 -y 4 --epsilon_hysteretic 1',
    # '-x 5 -y 5 --epsilon_hysteretic 1',
    # '-x 6 -y 6 --epsilon_hysteretic 1',

    # # quantiles


    # '-x 3 -y 3 -e .2 -q 16 ',
    # '-x 3 -y 3 -e .2 -q 32 ',

    # '-x 3 -y 3 -e .2 -q 128 ',
    # '-x 3 -y 3 -e .2 -q 256 ',

    # '-x 4 -y 4 --huber_delta 1.0',
    # '-x 4 -y 4 --huber_delta 1.5',
    # '-x 4 -y 4 --huber_delta 2.0',
    # '-x 4 -y 4 --huber_delta 3.0',

    # '-x 5 -y 5 --huber_delta 1.0',
    # '-x 5 -y 5 --huber_delta 1.5',
    # '-x 5 -y 5 --huber_delta 2.0',
    # '-x 5 -y 5 --huber_delta 3.0',

    # '-x 6 -y 6 --huber_delta 1.0',
    # '-x 6 -y 6 --huber_delta 1.5',
    # '-x 6 -y 6 --huber_delta 2.0',
    # '-x 6 -y 6 --huber_delta 3.0',

    # '-x 3 -y 3 -q 16 --quant_mean_loss 1',
    # '-x 3 -y 3 -q 16 ',
    # '-x 3 -y 3 -q 16 ',
    # '-x 3 -y 3 -q 16 ',

    # '-x 6 -y 6 -u 3000',
    # '-x 7 -y 7 -u 3000',
    # '-x 8 -y 8 -u 3000',

    # '-x 4 -y 4 -q 64 -u 1000 -e 0.25',
    # '-x 4 -y 4 -q 64 -u 2000 -e 0.25',
    # '-x 4 -y 4 -q 64 -u 4000 -e 0.25',
    # '-x 4 -y 4 -q 64 -u 3000 -e 0.25',
    # '-x 4 -y 4 -q 64 -u 5000 -e 0.25',
    # '-x 4 -y 4 -q 64 -u 1000 -e 0.25 -d 1',
    # '-x 4 -y 4 -q 64 -u 2000 -e 0.25 -d 1',
    # '-x 4 -y 4 -q 64 -u 4000 -e 0.25 -d 1',
    # '-x 4 -y 4 -q 64 -u 3000 -e 0.25 -d 1',
    # '-x 4 -y 4 -q 64 -u 5000 -e 0.25 -d 1',
    # '-x 7 -y 7 -q 64 -u 1000',
    # '-x 7 -y 7 -q 64 -u 2000',
    # '-x 7 -y 7 -q 64 -u 4000',
    # '-x 7 -y 7 -q 64 -u 3000',
    # '-x 7 -y 7 -q 64 -u 5000',
    # '-x 6 -y 6 -q 64 -u 1000',
    # '-x 6 -y 6 -q 64 -u 2000',
    # '-x 6 -y 6 -q 64 -u 4000',
    # '-x 6 -y 6 -q 64 -u 3000',
    # '-x 6 -y 6 -q 64 -u 5000',
    # '-x 4 -y 4 -u 2000 -e 0.25',
    # '-x 4 -y 4 -u 2000 -e 0.25',
    # '-x 4 -y 4 -r 0.0001',
    # '-x 4 -y 4 -r 0.005',
    # '-x 4 -y 4 -r 0.0005',
    # '-x 4 -y 4 -r 0.0001 -u 1000',

    # '-x 3 -y 3 -e .2 -q 84 --magic 1 --total_step 1000000 --end_hysteretic .4 --quantile_init_w 0.05',
    # '-x 3 -y 3 -e .2 -q 84 --magic 1 --total_step 1000000 --end_hysteretic .4 --quantile_init_w 0.1',
    # '-x 3 -y 3 -e .2 -q 84 --magic 1 --total_step 1000000 --end_hysteretic .4 --quantile_init_w 0.25',

    # '-x 3 -y 3 -e .2 -q 84 --magic 2 --total_step 1000000 --end_hysteretic .4 --quantile_init_w 1',
    # '-x 3 -y 3 -e .2 -q 84 --magic 2 --total_step 1000000 --end_hysteretic .4 --quantile_init_w 0.5',
    '-x 3 -y 3 -i 1 -e .2 -q 16 --magic 1 --total_step 1000000 --quantile_init_w 0.5 -i 1',
    '-x 4 -y 4 -i 1 -e .2 -q 16 --magic 1 --total_step 1000000 --quantile_init_w 0.5 -i 1',
    # '-x 4 -y 4 -e .2 -q 84 --magic 2 --total_step 1000000 --end_hysteretic .4 --quantile_init_w 0.5',
    
]


params = []
# for d in (3,):
#     params.append('-x {} -y {} --q {} --epsilon_hysteretic 1'.format(d, d, 32))
for run in range(10,20):
    params.append('--env_name capture_target -x 4 -y 4 -q 16 -e .3 --end_hysteretic .5  -i 1 --magic 1 --run_id {}'.format(run))

limit_cpu = False
for i, p in enumerate(params):
    cmd = 'CUDA_VISIBLE_DEVICES={} {} python main.py -p {} {} &'.format(
        i % NUM_GPU, 'taskset 0x3FF' if limit_cpu else '', int(random.random()*10), p)
    print(cmd)
    os.system(cmd)


#################################################

# currently running magic-IQN on 4x4
# need to run HDRQN on 4x4
# need to run muggle-IQN on 4x4 