from numpy import random
import os
import subprocess
import argparse

NUM_GPU = str(subprocess.check_output(['nvidia-smi', '-L'])).count('UUID')

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cpu_limit', action='store', type=int, default=0)
parser.add_argument('-d', '--dry_run', action='store', type=bool, default=False)
parser.add_argument('-t', '--task_id', action='store', type=int, default=-1)
parser.add_argument('-n', '--n_runs', action='store', type=int, default=2)


args = parser.parse_args()

# params = [
#     '--env_name capture_target --gridx 3 --gridy 3 --n_quant 0 --init_hysteretic 0.5 --end_hysteretic 0.55 ' + \
#     '--implicit 0 --likely 0 --distort_type identity --distort_param 0.0  --buffer_size 4000 --run_id {}{}'.format(
#         '' if args.task_id < 0 else '{}-'.format(args.task_id),
#         run)
#     for run in range(args.n_runs)
# ]

params = [
    '--env_name capture_target --gridx 5 --gridy 5 --n_quant 0 --result_dir 66 ' + \
    '--implicit 0 --likely 0 --distort_type identity --distort_param 0.0  --buffer_size 4000 --run_id {}{}'.format(
        '' if args.task_id < 0 else '{}-'.format(args.task_id), run)
    for run in range(args.n_runs)
    # for hys in (0.2, 0.4, 0.6, 0.8)
]

params += [
    '--env_name capture_target --gridx 5 --gridy 5 --n_quant 16 --result_dir 66 ' + \
    '--implicit 1 --likely 1 --distort_type identity --distort_param 0.0  --buffer_size 4000 --run_id {}{}'.format(
        '' if args.task_id < 0 else '{}-'.format(args.task_id), run)
    for run in range(args.n_runs)
    # for hys in (0.2, 0.4, 0.6, 0.8)
]

for i, param in enumerate(params):
    cmd = 'CUDA_VISIBLE_DEVICES={} {} python main.py --s_sleep {} {} &'.format(
        i % NUM_GPU,
        'taskset {}'.format(hex(int('1' * args.cpu_limit, 2))) if args.cpu_limit else '',
        int(random.random()*10),
        param
    )
    print(cmd, flush=True)
    if not args.dry_run:
        os.system(cmd)
