from numpy import random
import os
import subprocess
import argparse

NUM_GPU = str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cpu_limit', action='store', type=int, default=0)
parser.add_argument('-d', '--dry_run', action='store', type=bool, default=False)

args = parser.parse_args()

params = []
for run in range(4):
    params.append('--env_name capture_target -q 16 -e 0.2 --end_hysteretic 0.4  -i 1 --likely 1 --distort_type wang --distort_param 0.0 -o 0.0  --run_id {}'.format(run))

for i, p in enumerate(params):
    cmd = 'CUDA_VISIBLE_DEVICES={} {} python main.py -p {} {} &'.format(
        i % NUM_GPU,
        'taskset {}'.format(hex(int('1' * args.cpu_limit, 2))) if args.cpu_limit else '',
        int(random.random()*10),
        p
    )
    print(cmd, flush=True)
    if not args.dry_run:
        os.system(cmd)


#################################################100
