#!/bin/bash
#SBATCH --job-name=dqn
#SBATCH --output=job.out
#SBATCH --exclusive
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1g
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --array=6-19

cd /scratch/lu.xue/dec-hdrqn

srun python main.py --gridx 3 --gridy 3 --n_quant 16  --implicit 1 --likely 1 --distort_type wang --distort_param 0.0  --run_id $SLURM_ARRAY_TASK_ID
# srun python main.py --gridx 5 --gridy 5 --n_quant 16  --implicit 1 --likely 1  --run_id $SLURM_ARRAY_TASK_ID --result_dir 66
# srun python main.py --gridx 4 --gridy 4 --n_quant 0  --implicit 0 --likely 0  --run_id $SLURM_ARRAY_TASK_ID --result_dir 55
# srun python main.py --gridx 5 --gridy 5 --n_quant 0  --implicit 0 --likely 0  --run_id $SLURM_ARRAY_TASK_ID --result_dir 66