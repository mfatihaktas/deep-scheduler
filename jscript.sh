#!/bin/bash
#SBATCH --partition=main             # Partition (job queue)
#SBATCH --job-name=eval_wmpi
#SBATCH --nodes=31              # Number of nodes you require
#SBATCH --ntasks=31             # Total # of tasks across all nodes
#SBATCH --cpus-per-task=1            # Cores per task (>1 if multithread tasks)
#SBATCH --mem=6000                   # Real memory (RAM) required (MB)
#SBATCH --time=48:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --export=ALL                 # Export your current env to the job env
#SBATCH --output=log/eval_wmpi.%N.%j.out

export MV2_ENABLE_AFFINITY=0
srun --mpi=pmi2 python3 /home/mfa51/deep-scheduler/eval_wmpi.py
  
