#!/bin/bash
#SBATCH --partition=main             # Partition (job queue)
#SBATCH --job-name=learn_howtorep
#SBATCH --nodes=31              # Number of nodes you require
#SBATCH --ntasks=31             # Total # of tasks across all nodes
#SBATCH --cpus-per-task=1            # Cores per task (>1 if multithread tasks)
#SBATCH --mem=24000                  # Real memory (RAM) required (MB)
#SBATCH --time=20:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --export=ALL                 # Export your current env to the job env
#SBATCH --output=log/slurm.%N.%j.out

export MV2_ENABLE_AFFINITY=0
srun --mpi=pmi2 python3 /home/mfa51/deep-scheduler/learn_howtorep_wmpi.py
  
