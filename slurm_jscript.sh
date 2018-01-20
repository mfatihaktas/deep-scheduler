#!/bin/bash

#SBATCH --partition=main             # Partition (job queue)
#SBATCH --requeue                    # Return job to the queue if preempted
#SBATCH --job-name=learning_shortestq
#SBATCH --nodes=3                    # Number of nodes you require
#SBATCH --ntasks=3                   # Total # of tasks across all nodes
#SBATCH --cpus-per-task=1            # Cores per task (>1 if multithread tasks)
#SBATCH --mem=2000                   # Real memory (RAM) required (MB)
#SBATCH --time=00:05:00              # Total run time limit (HH:MM:SS)
#SBATCH --output=log/slurm.%N.%j.out
#SBATCH --error=log/slurm.%N.%j.err
#SBATCH --export=ALL                 # Export you current env to the job env

export MV2_ENABLE_AFFINITY=0

srun --mpi=pmi2 -n 3 python3 /home/mfa51/deep-scheduler/learning_shortestq_wmpi.py