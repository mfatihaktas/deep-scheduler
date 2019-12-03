#!/bin/bash
#SBATCH --partition=main             # Partition (job queue)
#SBATCH --job-name=redsmall_plots_wDolly
#SBATCH --nodes=1              # Number of nodes you require
#SBATCH --ntasks=1             # Total # of tasks across all nodes
#SBATCH --cpus-per-task=1            # Cores per task (>1 if multithread tasks)
#SBATCH --mem=8000                   # Real memory (RAM) required (MB)
#SBATCH --time=12:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --export=ALL                 # Export your current env to the job env
# #SBATCH --output=loglearning/redsmall_plots_wDolly.ro.slen3.node%N.jid%j.out
# #SBATCH --output=loglearning/redsmall_plots_wDolly.ro.slen2.out
#SBATCH --output=log/redsmall_plots_wDolly.out

export MV2_ENABLE_AFFINITY=0
srun --mpi=pmi2 python3 /home/mfa51/deep-scheduler/redsmall_plots_wDolly.py --ro 
  
