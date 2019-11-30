#!/bin/bash
echo $1 $2 $3

PWD=/home/mfa51/deep-scheduler

if [ $1 = 'i' ]; then
  module load intel/16.0.3 mvapich2/2.1
  # source ~/tensorflow/bin/activate
elif [ $1 = 'r' ]; then
  FILE='eval_wmpi' # 'learn_wmpi'
  NTASKS=21
  echo "#!/bin/bash
#SBATCH --partition=main             # Partition (job queue)
#SBATCH --job-name=$FILE
#SBATCH --nodes=$NTASKS              # Number of nodes you require
#SBATCH --ntasks=$NTASKS             # Total # of tasks across all nodes
#SBATCH --cpus-per-task=1            # Cores per task (>1 if multithread tasks)
#SBATCH --mem=6000                   # Real memory (RAM) required (MB)
#SBATCH --time=62:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --export=ALL                 # Export your current env to the job env
#SBATCH --output=log/$FILE.%N.%j.out

export MV2_ENABLE_AFFINITY=0
srun --mpi=pmi2 python3 $PWD/$FILE.py
  " > jscript.sh
  
  rm log/*
  sbatch jscript.sh
elif [ $1 = 'l' ]; then
  squeue -u mfa51
elif [ $1 = 'k' ]; then
  scancel --user=mfa51 # -n learning 
else
  echo "Arg did not match!"
fi
