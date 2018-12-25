#!/bin/bash
echo $1 $2 $3

PY=python3

if [ $1 = 'i' ]; then
  source ~/tensorflow/bin/activate
elif [ $1 = 'mm' ]; then
  FILE='model_checking_wmpi'
  NTASKS=20
  echo "#!/bin/bash
#SBATCH --partition=main             # Partition (job queue)
#SBATCH --job-name=$FILE
#SBATCH --nodes=$NTASKS              # Number of nodes you require
#SBATCH --ntasks=$NTASKS             # Total # of tasks across all nodes
#SBATCH --cpus-per-task=1            # Cores per task (>1 if multithread tasks)
#SBATCH --mem=4000                   # Real memory (RAM) required (MB)
#SBATCH --time=24:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --export=ALL                 # Export your current env to the job env
#SBATCH --output=logmodel/$FILE.%N.%j.out

export MV2_ENABLE_AFFINITY=0
srun --mpi=pmi2 python3 $PWD/$FILE.py
  " > jscript.sh
    
    rm logmodel/*
    sbatch jscript.sh
elif [ $1 = 's' ]; then
  $PY sim_exp.py
  # $PY scheduler.py
elif [ $1 = 'e' ]; then
  rm save_expreplay/*
  $PY experience_replay.py
  # nohup $PY experience_replay.py > experience_replay.out 2>&1 &
elif [ $1 = 'r' ]; then
  # $PY rlearning.py
  $PY rvs.py
elif [ $1 = 'p' ]; then
  # $PY plot_scher.py
  $PY paper_plotting.py
elif [ $1 = 'm' ]; then
  # $PY modeling.py
  $PY model_checking.py
else
  echo "Arg did not match!"
fi