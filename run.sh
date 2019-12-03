#!/bin/bash
echo $1 $2 $3

PY=python3

if [ $1 = 'i' ]; then
  source ~/tensorflow/bin/activate
elif [ $1 = 'me' ]; then
  FILE='redsmall_plots_wDolly' # 'experience_replay'
  NTASKS=1
  echo "#!/bin/bash
#SBATCH --partition=main             # Partition (job queue)
#SBATCH --job-name=$FILE
#SBATCH --nodes=$NTASKS              # Number of nodes you require
#SBATCH --ntasks=$NTASKS             # Total # of tasks across all nodes
#SBATCH --cpus-per-task=1            # Cores per task (>1 if multithread tasks)
#SBATCH --mem=8000                   # Real memory (RAM) required (MB)
#SBATCH --time=12:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --export=ALL                 # Export your current env to the job env
# #SBATCH --output=loglearning/$FILE.ro$2.slen3.node%N.jid%j.out
# #SBATCH --output=loglearning/$FILE.ro$2.slen2.out
#SBATCH --output=log/$FILE.out

export MV2_ENABLE_AFFINITY=0
srun --mpi=pmi2 python3 $PWD/$FILE.py --ro $2
  " > jscript.sh
  
  sbatch jscript.sh
# elif [ $1 = 'den' ]; then
#   echo "den $2"
elif [ $1 = 'ame' ]; then
  # rm loglearning/* save_expreplay/*
  for ro in $(seq 0.1 0.1 0.9)
  do
    echo "Launching MPI experience_replay for ro=$ro"
    ./run.sh me $ro
    sleep 1
  done
elif [ $1 = 's' ]; then
  # $PY sim_exp.py
  $PY sim_wrelaunch_exp.py
elif [ $1 = 'e' ]; then
  # rm save_expreplay/*
  # $PY experience_replay.py --ro=0.1
  # nohup $PY experience_replay.py > experience_replay.out 2>&1 &
  $PY eval_wmpi.py
elif [ $1 = 'd' ]; then
  $PY drl_plots.py
elif [ $1 = 'r' ]; then
  # $PY rlearning.py
  $PY redsmall_plots.py
  # $PY relaunch_plot.py
elif [ $1 = 'rd' ]; then
  $PY redsmall_plots_wDolly.py
elif [ $1 = 'm' ]; then
  # $PY modeling.py
  # $PY model_checking.py
  $PY modeling_wDolly.py
elif [ $1 = 'n' ]; then
  $PY numeric_comparison.py
else
  echo "Arg did not match!"
fi
