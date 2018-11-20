#!/bin/bash
echo $1 $2 $3

PY=python3

if [ $1 = 'i' ]; then
  source ~/tensorflow/bin/activate
elif [ $1 = 's' ]; then
  $PY sim_exp.py
  # $PY scheduler.py
elif [ $1 = 'e' ]; then
  $PY experience_replay.py
  # nohup $PY experience_replay.py > experience_replay.out 2>&1 &
elif [ $1 = 'r' ]; then
  # $PY rlearning.py
  $PY rvs.py
elif [ $1 = 'p' ]; then
  $PY plot_scher.py
elif [ $1 = 'm' ]; then
  $PY modeling.py
  # $PY mgn_model.py
else
  echo "Arg did not match!"
fi