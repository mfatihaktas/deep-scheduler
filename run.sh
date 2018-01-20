#!/bin/bash
echo $1 $2 $3

PYTHON=python3
PKILL=/usr/bin/pkill

rm -r __pycache__
if [ $1 = 'i' ]; then
  source ~/tensorflow/bin/activate
elif [ $1 = 't' ]; then
  rm log/*
  $PYTHON tutorial.py
elif [ $1 = 'tb' ]; then
  tensorboard --logdir=/home/ubuntu/deep-scheduler/log
elif [ $1 = 's' ]; then
  # $PYTHON scheduling.py
  $PYTHON learning_shortestq.py
elif [ $1 = 'r' ]; then
  $PYTHON learning_howtorep.py
elif [ $1 = 'e' ]; then
  $PYTHON exp.py
else
  echo "Arg did not match!"
fi