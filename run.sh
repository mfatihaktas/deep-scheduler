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
  $PYTHON shortestq_sching.py
elif [ $1 = 'r' ]; then
  $PYTHON rep_sching.py
elif [ $1 = 'e' ]; then
  $PYTHON exp.py
else
  echo "Argument did not match!"
fi