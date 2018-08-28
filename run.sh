#!/bin/bash
echo $1 $2 $3

PY=python3

if [ $1 = 'i' ]; then
  source ~/tensorflow/bin/activate
elif [ $1 = 's' ]; then
  $PY sim_exp.py
else
  echo "Arg did not match!"
fi