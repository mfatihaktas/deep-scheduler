#!/bin/bash
echo $1 $2 $3

if [ $1 = 'i' ]; then
  source ~/tensorflow/bin/activate
elif [ $1 = 'r' ]; then
  rm -r __pycache__
  rm log/*
  sbatch slurm_jscript.sh
elif [ $1 = 'l' ]; then
  squeue | grep mfa
elif [ $1 = 'k' ]; then
  scancel
else
  echo "Arg did not match!"
fi