#!/bin/bash
echo $1 $2 $3

if [ $1 = 'i' ]; then
  # srun --partition=main --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2000 --time=3:00:00 --export=ALL --pty bash -i
  srun --partition=main --nodes=1 --ntasks=1 --cpus-per-task=10 --mem=8000 --time=2:00:00 --export=ALL --pty bash -i
else
  echo "Argument did not match!"
fi
