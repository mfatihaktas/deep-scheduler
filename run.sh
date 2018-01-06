#!/bin/bash
echo $1 $2 $3

PYTHON=python3
PKILL=/usr/bin/pkill

rm -r __pycache__
if [ $1 = 't' ]; then
  $PYTHON tutorial.py
else
  echo "Argument did not match!"
fi