#!/bin/bash

nj=$1

for s in `seq 0 ${nj}`; do
  qsub -V -cwd ./run.sh ${s} egs/korean2/partition_seed${s}
done
