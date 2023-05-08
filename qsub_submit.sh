#!/bin/bash

nj=$1
odir=$2
config=$3
tsv=$4

mkdir -p ${odir}

for s in `seq 0 ${nj}`; do
  mkdir -p ${odir}/seed${s}
  qsub -l h_rt=20:00:00 -V -o ${odir}/seed${s}/split.log -e ${odir}/seed/split.err -N split.${s} -cwd ./run.sh ${s} ${odir}/seed${s} ${config} ${tsv}
done

