#!/bin/bash

use_components=false

. ./scripts/parse_options.sh

if [ $# -ne 4 ]; then
  echo "Usage: ./qsub_submit --use-components <true/false> <num-seeds> <odir> <config> <tsv>"
  exit 1;
fi

nj=$1
odir=$2
config=$3
tsv=$4


mkdir -p ${odir}

for s in `seq 0 ${nj}`; do
  mkdir -p ${odir}/seed${s}
  qsub -l h_rt=20:00:00 -V -o ${odir}/seed${s}/split.log -e ${odir}/seed${s}/split.err -N split.${s} -cwd ./run.sh --use-components ${use_components} ${s} ${odir}/seed${s} ${config} ${tsv}
done

