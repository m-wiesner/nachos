#!/bin/bash

use_components=false
partition=
split=

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
  (
    if [[ ! -z $partition && ! -z $split ]]; then
      qsub -l h_rt=20:00:00 -V -o ${odir}/seed${s}/split.log -e ${odir}/seed${s}/split.err -N split.${s} -cwd -sync y ./run.sh --partition-and-split "${partition}:${split}" --use-components ${use_components} ${s} ${odir}/seed${s} ${config} ${tsv}
    else
      qsub -l h_rt=20:00:00 -V -o ${odir}/seed${s}/split.log -e ${odir}/seed${s}/split.err -N split.${s} -cwd -sync y ./run.sh --use-components ${use_components} ${s} ${odir}/seed${s} ${config} ${tsv}
    fi
  ) &
done
wait

echo "Finished splitting data. Computing and plotting some statistics ..."
# Find the best seed and plot the evolution of the different constraints
# during the course of the split search.
python scripts/get_best_seed.py --top-k 1 ${odir} > ${odir}/best_seed
python scripts/plot_constraints.py ${odir} ${odir}/constraint_scores.png

