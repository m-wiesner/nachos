#!/bin/bash

use_components=false

. ./scripts/parse_options.sh

if [ $# -ne 4 ]; then
  echo "Usage: ./run.sh --use-components <true/false> <seed> <odir> <config> <tsv>"
  exit 1;
fi

seed=$1
odir=$2
config=$3
tsv=$4
mkdir -p $odir

use_components_flag=
if $use_components; then
  use_components_flag="--use-components"
fi

PYTHONUNBUFFERED=1 python run.py ${use_components_flag} --seed ${seed} ${odir} ${config} ${tsv}
