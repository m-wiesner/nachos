#!/bin/bash

use_components=false
partition_and_split=

. ./scripts/parse_options.sh

echo $#
echo $@
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

partition_flag=
if [[ ! -z $partition_and_split ]]; then
  partition_flag="--partition-and-split ${partition_and_split}"
fi

PYTHONUNBUFFERED=1 python run.py ${use_components_flag} ${partition_flag} \
  --seed ${seed} ${odir} ${config} ${tsv}
