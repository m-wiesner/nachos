#!/bin/bash

seed=$1
odir=$2
config=$3
tsv=$4
mkdir -p $odir
PYTHONUNBUFFERED=1 python run.py --seed ${seed} ${odir} ${config} ${tsv}
