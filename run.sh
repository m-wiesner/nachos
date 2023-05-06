#!/bin/bash

seed=$1
odir=$2
mkdir -p $odir
python run.py --seed ${seed} ${odir} egs/korean2/korean.yaml egs/korean2/korean_actual2.tsv 
