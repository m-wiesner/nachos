#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright 2022  Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0
 
import splitters
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('features', type=str, help="file with meatadata features "
        "on which to split"
    )
    parser.add_argument('output', help="write the split to this file")
    parser.add_argument('--log', help="output file for logging metrics", default=None)
    parser.add_argument('--num-splits', type=int, default=2, help='# splits to create')
    parser.add_argument('--metrics', nargs='+', type=str, help="The list of metrics to compute for each feature")
    parser.add_argument('--feature-names', nargs='+', type=str, default=None)
    parser.add_argument('--splitter', type=str,
        choices=[
            'SpectralClusteringSplitter',
            'GomoryHuSplitter',
            'RandomFeatureSplitter',
            'GreedyFeatureSplitter',
            'MinNodeCutSplitter',
        ],
    )
    args, leftover = parser.parse_known_args()
    splitter_class = getattr(splitters, args.splitter)
    splitter_class.add_args(parser)
    args = parser.parse_args()
    
    recordings = {}
    with open(args.features, 'r') as f:
        for l in f:
            fid, features = l.strip().split(None, 1)
            features = features.split('\t')
            features_list = []
            for f in features:
                features_list.append(set(f.split(',')))
            recordings[fid] = features_list

    splitter = splitter_class.from_args(args)
    splitter.split(recordings)
    if args.log is not None:
        splitter.compute_metrics(fname=args.log)
    splitter.clusters_to_file(args.output)


if __name__ == "__main__":
    main()
