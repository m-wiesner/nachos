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
            'VNSFeatureSplitter',
            'MinNodeCutSplitter',
        ],
    )
    parser.add_argument('--feat-idxs', nargs='+', type=int, default=None)
    parser.add_argument('--constraint-idxs', nargs='+', type=int, default=None)
    parser.add_argument('--constraint-weight', type=float, default=None) 
    args, leftover = parser.parse_known_args()
    splitter_class = getattr(splitters, args.splitter)
    splitter_class.add_args(parser)
    args = parser.parse_args()
    
    recordings, constraints = {}, {}
    with open(args.features, 'r') as f:
        for l in f:
            fid, features = l.strip().split(None, 1)
            features = features.split('\t')
            features_list, constraints_list = [], []
            for f_idx in args.feat_idxs:
                f = features[f_idx]
                features_list.append(set(f.split(',')))
            if args.constraint_idxs is not None:
                for c_idx in args.constraint_idxs:
                    c = features[c_idx]
                    constraints_list.append(float(c))
                constraints[fid] = constraints_list
            recordings[fid] = features_list
    splitter = splitter_class.from_args(args)
    if len(constraints) > 0:
        splitter.split(recordings, constraints=constraints)
    else:
        splitter.split(recordings)
    if args.log is not None:
        splitter.compute_metrics(fname=args.log)
    splitter.clusters_to_file(args.output)


if __name__ == "__main__":
    main()
