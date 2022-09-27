#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright 2022  Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0
from .BaseSplitter import BaseSplitter
from itertools import chain


class BoxSplitter(BaseSplitter):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--feature-order', nargs='+', type=int, default=None)
        parser.add_argument('--heldout-fractions', nargs='+', type=float, default=None)

    @classmethod
    def from_args(cls, args):
        with open(args.features, 'r', encoding='utf-8') as f:
            num_features = len(f.readline().strip().split()) - 1
        return cls(
            num_features,
            args.num_splits,
            args.heldout_fractions,
            feature_names=args.feature_names,
            feature_order=args.feature_order,
            metrics=args.metrics,
        )
    
    def __init__(self, num_features, num_clusters, heldout_fractions,
        feature_names=None, metrics=None, feature_order=None,
    ):
        self.metrics = ['overlap' for f in range(num_features)]
        if metrics is not None:
            self.metrics = metrics

        self.num_clusters = num_clusters
        self.feature_names = list(range(num_features))
        if feature_names is not None:
            self.feature_names = feature_names
   
        self.order = list(range(num_features))
        if feature_order is not None:
            self.order = feature_order

        self.heldout_fractions = heldout_fractions
    
    def split(self, recordings):
        fids = sorted(recordings.keys())
        self.fids = fids
        self.recordings = recordings
        feats = {}
        feats_cooccur = {}
        for fid in fids:
            for idx, feat_set in enumerate(recordings[fid]):
                if idx not in feats:
                    feats[idx] = {}
                    feats_cooccur[idx] = {}
                for feat in feat_set:
                    if feat not in feats[idx]:
                        feats[idx][feat] = []
                        feats_cooccur[idx][feat] = {}
                    feats[idx][feat].append(fid)
                    for idx2, feat_set2 in enumerate(recordings[fid]):
                        if idx2 not in feats_cooccur[idx][feat]:
                            feats_cooccur[idx][feat][idx2] = set()
                        feats_cooccur[idx][feat][idx2] = feats_cooccur[idx][feat][idx2].union(feat_set2)
      
        # for each feature, sort the values in order of the fewest of the next
        # feature that appeared. i.e., the speakers with the fewest number of
        # prompts, or interviewers who interviewed the fewest subjects
        feats_list = {}
        for idx, order in enumerate(self.order[:0:-1]):
            feats_list[order] = sorted(
                feats_cooccur[order].items(), key=lambda x: len(x[1][self.order[::-1][idx+1]])
            )
        
        heldout_feats = {} 
        heldout_feats_fids = {}
        # Using the specified order find the splits
        import pdb; pdb.set_trace()
        heldout_frac = self.heldout_fractions[self.order[-1]]
        num_heldout = max(
                1, int(len(feats_list[self.order[-1]]) * heldout_frac)
        )
        heldout_feats[self.order[-1]] = [f[0] for f in feats_list[self.order[-1]][0:num_heldout]]
        heldout_feats_fids[self.order[-1]] = set().union(
            *[feats[self.order[-1]][f] for f in heldout_feats[self.order[-1]]]
        )
        for idx, order in enumerate(self.order[-2::-1]):
            heldout = [
                feat2 
                for feat in heldout_feats[self.order[idx+1]]
                    for feat2 in feats_cooccur[self.order[idx+1]][feat][order]
            ]
            
            heldout = set().union(heldout)
            heldout_feats[order] = [h for h in heldout]
            heldout_feats_fids[order] = set().union(
                *[feats[order][f] for f in heldout_feats[order]]
            )

        import pdb; pdb.set_trace() 
        train_fids = []
        for f in recordings:
            is_train=True
            for idx in range(len(heldout_feats_fids)):
                is_train = is_train and (f not in heldout_feats_fids[idx]) 
            if is_train:
                train_fids.append(f)



