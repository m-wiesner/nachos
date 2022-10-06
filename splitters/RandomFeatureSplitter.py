#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright 2022  Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0
from .BaseSplitter import BaseSplitter
import random


class RandomFeatureSplitter(BaseSplitter):
    '''
          
    '''
    @staticmethod
    def add_args(parser):
        parser.add_argument('--train-ratio', type=float, default=0.8)
        parser.add_argument('--heldout-ratio', type=float, default=0.1)
        parser.add_argument('--heldout-min', type=float, default=0.01)
        parser.add_argument('--max-iter', type=int, default=2)
        parser.add_argument('--tol', type=float, default=0.05)
        parser.add_argument('--seed', type=int, default=0) 

    @classmethod
    def from_args(cls, args):
        with open(args.features, 'r', encoding='utf-8') as f:
            num_features = len(f.readline().strip().split()) - 1
        return cls(
            num_features,
            args.train_ratio,
            args.heldout_ratio,
            feature_names=args.feature_names,
            metrics=args.metrics,
            tol=args.tol,
            max_iter=args.max_iter,
            heldout_min=0.01,
            seed=args.seed,
        )
    
    def __init__(self, num_features, train_ratio, heldout_ratio,
        feature_names=None, metrics=None, tol=0.05, max_iter=1000,
        heldout_min=0.01, seed=0, 
    ):
        self.metrics = ['overlap' for f in range(num_features)]
        if metrics is not None:
            self.metrics = metrics

        self.feature_names = list(range(num_features))
        if feature_names is not None:
            self.feature_names = feature_names
   
        self.train_ratio = train_ratio
        self.heldout_ratio = heldout_ratio
        self.tol = tol
        self.max_iter = max_iter
        self.heldout_min = heldout_min
        self.seed = seed 
    
    def split(self, recordings):
        random.seed(self.seed)
        fids = sorted(recordings.keys())
        self.fids = fids
        self.recordings = recordings
        
        # We want to select random subsets by partitioning on each feature.
        # Let F_i denote the set of subsets created by all possible (non-empty)
        # partitions on the i-th feature. We will randomly select a subset,
        #
        #     S_i ~ F_i
        #
        # The training and heldout sets are then
        #
        # intersection_i(S_i) , intersection_i(!S_i)
        feats = {}
        for fid in fids:
            for idx, feat_set in enumerate(recordings[fid]):
                if idx not in feats:
                    feats[idx] = {}
                for feat in feat_set:
                    if feat not in feats[idx]:
                        feats[idx][feat] = []
                    feats[idx][feat].append(fid)
       
        train_ratio, heldout_ratio, iter_num = 999., 999., 0
        best_score = 999.
        recordings_set = set(recordings.keys())
        while (
            (
                abs(train_ratio - self.train_ratio) > self.tol or
                abs(heldout_ratio - self.heldout_ratio) > self.tol
            )
            and iter_num < self.max_iter
        ):
            train, heldout = self.draw_random_split(feats, recordings_set)
            train_ratio = len(train) / len(fids)
            heldout_ratio = len(heldout) / len(fids)
            train_score = abs(train_ratio - self.train_ratio)
            heldout_score = abs(heldout_ratio - self.heldout_ratio)
            score = train_score + heldout_score
            if score < best_score and heldout_ratio > self.heldout_min:
                best_score = score
                best_split = ((train, train_ratio), (heldout, heldout_ratio))
                print(f'i: {iter_num}, T: {train_ratio:0.2f}, H: {heldout_ratio:0.2f}')
                for feat_idx in range(len(self.feature_names)):
                    train_features = set().union(
                        *[recordings[i][feat_idx] for i in train]
                    )
                    heldout_features = set().union(
                        *[recordings[i][feat_idx] for i in heldout]
                    )
                    assert len(train_features.intersection(heldout_features)) == 0
            iter_num += 1
        self.clustering = []
        for i in fids:
            if i in best_split[0][0]: # train
                self.clustering.append(0)
            elif i in best_split[1][0]:
                self.clustering.append(1)
            else:
                self.clustering.append(2)
        self.num_clusters = 3
            
    def draw_random_split(self, feats, recordings):
        feat_subsets = []
        feat_subsets_complement = []
        for idx, feat in feats.items():
            feat_types = sorted(feat.keys())
            subset_idx = random.randint(0, 2**len(feat_types) - 1)
            subset_idx_binary = format(subset_idx, f'0{len(feat_types)}b')
            include_speakers, exclude_speakers = [], []
            for i, j in enumerate(subset_idx_binary):
                if j == '1':
                    include_speakers.append(feat_types[int(i)])
                elif j == '0':
                    exclude_speakers.append(feat_types[int(i)])
            feat_subset = set().union(
                *[feat[spk] for spk in include_speakers]
            )
            feat_subset_complement = set().union(
                *[feat[spk] for spk in exclude_speakers]
            )
            # For multilabel problems there may be intersection 
            intersection = feat_subset.intersection(feat_subset_complement)
            feat_subset = feat_subset.difference(intersection)
            feat_subset_complement = feat_subset_complement.difference(intersection)
            feat_subsets.append(feat_subset)
            feat_subsets_complement.append(feat_subset_complement)
        train = set.intersection(*feat_subsets)
        held_out = set.intersection(*feat_subsets_complement)
        return train, held_out
        
