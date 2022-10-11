#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright 2022  Johns Hopkins University (Author: Matthew Wiesner, Kiran Karra)
# Apache 2.0
from .BaseSplitter import BaseSplitter
from .BaseGraphSplitter import SimFuns
from .RandomFeatureSplitter import RandomFeatureSplitter
import random
import numpy as np
import networkx as nx
from copy import deepcopy
from tqdm import tqdm


class GreedyFeatureSplitter(RandomFeatureSplitter):
    @staticmethod
    def add_args(parser):
        RandomFeatureSplitter.add_args(parser)
        parser.add_argument('--neighborhood-size', type=int, default=10)
     
    @classmethod
    def from_args(cls, args):
        with open(args.features, 'r', encoding='utf-8') as f:
            num_features = len(f.readline().strip().split()) - 1
        simfuns = [getattr(SimFuns, s) for s in args.simfuns]
        return cls(
            num_features,
            args.train_ratio,
            args.heldout_ratio,
            feature_names=args.feature_names,
            metrics=args.metrics,
            tol=args.tol,
            max_iter=args.max_iter,
            heldout_min=args.heldout_min,
            simfuns=simfuns,
            neighborhood_size=args.neighborhood_size,
            seed=args.seed,
        )

    def __init__(self, num_features, train_ratio, heldout_ratio,
        feature_names=None, metrics=None, tol=0.05, max_iter=100,
        heldout_min=0.01, simfuns=None, seed=0, neighborhood_size=10
    ):
        super(GreedyFeatureSplitter, self).__init__(num_features, train_ratio,
            heldout_ratio, feature_names=feature_names, metrics=metrics,
            tol=tol, max_iter=max_iter, heldout_min=heldout_min,
            simfuns=simfuns, seed=seed,
        )
        self.neighborhood_size = neighborhood_size
    
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
       
        recordings_set = set(recordings.keys())
        # Draw first candidate
        train, held_out, curr_sets, curr_complements = self.draw_random_split(feats, recordings_set)
        train_ratio = len(train) / len(fids)
        if train_ratio == 0:
            self.check_complete()
        heldout_ratio = len(held_out) / len(fids)
        for iter_num in tqdm(range(self.max_iter)):
            if (
                abs(train_ratio - self.train_ratio) <= self.tol and
                abs(heldout_ratio - self.heldout_ratio) <= self.tol
            ):
                break 

            score, _, _ = self.score(train, held_out)
            samples = self.draw_k_random_samples(feats, recordings_set)
            curr_sets, curr_complements = self.update_curr_sets(
                curr_sets, curr_complements, samples, score
            ) 
            train = set.intersection(*curr_sets)
            held_out = set.intersection(*curr_complements)
                    
        self.clustering = []
        d2 = []
        for i in fids:
            if (i not in train) and (i not in held_out):
                d2.append(i)  
        other_test_sets = self.make_overlapping_test_sets(
            train, d2, feats,
        )
        for i in fids:
            if i in train:
                self.clustering.append(0)
            elif i in held_out:
                self.clustering.append(1)
            for j in other_test_sets:
                # Should be mututally exclusive
                if i in other_test_sets[j]:
                    self.clustering.append(j+2)
                    continue;
        self.num_clusters = 2**(len(feats)) + 2

    def draw_k_random_samples(self, feats, recordings_set):
        samples = []
        for k in range(self.neighborhood_size):
            _, _, subsets, complements = self.draw_random_split(feats, recordings_set) 
            samples.append((subsets, complements))    
        return samples

    def update_curr_sets(self, curr_sets, curr_complements, samples, score):
        best_score = score
        best_update = curr_sets
        best_complement = curr_complements
        for i, f in enumerate(curr_sets):
            update = [j for j in curr_sets]
            complement = [j for j in curr_complements]
            for s_idx, s in enumerate(samples):       
                update[i] = s[0][i]
                complement[i] = s[1][i]
                # evaluate
                train = set.intersection(*update)
                held_out = set.intersection(*complement)
                score, tr, hr = self.score(train, held_out)
                if score > best_score and len(held_out) / len(self.fids) > self.heldout_min:
                    print("---------------------------------")
                    print(f'T: {tr:0.2f}, H: {hr:0.2f}, S: {score}')
                    best_score = score
                    best_update = [j for j in update]
                    best_complement = [j for j in complement]
        return best_update, best_complement
    
    def score(self, train, heldout):
        train_ratio = len(train) / len(self.fids)
        heldout_ratio = len(heldout) / len(self.fids)
        train_error = abs(train_ratio - self.train_ratio)
        heldout_error = abs(heldout_ratio - self.heldout_ratio)
        score = 2 - (train_error + heldout_error) - abs(train_error - heldout_error)
        return score, train_ratio, heldout_ratio
           
