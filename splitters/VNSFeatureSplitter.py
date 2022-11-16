#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright 2022  Johns Hopkins University (Author: Matthew Wiesner, Kiran Karra)
# Apache 2.0
from .BaseGraphSplitter import SimFuns
from .RandomFeatureSplitter import RandomFeatureSplitter
from .GreedyFeatureSplitter import GreedyFeatureSplitter
import random
import numpy as np
import networkx as nx
from copy import deepcopy
from tqdm import tqdm
from itertools import groupby


class VNSFeatureSplitter(GreedyFeatureSplitter):
    @staticmethod
    def add_args(parser):
        RandomFeatureSplitter.add_args(parser)
        parser.add_argument('--num-shake-neighborhoods', type=int, default=2)
        parser.add_argument('--num-search-neighborhoods', type=int, default=4)
    
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
            constraint_weight=args.constraint_weight,
            num_shake_neighborhoods=args.num_shake_neighborhoods,
            num_search_neighborhoods=args.num_search_neighborhoods,
            seed=args.seed,
        )

    
    def __init__(self, num_features, train_ratio, heldout_ratio,
        feature_names=None, metrics=None, tol=0.05, max_iter=100,
        heldout_min=0.01, simfuns=None, seed=0, num_shake_neighborhoods=2,
        num_search_neighborhoods=4, constraint_weight=0.5,
    ):
        super(GreedyFeatureSplitter, self).__init__(num_features, train_ratio,
            heldout_ratio, feature_names=feature_names, metrics=metrics,
            tol=tol, max_iter=max_iter, heldout_min=heldout_min,
            simfuns=simfuns, constraint_weight=constraint_weight, seed=seed,
        )
        self.num_search_neighborhoods = num_search_neighborhoods
        self.num_shake_neighborhoods = num_shake_neighborhoods

    def split(self, recordings, constraints=None):
        random.seed(self.seed)
        fids = sorted(recordings.keys())
        self.fids = fids
        self.recordings = recordings
        self.constraints = constraints
        self.num_constraints = len(constraints[fids[0]]) if constraints is not None else None
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
       
        self.feats = feats
        self.feat_types = [sorted(feat.keys()) for feat in feats.values()] 
        recordings_set = set(recordings.keys())
        # Check that the graph is not complete
        self.check_complete()
        
        # Draw first candidate
        train, held_out, curr_sets, curr_complements, curr_feats = self.draw_random_split()
        score_init = 0
        score = self.score(train, held_out)[0] 
        for iter_num in tqdm(range(self.max_iter)):
            score_init = score
            for k in range(self.num_shake_neighborhoods):
                new_t, new_h, new_sets, new_complements, new_feats = self.shake(
                    curr_sets, curr_complements, curr_feats, k,
                )
                new_score, new_tr, new_hr, *new_constraints = self.score(new_t, new_h)
                best_score = new_score
                for l in range(self.num_search_neighborhoods): 
                    for t, h, fs, fsc, fi in self.get_neighborhood(new_sets, new_complements, new_feats, l,):
                        new_new_score, new_new_tr, new_new_hr, *new_new_constraints = self.score(t, h)
                        if new_new_score < best_score:
                            best_score = new_new_score
                            best_sets, best_complements, best_feats = fs, fsc, fi
                            best_t, best_h = t, h
                            best_tr, best_hr = new_new_tr, new_new_hr
                            best_constraints = new_new_constraints
                    if best_score < new_score:
                        new_score = best_score
                        new_curr_sets = best_sets
                        new_curr_complements = best_complements
                        new_curr_feats = best_feats
                        new_t, new_h = best_t, best_h
                        new_tr, new_hr = best_tr, best_hr
                        new_constraints = best_constraints
                        break
                if (new_score < score) and (new_hr >= self.heldout_min):
                    score = new_score 
                    curr_sets = new_curr_sets
                    curr_complements = new_curr_complements
                    curr_feats = new_curr_feats
                    train, held_out = new_t, new_h
                    print("--------------------")
                    print(f'T: {new_tr:0.2f}, H: {new_hr:0.2f}, C: {new_constraints}, S: {score}')
                    break 
        
        self.clustering = []
        d2 = []
        for i in fids:
            if (i not in train) and (i not in held_out):
                d2.append(i)  
        other_test_sets = self.make_overlapping_test_sets(
            train, d2,
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

    def shake(self, curr_sets, curr_complements, curr_feats, k):
        generator = self.get_neighborhood(curr_sets, curr_complements, curr_feats, k)            
        return next(generator, None)
                     
    def nearby_splits(self, curr_sets, curr_complements, curr_feats):
        # curr_feats includes for each feature, an index in 
        #
        #          (0, 2^len(feat_types))
        # 
        # corresponding to the set of values that are included in the set.
        # The neighborhood we want is any set that differs by the
        # inclusion or exclusion of one value for a specific set. For example
        # assuming 5 speakers, and 3 prompts with 
        #
        # curr_feats = (['spk1', 'spk2', 'spk4'], ['prompt1', 'prompt2'])
        #
        # then the neighborhodd N(curr_feats) is:    11010
        # N(curr_feats)[0] = [
        #    ['spk1', 'spk2',],                      11000
        #    ['spk1', 'spk4',],                      10010
        #    ['spk2', 'spk4',],                      01010
        #    ['spk1', 'spk2', 'spk3', 'spk4',],      11110
        #    ['spk1', 'spk2', 'spk4', 'spk5',],      11011
        # ]
        # N(curr_feats)[1] = [                       110
        #    ['promt1',],                            100
        #    ['prompt2',],                           010
        #    ['prompt1', 'prompt2', 'prompt3'],      111
        # ]

        # curr_sets is [ feat1_set, feat2_set]
        # curr_feats is [feat1_binary_values, feat2_binary_values]
        feats = self.feats
        k = 0
        values = [curr_feats[idx] for idx in feats]
        sequence = list(range(len(values)))
        random.shuffle(sequence)
        for idx_v in sequence:
            v = values[idx_v]
            feat = feats[idx_v]
            for n in self._1bit_different_numbers(v):
                feat_subsets = [s for s in curr_sets]
                feat_subsets_complements = [s for s in curr_complements]
                feat_idxs = [f for f in curr_feats]
                feat_idxs[idx_v] = n
                include_speakers, exclude_speakers = [], []
                for i, j in enumerate(n):
                    if j == '1':
                        include_speakers.append(self.feat_types[idx_v][i])
                    elif j == '0':
                        exclude_speakers.append(self.feat_types[idx_v][i])
                feat_subsets[idx_v] = set().union(
                    *[feat[spk] for spk in include_speakers]
                )
                feat_subsets_complements[idx_v] = set().union(
                    *[feat[spk] for spk in exclude_speakers]
                )
                intersection = feat_subsets[idx_v].intersection(feat_subsets_complements[idx_v])
                feat_subsets[idx_v] = feat_subsets[idx_v].difference(intersection)
                feat_subsets_complements[idx_v] = feat_subsets_complements[idx_v].difference(intersection)
                train = set.intersection(*feat_subsets)
                held_out = set.intersection(*feat_subsets_complements)
                yield train, held_out, feat_subsets, feat_subsets_complements, feat_idxs
                  
    def get_neighborhood(self, curr_sets, curr_complements, curr_feats, k):
        # k is the number of nested iterations N(N(...N(x))) to perform
        neighbors = list(
            map(
                lambda x: (x, 1), 
                self.nearby_splits(
                    curr_sets, curr_complements, curr_feats
                )
            )
        )
        while len(neighbors) > 0:
            (t, h, fs, fsc, fi), idx_k = neighbors.pop()
            if idx_k < k:
                for x in self.nearby_splits(fs, fsc, fi):
                    neighbors.append((x, idx_k + 1))
            else:
                yield t, h, fs, fsc, fi
                
             
    def _1bit_different_numbers(self, v):
        # Special edge case for len(v) == 2
        if v == '01':
            yield '10'
        elif v == '10':
            yield '01'
        # We shuffle to ensure we are not always trying the same neighborhoods
        # first
        sequence = list(range(len(v)))
        random.shuffle(sequence)
        for i in sequence:
            new_val = list(v)
            new_val[i] = '1' if v[i] == '0' else '0'
            g = groupby(new_val, lambda x: x)
            if not (next(g, True) and not next(g, False)): 
                yield ''.join(new_val)

      
