#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright 2022  Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0
from .BaseGraphSplitter import BaseGraphSplitter
from .BaseGraphSplitter import BaseGraphSplitter, SimFuns
import networkx as nx
import numpy as np
import random
from tqdm import tqdm


class DisconnectedSplitter(BaseGraphSplitter):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--train-ratio', type=float, default=0.8)
        parser.add_argument('--heldout-ratio', type=float, default=0.1)
        parser.add_argument('--heldout-min', type=float, default=0.01)
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--simfuns', nargs='+', type=str)
        parser.add_argument('--feature-weights', nargs='+', type=int)
        parser.add_argument('--max-iter', type=int, default=200)

    @classmethod
    def from_args(cls, args):
        with open(args.features, 'r', encoding='utf-8') as f:
            num_features = len(f.readline().strip().split()) - 1
        simfuns = [getattr(SimFuns, s) for s in args.simfuns]
        return cls(
            num_features,
            args.train_ratio,
            args.heldout_ratio,
            max_iter=args.max_iter,
            feature_names=args.feature_names,
            metrics=args.metrics,
            heldout_min=args.heldout_min,
            simfuns=simfuns,
            feature_weights=args.feature_weights,
            constraint_weight=args.constraint_weight,
            seed=args.seed,
        )

    def __init__(self, num_features, train_ratio, heldout_ratio,
        feature_names=None, metrics=None, tol=0.05, max_iter=1000,
        feature_weights=None, heldout_min=0.01, simfuns=None, seed=0,
        constraint_weight=0.5,
    ):
        super(DisconnectedSplitter, self).__init__(simfuns, 3, metrics=metrics,
            feature_weights=feature_weights, feature_names=feature_names,
        )
        self.train_ratio = train_ratio
        self.heldout_ratio = heldout_ratio
        self.max_iter = max_iter
        self.heldout_min = heldout_min
        self.simfuns = simfuns
        self.constraint_weight = constraint_weight
        self.seed = seed

    def split(self, recordings, constraints=None):
        random.seed(self.seed)
        fids = sorted(recordings.keys())
        self.fids = fids
        self.recordings = recordings
        self.constraints = constraints
        self.num_constraints = len(constraints[fids[0]]) if constraints is not None else None
       
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

        train_ratio, heldout_ratio = 999., 999.
        best_score = 999.
        recordings_set = set(recordings.keys())
        # Create the Graph
        triu_idxs = np.triu_indices(len(self.recordings), k=1)
        G = nx.Graph()
        for i, j in zip(triu_idxs[0], triu_idxs[1]):
            iterator = zip(
                self.recordings[self.fids[i]],
                self.recordings[self.fids[j]],
                self.simfuns,
            )
            sims = np.array([fun(f, g) for f, g, fun in iterator])
            capacity = np.dot(np.ones(len(self.feature_names),), sims)
            if capacity > 0:
                G.add_edge(i, j, capacity=capacity)

        # Check disconnected
        num_components = nx.number_connected_components(G)
        if num_components < 2:
            raise ValueError(f"A Connected graph with {num_components} "
                "components was detected. This algorithm does not work on "
                "connected graphs."
            )
        
        # We need to put components into
        components = list(nx.connected_components(G))
        best_train, best_ho, best_score = [], [], 999999999.
        for iter_num in tqdm(range(self.max_iter)):
            train, heldout, toss = [], [], []
            train_size, heldout_size = 0, 0
            random.shuffle(components)
            for c in components:
                if train_size/len(fids) < self.train_ratio:
                    train.extend([self.fids[i] for i in c])
                    train_size += len(c)
                elif heldout_size/len(fids) < self.heldout_ratio:
                    heldout.extend([self.fids[i] for i in c])
                    heldout_size += len(c)
            
            score, tr, hr, *cr = self.score(train, heldout)  
            if score < best_score and hr > self.heldout_min:
                best_score = score
                best_train = [i for i in train]
                best_ho = [i for i in heldout]
                print(f"T: {tr:0.2f}, H: {hr:0.2f}, C:{cr}, S:{best_score}")
        
        train = best_train
        heldout = best_ho
        self.clustering = []
        d2 = []
        for i in fids:
            if (i not in train) and (i not in heldout):
                d2.append(i)  
        other_test_sets = self.make_overlapping_test_sets(
            train, d2,
        )
        for i in fids:
            if i in train:
                self.clustering.append(0)
            elif i in heldout:
                self.clustering.append(1)
            for j in other_test_sets:
                # Should be mututally exclusive
                if i in other_test_sets[j]:
                    self.clustering.append(j+2)
                    continue;
        self.num_clusters = 2**(len(feats)) + 2
    
    def score(self, train, heldout):
        train_ratio = len(train) / len(self.fids)
        heldout_ratio = len(heldout) / len(self.fids)
        train_error = abs(train_ratio - self.train_ratio)
        heldout_error = abs(heldout_ratio - self.heldout_ratio)
        score = train_error + heldout_error
        if self.constraints is not None:
            if len(heldout) == 0 or len(train) == 0:
                ho_constraints = np.array([999.0 for i in range(self.num_constraints)])
                train_constraints = np.array([999. for i in range(self.num_constraints)])
            else:
                train_constraints = np.array(
                    [
                        [
                            self.constraints[fid][idx_c]
                            for idx_c in range(self.num_constraints)
                        ] for fid in train
                    ]
                ).sum(axis=0) / len(train)
                ho_constraints = np.array(
                    [
                        [
                            self.constraints[fid][idx_c]
                            for idx_c in range(self.num_constraints)
                        ] for fid in heldout
                    ]
                ).sum(axis=0) / len(heldout)
            score = (
                (1-self.constraint_weight) * score 
                + self.constraint_weight * np.abs(train_constraints - ho_constraints).sum()
            )
            #score = score + np.abs(train_constraints - 0.5).sum() + np.abs(ho_constraints - 0.5).sum()
            return score, train_ratio, heldout_ratio, *train_constraints, *ho_constraints
        return score, train_ratio, heldout_ratio
            
    def make_overlapping_test_sets(self, d1, d2):
        '''
            Given a partition (d1, d2, d3), we want to create, from d2, all the
            test sets with different amounts of overlap. For a set of N
            features, there are 2^N different kinds of overlap. This function
            returns those sets (all of which are subsets of d2). d2 is the
            set of data points that overlap in features with both d1 and d3.

            A_1, A_2, A_3, ..., A_N

            1. d2 intersect overlap feature_i --> A_1, A_2 ...
            2. 


        '''
        N = len(self.feats)
        subsets = {i: set(d2) for i in range(0, 2**N)}
        feature_subsets = {}
        for i in range(N):
            feat_types = sorted(self.feats[i].keys())
            d1_feats = set().union(*[val[i] for reco, val in self.recordings.items() if reco in d1])
            d2_overlapping_recos = [
                reco for reco, val in self.recordings.items() 
                if reco in d2 and len(val[i].intersection(d1_feats)) > 0
            ]
            feature_subsets[i] = d2_overlapping_recos
        
        for subset in subsets:
            subset_binary = format(subset, f'0{N}b')
            new_set = subsets[subset]
            for i, idx in enumerate(subset_binary): #(ABC) 011 --> A and !B and !C 
                if idx == '0':
                    new_set = new_set.intersection(feature_subsets[i])
                else:
                    new_set = new_set.intersection(
                        set(d2).difference(feature_subsets[i])
                    )
            subsets[subset] = new_set
        return subsets
