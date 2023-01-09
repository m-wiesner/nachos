#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright 2022  Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0
from .BaseSplitter import BaseSplitter
from .BaseGraphSplitter import SimFuns
import random
import numpy as np
import networkx as nx
from tqdm import tqdm


class RandomFeatureSplitter(BaseSplitter):
    '''
          
    '''
    @staticmethod
    def add_args(parser):
        parser.add_argument('--train-ratio', type=float, default=0.8)
        parser.add_argument('--heldout-ratio', type=float, default=0.1)
        parser.add_argument('--heldout-min', type=float, default=0.01)
        parser.add_argument('--max-iter', type=int, default=100000)
        parser.add_argument('--tol', type=float, default=0.05)
        parser.add_argument('--seed', type=int, default=0) 
        parser.add_argument('--simfuns', nargs='+', type=str)

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
            seed=args.seed,
        )
    
    def __init__(self, num_features, train_ratio, heldout_ratio,
        feature_names=None, metrics=None, tol=0.05, max_iter=1000,
        heldout_min=0.01, simfuns=None, seed=0, constraint_weight=0.5,
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
        
        # If the training set is emptpy there is a good chance that the
        # the relationships between data form a complete graph. In a
        # complete graph, there is no node cut that will result in 2+
        # connected components. Hence, there is no point in running the
        # rest of this algorithm. We check to see if the affinity matrix
        # corresponding to the graph of the data is complete.
        self.check_complete()    

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
        
        train_ratio, heldout_ratio, iter_num = 999., 999., 0
        best_score = 999.
        recordings_set = set(recordings.keys())
        for iter_num in tqdm(range(self.max_iter)):
            train, heldout, _, _, _ = self.draw_random_split()
            score, train_ratio, heldout_ratio, *new_constraints = self.score(train, heldout)
            if score < best_score and heldout_ratio > self.heldout_min:
                best_score = score
                best_split = ((train, train_ratio), (heldout, heldout_ratio))
                print(f'i: {iter_num}, T: {train_ratio:0.2f}, H: {heldout_ratio:0.2f}, C: {new_constraints}, S: {score}')
                for feat_idx in range(len(self.feature_names)):
                    train_features = set().union(
                        *[recordings[i][feat_idx] for i in train]
                    )
                    heldout_features = set().union(
                        *[recordings[i][feat_idx] for i in heldout]
                    )
                    assert len(train_features.intersection(heldout_features)) == 0
        self.clustering = []
        d2 = []
        for i in fids:
            if (i not in best_split[0][0]) and (i not in best_split[1][0]):
                d2.append(i)  
        assert(len(set(d2).intersection(best_split[0][0])) == 0)
        assert(len(set(d2).intersection(best_split[1][0])) == 0)
        other_test_sets = self.make_overlapping_test_sets(
            best_split[0][0], d2,
        )
        for i in fids:
            if i in best_split[0][0]:
                self.clustering.append(0)
            elif i in best_split[1][0]:
                self.clustering.append(1)
            for j in other_test_sets:
                # Should be mututally exclusive
                if i in other_test_sets[j]:
                    self.clustering.append(j+2)
                    continue;
        self.num_clusters = 2**(len(feats)) + 2
            
    def draw_random_split(self):
        feat_subsets = []
        feat_subsets_complement = []
        feat_idxs = []
        for idx, feat in self.feats.items():
            feat_types = self.feat_types[idx]
            subset_idx = random.randint(1, 2**len(feat_types) - 2)
            subset_idx_binary = format(subset_idx, f'0{len(feat_types)}b')
            feat_idxs.append(subset_idx_binary)
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
        return train, held_out, feat_subsets, feat_subsets_complement, feat_idxs

    def check_complete(self):
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
        #num_components = nx.number_connected_components(G)
        #if num_components > 1:
        #    raise ValueError(f"A disconnected graph with {num_components} "
        #        "components was detected. This algorithm does not work on "
        #        "disconnected graphs."
        #    )
        
        is_complete = True
        for n in range(len(G)):
            try:
                if G.degree(n) != len(G) - 1:
                    is_complete = False
                    break;
            except:
                import pdb; pdb.set_trace()
        if is_complete:
            raise ValueError("A complete graph was detected. This "
                "algorithm does not work on complete graphs."
            )
    
    def make_overlapping_test_sets(self, d1, d2,):
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
        feats = self.feats
        N = len(feats)
        subsets = {i: set(d2) for i in range(0, 2**N)}
        feature_subsets = {}
        for i in range(N):
            feat_types = sorted(feats[i].keys())
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
            #score = (
            #    (1-self.constraint_weight) * score
            #    + self.constraint_weight * (
            #        np.abs(train_constraints - 0.5).sum() 
            #      + np.abs(ho_constraints - 0.5).sum()
            #      )
            #)
            return score, train_ratio, heldout_ratio, *train_constraints, *ho_constraints
        return score, train_ratio, heldout_ratio
