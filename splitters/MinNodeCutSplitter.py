#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright 2022  Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0
from .BaseSplitter import BaseSplitter
from .BaseGraphSplitter import BaseGraphSplitter, SimFuns
import random
import numpy as np
import networkx as nx
from copy import deepcopy
from tqdm import tqdm


class MinNodeCutSplitter(BaseGraphSplitter):
    '''
        We randomly sample two independent nodes and find the minimum node cut
        between these nodes. This cut will disconnect the graph forming at
        least 3 components: (i) the node cut, which is overlapping with respect
        to one or more of the other components in one or more features; and two
        components (or more)formed by the removal of the node cut. Combining
        disconnected components gives the training and test set. To form the
        training and test sets, we randomly select components (we don't just start
        with the largest as this can result in biased training / test splits),
        and greedily add them to the training set until either we have reached
        the desired train_ratio (within some tolerance), or there is only one
        more component remaining.
        
        In practice, most node cuts will result in exactly two components. In
        this case, the large component is included in training and the smaller
        component becomes the held out set.     
    '''
    @staticmethod
    def add_args(parser):
        parser.add_argument('--train-ratio', type=float, default=0.8)
        parser.add_argument('--heldout-ratio', type=float, default=0.1)
        parser.add_argument('--heldout-min', type=float, default=0.01)
        parser.add_argument('--max-iter', type=int, default=100)
        parser.add_argument('--tol', type=float, default=0.05)
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--simfuns', nargs='+', type=str)
        parser.add_argument('--feature-weights', nargs='+', type=int)

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
            heldout_min=0.01,
            simfuns=simfuns,
            feature_weights=args.feature_weights,
            seed=args.seed,
        )

    def __init__(self, num_features, train_ratio, heldout_ratio,
        feature_names=None, metrics=None, tol=0.05, max_iter=1000,
        feature_weights=None, heldout_min=0.01, simfuns=None, seed=0,
    ):
        super(MinNodeCutSplitter, self).__init__(simfuns, 3, metrics=metrics,
            feature_weights=feature_weights, feature_names=feature_names,
        )
        self.train_ratio = train_ratio
        self.heldout_ratio = heldout_ratio
        self.tol = tol
        self.max_iter = max_iter
        self.heldout_min = heldout_min
        self.simfuns = simfuns
        self.seed = seed

    def split(self, recordings):
        random.seed(self.seed)
        fids = sorted(recordings.keys())
        self.fids = fids
        self.recordings = recordings
       
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
                #G.add_edge(j, i, capacity=capacity)

        # Sample node cuts.
        for iter_num in tqdm(range(self.max_iter)):
            if (
                    abs(train_ratio - self.train_ratio) <= self.tol and
                    abs(heldout_ratio - self.heldout_ratio) <= self.tol
            ):
                break;
            train, heldout, cut = self.draw_random_node_cut(G)
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

    def draw_random_node_cut(self, G):
        nodes_are_adjacent = True
        while nodes_are_adjacent:
            sampled_nodes = random.sample(range(len(G)), 2)
            if sampled_nodes[1] not in G.neighbors(sampled_nodes[0]):
                nodes_are_adjacent = False

        cut = nx.minimum_node_cut(G, s=sampled_nodes[0], t=sampled_nodes[1])
        H = deepcopy(G)
        for n in cut:
            H.remove_node(n)
        cut = [self.fids[i] for i in cut]
        train, heldout = [], []
        num_train, num_heldout = 0, 0
        comps = sorted(nx.connected_components(H), key=len, reverse=True)
        num_comps = len(comps)
        for i, comp in enumerate(comps):
            comp_ = [self.fids[i] for i in comp]
            if i == num_comps - 1:
                heldout.append(comp_)
                break
            
            train_ratio = num_train / len(G)
            if train_ratio < self.train_ratio:
                train.append(comp_)
            else:
                heldout.append(comp_)
        return set().union(*train), set().union(*heldout), cut
