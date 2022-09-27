#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright 2022  Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0
 
import numpy as np
from sklearn.cluster import SpectralClustering
import sys
from .BaseSplitter import BaseSplitter


class SimFuns(object):
    @staticmethod
    def bool(f, g):
        return f==g

    @staticmethod
    def gaussian(f, g):
        return np.exp(-(float(f) - float(g))**2)

    @staticmethod
    def negative_gaussian(f, g):
        return 1.0 - np.exp(-(float(f) - float(g))**2)

    @staticmethod
    def set_intersect(f, g):
        return len(f.intersection(g))


class SpectralClusteringSplitter(BaseSplitter):
    '''
        This class splits the data using spectral clustering.
        An affinity matrix is created from provided metadata. This matrix can
        then be used to partition the data. This roughly corresponds to a
        relaxed normalized min-cut on the graph described by the affinity
        matrix.
    '''
    @staticmethod
    def add_args(parser):
        parser.add_argument('--feature-weights', nargs='+', type=float, default=None)
        parser.add_argument('--simfuns', nargs='+', type=str)
    
    @classmethod
    def from_args(cls, args):
        simfuns = [getattr(SimFuns, s) for s in args.simfuns]
        return cls(
            simfuns,
            args.num_splits,
            metrics=args.metrics,
            feature_weights=args.feature_weights,
            feature_names=args.feature_names,
        )
         
    def __init__(self, simfuns, num_clusters, metrics=None,
        feature_weights=None, feature_names=None,
    ):
        self.simfuns = simfuns
        self.num_clusters = num_clusters
        self.splitter = SpectralClustering(
            n_clusters=num_clusters,
            affinity='precomputed',
        )
        self.feature_weights = np.array([1.0 for f in simfuns])
        if feature_weights is not None:
            self.feature_weights = np.array(feature_weights)
        self.metrics = ['overlap' for f in simfuns]
        if metrics is not None:
            self.metrics = metrics
        self.feature_names = list(range(len(simfuns)))
        if feature_names is not None:
            self.feature_names = feature_names

    def split(self, recordings):
        fids = sorted(recordings.keys())
        self.fids = fids
        self.recordings = recordings
        A = np.zeros((len(recordings), len(recordings)))
        triu_idxs = np.triu_indices(len(recordings))
        
        for i, j in zip(triu_idxs[0], triu_idxs[1]):
            iterator = zip(
                recordings[fids[i]],
                recordings[fids[j]],
                self.simfuns,
            )
            sims = np.array([fun(f, g) for f, g, fun in iterator])
            A[i, j] = np.dot(self.feature_weights, sims)
            A[j, i] = A[i, j]
        
        clustering = self.splitter.fit(A).labels_
        cluster_sizes = {i: sum(clustering == i) for i in range(self.num_clusters)}
        clusters_sorted = [c[0] for c in sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)]
        self.clustering = list(
            map(lambda x: clusters_sorted.index(x), clustering)
        )
