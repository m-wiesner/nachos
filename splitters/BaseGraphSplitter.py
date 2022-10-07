#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright 2022  Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0
from .BaseSplitter import BaseSplitter
import numpy as np


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

    @staticmethod
    def greater_than(f, g):
        return int(f > g)


class BaseGraphSplitter(BaseSplitter):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--feature-weights', nargs='+', type=float, default=None)
        parser.add_argument('--simfuns', nargs='+', type=str)
    
    @classmethod
    def from_args(cls, args):
        simfuns = [getattr(SimFuns, s) for s in args.simfuns]
        if SimFuns.greater_than in simfuns:
            raise ValueError("The Affinity matrix is undirected and the > "
                " similarity function is directional."
            )
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
        triu_idxs = np.triu_indices(len(recordings), k=1)
        
        for i, j in zip(triu_idxs[0], triu_idxs[1]):
            iterator = zip(
                recordings[fids[i]],
                recordings[fids[j]],
                self.simfuns,
            )
            sims = np.array([fun(f, g) for f, g, fun in iterator])
            A[i, j] = np.dot(self.feature_weights, sims)
            A[j, i] = A[i, j]
        self.A = A

