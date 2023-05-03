#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright 2022  Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0
 
import numpy as np
from sklearn.cluster import SpectralClustering
import sys
from .BaseGraphSplitter import BaseGraphSplitter


class SpectralClusteringSplitter(BaseGraphSplitter):
    '''
        This class splits the data using spectral clustering.
        An affinity matrix is created from provided metadata. This matrix can
        then be used to partition the data. This roughly corresponds to a
        relaxed normalized min-cut on the graph described by the affinity
        matrix.
    '''
    def __init__(self, simfuns, num_clusters, metrics=None,
        feature_weights=None, feature_names=None,
    ):
        super(SpectralClusteringSplitter, self).__init__(
            simfuns, num_clusters,
            metrics=metrics, feature_weights=feature_weights,
            feature_names=feature_names,
        )
        self.splitter = SpectralClustering(
            n_clusters=num_clusters,
            affinity='precomputed',
        )
    
    def split(self, recordings):
        super().split(recordings)
        clustering = self.splitter.fit(self.A).labels_
        cluster_sizes = {i: sum(clustering == i) for i in range(self.num_clusters)}
        clusters_sorted = [c[0] for c in sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)]
        self.clustering = list(
            map(lambda x: clusters_sorted.index(x), clustering)
        )
