#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright 2022  Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0
from .BaseGraphSplitter import BaseGraphSplitter, SimFuns
import networkx as nx
import numpy as np
from networkx.algorithms.flow import shortest_augmenting_path


class GomoryHuSplitter(BaseGraphSplitter):
    def split(self, recordings):
        fids = sorted(recordings.keys())
        self.fids = fids
        self.recordings = recordings
        triu_idxs = np.triu_indices(len(recordings), k=1)
        G = nx.Graph()
        for i, j in zip(triu_idxs[0], triu_idxs[1]):
            iterator = zip(
                recordings[fids[i]],
                recordings[fids[j]],
                self.simfuns,
            )
            sims = np.array([fun(f, g) for f, g, fun in iterator])
            capacity = np.dot(self.feature_weights, sims)
            if capacity > 0:
                G.add_edge(i, j, capacity=capacity)
        import pdb; pdb.set_trace()
        T = nx.gomory_hu_tree(G, flow_func=shortest_augmenting_path)

