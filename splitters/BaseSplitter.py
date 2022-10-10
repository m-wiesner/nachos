#!/usr/bin/env python
#-*- coding: utf-8 -*-
# Copyright 2022  Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0


class BaseSplitter(object):
    @staticmethod
    def add_pargs(parser):
        raise NotImplementedError

    @classmethod
    def from_args(cls, args):
        raise NotImplementedError

    def __init__(num_clusters):
        raise NotImplementedError

    def split(self, recordings):
        raise NotImplementedError

    def clusters_to_file(self, fname):
        with open(fname, 'w') as fh:
            for i, l in enumerate(self.clustering):
                print(f'{self.fids[i]} {l}', file=fh)

    def compute_metrics(self, fname=None):
        cluster_features = {}
        cluster_recordings = {}
        with open(fname, 'w') as f:
            for k in range(self.num_clusters):
                cluster_features[k] = {}
                cluster_recordings[k] = [
                    self.recordings[self.fids[i]]
                    for i, l in enumerate(self.clustering) if l == k
                ]
                for reco in cluster_recordings[k]:
                    for feat_idx, feat in enumerate(reco):
                        if feat_idx not in cluster_features[k]:
                            cluster_features[k][feat_idx] = []
                        cluster_features[k][feat_idx].append(feat)
                cluster_size = len(cluster_recordings[k])
                cluster_percent = 100 * cluster_size / len(self.recordings)
                print(f"Cluster {k} len = {cluster_size} ({cluster_percent:0.2f}%)",
                    file=f
                )

            for j in range(self.num_clusters):
                if len(cluster_recordings[j]) == 0:
                    continue
                for k in range(self.num_clusters):
                    if j == k:
                        continue
                    if len(cluster_recordings[k]) == 0:
                        continue
              
                    for feat_idx in range(len(cluster_features[k])):
                        if self.metrics[feat_idx] == 'overlap':
                            score = sum(
                                len(feat.intersection(set().union(*cluster_features[k][feat_idx]))) > 0
                                for feat in cluster_features[j][feat_idx]
                            ) / len(cluster_recordings[j])
                        elif self.metrics[feat_idx] == 'mean':
                            score = sum(
                                float(i)
                                for i in cluster_features[j][feat_idx]
                            ) / len(cluster_recordings[j])
                        else:
                            raise ValueError("Only overlap and mean metrics supported") 
                        
                        print(f"(cluster j, cluster k, feat) "
                            f"({j}, {k}, {self.feature_names[feat_idx]}) "
                            f"--> {self.metrics[feat_idx]}={score:0.2f}", file=f
                        )
