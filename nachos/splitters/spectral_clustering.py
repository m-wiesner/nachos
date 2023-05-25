import numpy as np
from sklearn.cluster import SpectralClustering
from nachos.splitters.abstract_splitter import AbstractSplitter
from nachos.similarity_functions import build_similarity_functions as build_sims
from nachos.constraints import build_constraints
from nachos.data.Data import Dataset, InvertedIndex, Split, FactoredSplit
from nachos.data.Data import collapse_factored_split
from nachos.similarity_functions.SimilarityFunctions import SimilarityFunctions
from nachos.constraints.Constraints import Constraints
from nachos.splitters import register
from typing import Optional, List, Tuple, Generator
from tqdm import tqdm
import random
import networkx as nx


@register("spectral_clustering")
class SpectralClustering(AbstractSplitter):
    @classmethod
    def build(cls, conf: dict):
        return cls(
            build_sims(conf),
            build_constraints(conf),
            num_clusters=conf.get('num_cluster', 2),  
            smooth=conf.get('spectral_clustering_smoothing', 0.0001),
        )

    def __init__(self,
        sim_fn: SimilarityFunctions,
        constraints: Constraints,
        num_clusters: float = 2,
        smooth: float = 0.0001,
    ):
        super().__init__(sim_fn, constraints)
        self.num_clusters = num_clusters
        # Since spectral clustering will not work on disconnected graphs, we
        # simply add a small weight on each arc to make sure the graph is not
        # disconnected.
        self.smooth = smooth

    def __call__(self, d: Dataset) -> Tuple[FactoredSplit, List[float]]:
        sc = SpectralClustering(
            n_clusters=self.num_clusters,
            affinity='precomputed',
        )
        # We add self.smooth to each arc, but we want to remove it from the
        # diagonal entries.
        A = nx.to_numpy_array(d.graph) + self.smooth*(1.0 - np.eye(A.shape[0])) 
        clustering = sc.fit(A).labels_
        
        # Sort the clusters by size (number of nodes)
        cluster_sizes = {i: sum(clustering == i) for i in range(self.num_clusters)}
        clusters_sorted = [c[0] for c in sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)]
        clustering = list(
            map(lambda x: clusters_sorted.index(x), clustering)
        )

        # Assume clusters, 0 and 1, are the train/test split
        subset = [d.data[i] for i, j in enumerate(clustering) if j == 0]
        not_subset = [d.data[i] for i, j in enumerate(clustering) if j == 1]
        split = (subset, not_subset)
        constraint_scores = self.score(d, split, all_scores=True)
        scores = [constraint_scores]
        return (split, scores)
