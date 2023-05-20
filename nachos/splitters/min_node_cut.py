from nachos.splitters.abstract_splitter import AbstractSplitter
from nachos.similarity_functions import build_similarity_functions as build_sims
from nachos.constraints import build_constraints
from nachos.data.Data import Dataset, InvertedIndex, Split, FactoredSplit
from nachos.similarity_functions.SimilarityFunctions import SimilarityFunctions
from nachos.constraints.Constraints import Constraints
from nachos.splitters import register

from typing import Optional, List, Tuple, Generator
from tqdm import tqdm
import random


@register("min_node_cut")
class MinNodeCut(AbstractSplitter):
    @classmethod
    def build(cls, conf: dict):
        return cls(
            build_sims(conf),
            build_constraints(conf),
            max_iter=conf['max_iter'],
            seed=conf['seed'],
        )

    def __init__(self,
        sim_fn: SimilarityFunctions,
        constraints: Constraints,
        max_iter: int = 200,
        seed: int = 0,
    ):
        super().__init__(sim_fn, constraints)
        self.max_iter = max_iter
        self.seed = seed

    def __call__(self, d: Dataset) -> Tuple[FactoredSplit, List[float]]:
        '''
            Summary:
                Given a dataset, split according to a search over minimum-st
                node cuts, picking the s-source and t-target vetrices that 
                minimize the constraint cost function of the split.

            Inputs
            ---------------------
            :param d: The dataset to split
            :type d: Dataset

            Returns
            --------------------
            :return: The dataset split and scores
            :rtype: Tuple[FactoredSplit, List[float]]
        '''
        random.seed(self.seed)
        d.set_random_seed(self.seed)

        # Set the random seed (in a slightly paranoid way, i.e., everywhere)
        random.seed(self.seed)
        d.set_random_seed(self.seed)

        # First verify that the graph is not complete.
        # This could take some time.
        if d.graph is None:
            d.make_graph(self.sim_fn)
        if d.check_complete():
            raise ValueError("Minimum Node Cut Splitting cannot work on a "
                "complete graph"
            )
        if d.check_disconnected():
            raise ValueError("Minimum Node Cut Splitting cannot work on a "
                "disconnected graph"
            )

        # Initialize some values
        best_split = d.draw_random_node_cut()
        constraint_scores = self.score(d, best_split, all_scores=True)
        best_score = constraint_scores['total']
        print(f"Iter 0: Best Score: {best_score:0.4f}")
        scores = []
        # Try up to max_iter number of st cuts
        for iter_num in tqdm(range(self.max_iter)):
            scores.append(constraint_scores)
            split = d.draw_random_node_cut()
            constraint_scores = self.score(d, split, all_scores=True)
            scores = constraint_score['total']
            if score < best_score:
                best_score = score
                print(f"Iter {iter_num}: Best Score: {best_score:0.4f}")
                best_split = split
        return (best_split, scores)
