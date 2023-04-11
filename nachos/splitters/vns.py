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


@register("vns")
class VNS(AbstractSplitter):
    @classmethod
    def build(cls, conf: dict):
        return cls(
            build_sims(conf),
            build_constraints(conf),
            max_iter=conf['max_iter'],
            max_neighbors=conf['max_neighbors'],
            seed=conf['seed'],
            num_shake_neighborhoods=conf['num_shake_neighborhoods'],
            num_search_neighborhoods=conf['num_search_neighborhoods'],
            check_complete=conf['check_complete'],
        )

    def __init__(self,
        sim_fn: SimilarityFunctions,
        constraints: Constraints,
        num_shake_neighborhoods: int = 4,
        num_search_neighborhoods: int = 10,
        max_iter: int = 200,
        max_neighbors: int = 2000,
        seed: int = 0,
        check_complete: bool = True,
    ):
        super().__init__(sim_fn, constraints)
        self.max_iter = max_iter
        self.seed = seed
        self.K = num_shake_neighborhoods
        self.L = num_search_neighborhoods
        self.max_neighbors = max_neighbors
        self.check_complete = check_complete
    
    def __call__(self, d: Dataset) -> Tuple[FactoredSplit, List[float]]:
        '''
            Summary:
                Given a dataset, split according using a Variable Neighborhood
                Search method over feasible solutions. Feasible solutions are
                constructed by drawing subsets by selecting values from each
                factor independently (and including all associated data points),
                and then intersecting these sets. The intersection of these sets
                is guaranteed to be disjoint from the intersection of the 
                complements of these sets.

            Inputs
            -------------------
            :param d: The dataset to split
            :type d: Dataset

            Returns
            ------------------
            :return: The dataset splits and scores
            :rtype: Tuple[FactoredSplit, List[float]]
        '''
        # Set the random seed (in a slightly paranoid way, i.e., everywhere)
        random.seed(self.seed)
        d.set_random_seed(self.seed)

        # First verify that the graph is not complete.
        # This could take some time.
        if d.graph is None:
            d.make_graph(self.sim_fn)
        if self.check_complete and d.is_complete():
            raise ValueError("Random Splitting cannot work on a complete graph")

        # Make the inverted indices
        d.make_constraint_inverted_index()

        # Draw the first candidate randomly
        indices, split = d.draw_random_split()
        split_collapsed = collapse_factored_split(split)
        score = self.score(d, split_collapsed)
        scores = []
        print(f"Iter 0: Best Score: {score:0.4f}")

        # Repeat the VNS algorithm steps for up to max_iter iterations
        for iter_num in tqdm(range(self.max_iter)):
            scores.append(score)
            # First draw a random point from the neighbor around the current
            # split. We will try to optimize starting from this random point
            # but if we cannot find a better point whose score is less than the
            # the score of the current split, we can start from different random
            # points in the neighborhood. We will try this up to self.K times
            # before just incrementing the iteration number.
            for k in range(1, self.K+1):
                #if k > 1:
                #    print(f"K: {k}")
                # Get a random split (shake_split) selected from the
                # neighborhood of the current split (split)
                shake_indices, shake_split = d.shake(indices, split, k)
                shake_split_collapsed = collapse_factored_split(shake_split)
                shake_score = self.score(d, shake_split_collapsed)
                # We will search exhaustively among candidates in the
                # neighborhood of shake_split for the best scoring split. If
                # the best scoring split in the neighborhood is no better than
                # the current split, we increase the size of the neighborhood
                # and search for the best scoring split from candidate splits
                # that are "farther" from the current one. See
                # self.get_neighborhood for more details about how we form the
                # neighborhoods.
                for l in range(1, self.L+1):
                    #if l > 1:
                    #    print(f"L: {l}")
                    # Find the smallest cost neighbor in the neighborhood of l
                    neighborhood = d.get_neighborhood(
                        shake_indices, shake_split, l, 
                        max_neighbors=self.max_neighbors,
                    )
                    best_score, (best_indices, best_split) = min(
                        [
                            (self.score(d, collapse_factored_split(n[1])), n)
                            for n in neighborhood
                        ]
                    )
                    # If we have found a new split that scores better than
                    # the starting point (a descent direction), then we do not
                    # have to search through more distant candidates.
                    if best_score < shake_score:
                        break
                # The the descent direction w/r to the shake_split is also a
                # descent direction w/r to the original split, then we update
                # the split to be this newly discovered descent direction. 
                if best_score < score: 
                    split = best_split
                    collapsed_split = collapse_factored_split(split)
                    for s in collapsed_split:
                        stats = self.constraint_fn.stats(d, s)
                        print(f'Stats: {stats}')
                    indices = best_indices 
                    score = best_score
                    print(f"Iter {iter_num}: Best Score: {score:0.4f}")
                    break
        return (collapse_factored_split(split), scores) 
