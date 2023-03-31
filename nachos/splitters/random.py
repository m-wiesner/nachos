from nachos.splitters.abstract_splitter import AbstractSplitter
from nachos.similarity_functions import build_similarity_functions as build_sims
from nachos.constraints import build_constraints
from nachos.data.Data import Dataset, InvertedIndex, Split, FactoredSplit
from nachos.data.Data import collapse_factored_split
from nachos.similarity_functions.SimilarityFunctions import SimilarityFunctions
from nachos.constraints.Constraints import Constraints
from nachos.splitters import register
from typing import Optional, List, Tuple
from tqdm import tqdm
import random


@register("random")
class Random(AbstractSplitter):
    r'''
        Summary:
            Defines the random search splitter. This splitter works by randomly
            selecting a subset of values for each of the factors to include in
            a "training" set. The complements of these sets are also kept 
            track of and splits are created by intersection all of the selected
            splits for each factor.

            .. math::
                    \mathcal{T}_{I} = \bigcap_{n=1}^N S_{n}\left[I\left(n\right)\right]
                    \mathcal{H}_I = \bigcap_{i=1}^N \overline{S}_{n}\left[I\left(n\right)\right]

            The search is performed randomly and the best scoring splits are
            kept.
    '''
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
        max_iter: int = 100000,
        seed: int = 0,
    ):
        super().__init__(sim_fn, constraints)
        self.max_iter = max_iter
        self.seed = seed

    def __call__(self, d: Dataset) -> Tuple[FactoredSplit, List[float]]:
        '''
            Summary:
                Given a dataset, split according to the Random splitter
                algorithm. We draw random splits (a train and heldout split)
                keeping track of the one with the best score and return that
                split. We draw a random subset of values from each factor
                independently.

            Inputs
            -------------------------
            :param d: The dataset to split
            :type d: Dataset
            
            Returns
            -------------------------
            :return: The dataset splits
            :rtype: FactoredSplit
        '''
        # Set the random seed (in a slightly paranoid way, i.e., everywhere)
        random.seed(self.seed)
        d.set_random_seed(self.seed)

        # First verify that the graph is not complete.
        # This could take some time.
        if d.graph is None:
            d.make_graph(self.sim_fn)
        if d.check_complete():
            raise ValueError("Random Splitting cannot work on a complete graph")
       
        # Make the inverted indices
        d.make_constraint_inverted_index()
        d.make_factor_inverted_index()
          
        # Initialize some values
        indices, best_split = d.draw_random_split()
        split = collapse_factored_split(best_split) 
        best_score = self.score(d, split) 
        print(f"Iter 0: Best Score: {best_score:0.4f}")
        scores = [best_score]
        for iter_num in tqdm(range(self.max_iter)):
            indices, split = d.draw_random_split()
            collapsed_split = collapse_factored_split(split)
            score = self.score(d, collapsed_split) 
            if score < best_score:
                best_score = score
                print(f"Iter {iter_num}: Best Score: {best_score:0.4f}")
                scores.append(best_score)
                best_split = split
        return (collapse_factored_split(best_split), scores)
