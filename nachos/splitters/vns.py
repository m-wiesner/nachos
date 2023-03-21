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


@register("vns")
class VNS(AbstractSplitter):
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
        num_shake_neighborhood: int = 4,
        num_search_neighborhood: int = 10,
        max_iter: int = 200,
        seed: int = 0,
    ):
        super().__init__(sim_fn, constraints)
        self.max_iter = max_iter
        self.seed = seed
        self.K = num_shake_neighborhood
        self.L = num_search_neighborhood
    
    def __call__(self, d: Dataset) -> Tuple[FactoredSplit, List[float]]:
        '''
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


        indices, best_split = d.draw_random_split()
        split = collapse_factored_split(best_split)
        scores = [self.score(d, split)]
        print(f"Iter 0: Best Score: {best_score:0.4f}")
        for iter_num in tqdm(range(self.max_iter)):
           pass 
