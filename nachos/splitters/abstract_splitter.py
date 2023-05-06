from abc import ABC, abstractmethod
from nachos.data.Data import Dataset, Split
from nachos.similarity_functions.SimilarityFunctions import SimilarityFunctions
from nachos.constraints.Constraints import Constraints
from typing import List


class AbstractSplitter(ABC):
    @classmethod
    @abstractmethod
    def build(cls, conf: dict):
        pass

    def __init__(self, sim_fn: SimilarityFunctions, constraint_fn: Constraints):
        self.sim_fn = sim_fn
        self.constraint_fn = constraint_fn

    @abstractmethod
    def __call__(self, d: Dataset) -> List[Dataset]:
        raise NotImplementedError
        pass

    def score(self, u: Dataset, s: Split) -> float:
        return self.constraint_fn(u, s)
