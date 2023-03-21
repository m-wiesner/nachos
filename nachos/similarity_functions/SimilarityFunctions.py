from typing import Optional, Union, List
from nachos.data.Data import Data, Dataset
from nachos.similarity_functions.abstract_similarity import AbstractSimilarity


class SimilarityFunctions(object):
    @classmethod
    def build(cls, conf: dict):
        fns = [fn.build(conf) for fn in conf['similarity_functions']]
        weights = conf['factor_weights']
        return cls(fns, weights)

    def __init__(self, fns: List[AbstractSimilarity], weights: List[float]):
        self.fns = fns
        self.weights = weights

    def __call__(self,
        u: Dataset,
        v: Dataset,
        n: Optional[int] = None
    ) -> float:
        '''
            Summary:
                This function is overloaded to operate with a few different
                kinds of data. It can either work to compare the similarities
                between two data points, between a data point and a dataset, or
                either of the previous two functions with respect to a single
                factor, n.

            Inputs
            -----------------
            :param u: A data point (defined by the Dataset class)
            :type u: Dataset
            :param v: A data set
            :type v: Dataset
            :param n: The index of the factor with respect to which to compute
                similarity. None means use the sum of all factors
            :type n: Optional[int]

            Returns
            -------------------
            :return: The similarity score
            :rtype: float
        '''
        if len(u) == 1 and len(v) == 1 and n is None:
            factors_zipped = zip(
                self.weights,
                self.fns,
                u.factors.values(),
                v.factors.values(),
            )
            return sum(w*fn(f, g) for w, fn, f, g in factors_zipped)
        elif len(u) == 1 and len(v) > 1:
            return self.score_set(u.data[0], v, n)
        elif len(u) == 1 and len(v) == 1 and n is not None:
            return self.score(u.data[0], v.data[0], n)
        else:
            raise NotImplementedError

    def score(self, u: Data, v: Data, n: int) -> float:
        return self.fns[n](u.factors[n], v.factors[n])
    
    def score_set(self, u: Data, v: Dataset, n: Optional[int] = None) -> float:
        if n is not None:
            return max(self.score(u, x.data[0], n) for x in v)
        else:
            # Return the maximum sum 
            return max(
                sum(
                    w*self.score(u, x.data[0], n)
                    for w, n in zip(self.weights, range(len(u.factors)))
                )
                for x in v
            )
