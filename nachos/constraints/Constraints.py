from typing import Optional, List, Any, Dict, Union
from nachos.data.Data import Dataset, Split
from nachos.constraints.abstract_constraint import AbstractConstraint


WORST_SCORE = 99999.


class Constraints(object):
    @classmethod
    def build(cls, conf: dict):
        fns = [fn.build(conf) for fn in conf['constraints']]
        weights = conf['constraint_weights']
        return cls(fns, weights)

    def __init__(self, fns: List[AbstractConstraint], weights: List[float]):
        self.fns = fns
        self.weights = weights

    def __call__(self,
        u: Dataset,
        s: Split,
        n: Optional[int] = None
    ) -> float:
        '''
            Summary:
                This function computes the discompatibility score according to
                predined constraints (self.fns) of a split s.

            Inputs
            ----------------------------
            :param u: A dataset
            :type u: Dataset
            :param s: A proposed split
            :type v: Split
            :param n: The index of the constraint with respect to which to
                compute the discompatibility score. None means compute the 
                weighted sum of all constraints
            :type n: Optional[int]

            Returns
            --------------------------
            :return: The discompatibility score
            :rtype: float
        '''
        if len(s[0]) == 0 or len(s[1]) == 0:
            return WORST_SCORE
                     
        if n is None:
            constraints_zipped = zip(
                self.weights, self.fns,
                [u.get_constraints(subset=s[0], n=m) for m in range(len(u.constraint_idxs))],
                [u.get_constraints(subset=s[1], n=m) for m in range(len(u.constraint_idxs))],
            )
            return sum(w*fn(f, g) for w, fn, f, g in constraints_zipped)
       
        subset_constraints = u.get_constraints(subset=s[0], n=n)
        # not_subset here means the complement of subset. i.e., the points
        # in u that are not in subset (s[0])
        not_subset_constraints = u.get_constraints(subset=s[1], n=n)

        return self.fns[n](subset_constraints, not_subset_constraints)
