from typing import Union, Generator, Optional
from nachos.constraints.abstract_constraint import AbstractConstraint
from nachos.constraints import register
from typing import Any
import numpy as np


@register('max_entropy')
class MaxEntropy(AbstractConstraint):
    r'''
        Summary:
            This constraint tries to enforce a uniform distribution over
            selected items (without a particular vocabulary set). For instance
            if we select a set of speakers or documents as a test set, then the
            test set should ideally not be dominated by any one document, or 
            speaker. In general we only apply this to the test set, so unlike
            the other constraints, we will actually only consider values in
            the second of the two sets created by a splitter.
            
            Since we are trying to maximize the entropy, the cost is 
            
            .. math::
                H\left(X\right) =  \log{|V|} + \sum_{x} w\left(x\right) \log{p\left(x\right)}
            
            where V is the domain of variable, x.
    '''

    @classmethod
    def build(cls, conf: dict):
        return cls(
            reduction='concat',
            set1_w=conf.get('values', (0.0, 1.0))[0],
            set2_w=conf.get('values', (0.0, 1.0))[1],
        )

    def __init__(self, set1_w=0.0, set2_w=1.0, reduction='concat'):
        super().__init__(reduction=reduction)
        self.set1_w = set1_w
        self.set2_w = set2_w 
    
    def __call__(self,
        c1: Union[list, Generator],
        c2: Union[list, Generator],
        weights1: Optional[Union[list, Generator]] = None,
        weights2: Optional[Union[list, Generator]] = None,
    ) -> float:
        '''
            Summary:
                Computes the entropy over the distribution of values present in
                c2. This function completely ignores c1.

            Inputs
            -------------------------------------------
            :param c1: the values to constrain in dataset 1
            :type c1: Union[list, Generator]
            :param c2: the values to constrain in dataset 2
            :type c2: Union[list, Generator]
            :param weights1: the list of weights on each value in c1
            :type weights1: Optional[Union[list, Generator]]
            :param weights2: the list of weights on each value of c2
            :type weights2: Optional[Union[list, Generator]]

            Returns
            -----------------------------------
            :return: how closely (0 is best) the distribution over values in set
                c2 is to uniform
            :rtype: float 
        '''
        return self.set1_w * self.stat(c1, weights1) + self.set2_w * self.stat(c2, weights2)

    def stat(self,
        c1: Union[list, Generator],
        weights1: Optional[Union[list, Generator]],
    ) -> float:
        '''
            Summary:
                Esimates the empirical (categorical) distribution of the values
                in c1. Unseen elements are automatically smoothed by giving
                those categories a count of self.smooth.

            Inputs
            ------------------------
            :param c1: the list of values over which to estimate the categorical
                distribution.
            :type c1: Union[list, Generator]
            :param weights1: the list of weights on each value of c1
            :type weights1: Optional[Union[list, Generator]]

            Returns
            -------------------
            :return: The statistic, (weighted) mean value, on c1
            :rtype: float 
        '''
        c1 = sorted(c1)
        weights1 = list(weights1) if weights1 is not None else weights1
        iterator = zip(weights1, c1) if weights1 is not None else zip([1.0]*len(c1), c1)
        c1_counts = {}
        c1_total = 0.
        for w, item in iterator:
            item_ = self.reduce(item, weights=w)
            for i, w_ in item_:
                # add it with count of self.smooth to both c1 and c2 counts
                c1_counts[i] = c1_counts.get(i, 0.0) + w_
                c1_total += w_
        c1_dist = np.array(
            [v for k, v in sorted(c1_counts.items(), key=lambda x: x[0])]
        ) / c1_total
        return 1.0 + np.dot(c1_dist, np.log(c1_dist)) / np.log(len(c1_dist)) 

