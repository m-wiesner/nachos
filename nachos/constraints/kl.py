from typing import Union, Generator
from nachos.constraints.abstract_constraint import AbstractConstraint
from nachos.constraints import register
from itertools import chain
import numpy as np



@register('kl')
class KL(AbstractConstraint):
    '''
        Summary:
            Defines the constraint on the categorical distribution over values
            between two datasets. The cost of mismatch is computed as the 
            kl-divergence between two sets. In general, the smaller
            set is the test set and we would like it to have specific
            characteristics w/r to the large (training) set. The forward kl,
            i.e.,
            
            The forward KL, i.e., 
            .. math::
                kl\left(p \vert\vert q_\theta\right) 
            
            is mean seeking  

            cost = KL(d1_train || d2_test)

            This will encourge selecting data with good coverage of the dataset,
            including data points that may have been seen only occasionally in
            the training data. See ReverseKL, Jeffrys for more information. 

            Reverse KL is 
            .. math::
                kl\left(q_\theta \vert\vert p\right)
            
            cost = KL(d2_test || d1_train)

            This encourages mode seeking behavior.

            The Jeffry's divergence symmetrizes the KL divergence as
            .. math::
                \frac{1}{2}\left[KL\left(p \vert\vert q_\theta\right) + KL\left(q_\theta \vert\vert p\right)\right]

    '''
    @classmethod
    def build(cls, conf: dict):
        return cls(smooth=conf['kl_smooth'], direction=conf['kl_direction'])

    def __init__(self, smooth: float = 0.000001, direction: str = 'forward'):
        super().__init__()
        self.smooth = smooth
        self.direction = direction 
        self.vocab = None

    def __call__(self,
        c1: Union[list, Generator],
        c2: Union[list, Generator],
    ) -> float:
        '''
            Summary:
                Computes the KL divergence between the empircal distributions
                defined by values in c1 and values in c2.
            
            Inputs
            ---------------------------
            :param c1: the values to constrain seen in dataset 1
            :type c1: Union[list, Generator]
            :param c2: the values to constrain seen in dataset 2
            :type c2: Union[list, Generator]

            Returns
            ---------------------------
            :return: how closely (0 is best) the sets c1, c2 satisfy the constraint
            :rtype: float
        '''
        # Get vocab. In general, c1 and c2 should be lists of sets
        if self.vocab is None:
            vocab = set()
            for item in chain(c1, c2):
                try:
                    for i in item:
                        vocab.add(i)
                except TypeError:
                    vocab.add(item)
            self.vocab = vocab

        # Get counts (i.e., distributions) for set1 and set2
        c1_counts = {v: self.smooth for v in vocab}
        c2_counts = {v: self.smooth for v in vocab}
        c1_total = self.smooth * len(vocab)
        c2_total = self.smooth * len(vocab) 
        for item in c1:
            try:
                for i in item:
                    # if i wasn't seen in the vocab,
                    # add it with count of self.smooth to both c1 and c2 counts
                    if i not in self.vocab:
                        self.ocab.add(i)
                        c1_total += self.smooth
                        c2_total += self.smooth 
                        c1_counts[i] = self.smooth
                        c2_counts[i] = self.smooth
                    c1_counts[i] += 1
            except TypeError:
                if item not in self.vocab:
                    self.vocab.add(item)
                    c1_total += self.smooth
                    c2_total += self.smooth
                    c1_counts[item] = self.smooth
                    c2_counts[item] = self.smooth
                c1_counts[item] += 1  
        for item in c2:
            try:
                for i in item:
                    if i not in self.vocab:
                        self.vocab.add(i)
                        c2_total += self.smooth
                        c1_total += self.smooth
                        c2_counts[i] = self.smooth
                        c1_counts[i] = self.smooth
                    c2_counts[i] += 1
            except TypeError:
                if item not in self.vocab:
                    self.vocab.add(item)
                    c2_total += self.smooth
                    c1_total += self.smooth
                    c2_counts[item] = self.smooth
                    c1_counts[item] = self.smooth
                c2_counts[item] += 1
       
        # Normalize each count by the total count to get a distribution 
        c1_dist = np.array(
            [v for k, v in sorted(c1_counts.items(), key=lambda x: x[0])]
        ) / c1_total
        c2_dist = np.array(
            [v for k, v in sorted(c2_counts.items(), key=lambda x: x[0])]
        ) / c2_total
    
        # Return the appropriate direction kl
        if self.direction == "forward":
            return np.dot(c1_dist, np.log(c1_dist) - np.log(c2_dist))
        if self.direction == "reverse":
            return np.dot(c2_dist, np.log(c2_dist) - np.log(c1_dist))
        if self.direction == "symmetric":
            return 0.5 * (
                np.dot(c1_dist, np.log(c1_dist) - np.log(c2_dist)) + 
                np.dot(c2_dist, np.log(c2_dist) - np.log(c2_dist))
            )
    
