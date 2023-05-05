from typing import Union, Generator, Optional
from nachos.constraints.abstract_constraint import AbstractConstraint
from nachos.constraints import register
from itertools import chain
import numpy as np


@register('kl')
class KL(AbstractConstraint):
    r'''
        Summary:
            Defines the constraint on the categorical distribution over values
            between two datasets. The cost of mismatch is computed as the
            kl-divergence between two sets. In general, the smaller
            set is the test set and we would like it to have specific
            characteristics w/r to the large (training) set. The forward kl,
            i.e.,

            The forward KL, i.e.,

            .. math::
                KL\left(p \vert\vert q_\theta\right)

            is mean seeking

            cost = KL(d1_train || d2_test)

            This will encourge selecting data with good coverage of the dataset,
            including data points that may have been seen only occasionally in
            the training data. See ReverseKL, Jeffrys for more information.

            Reverse KL is

            .. math::
                KL\left(q_\theta \vert\vert p\right)

            cost = KL(d2_test || d1_train)

            This encourages mode seeking behavior.

            The Jeffry's divergence symmetrizes the KL divergence as

            .. math::
                \frac{1}{2}\left[KL\left(p \vert\vert q_\theta\right) + KL\left(q_\theta \vert\vert p\right)\right]

    '''
    @classmethod
    def build(cls, conf: dict):
        if 'vocab' in conf['values']:
            return cls(
                smooth=conf['values']['smooth'],
                direction=conf['values']['direction'],
                vocab=conf['values']['vocab'],
            )
        else:
            return cls(
                smooth=conf['values']['smooth'],
                direction=conf['values']['direction'],
            )

    def __init__(self,
        smooth: float = 0.000001,
        direction: str = 'forward',
        vocab: Optional[list] = None,
    ):
        super().__init__()
        self.smooth = smooth
        self.direction = direction
        self.vocab = set()
        if vocab is not None:
            self.set_vocab(vocab)

    def __call__(self,
        c1: Union[list, Generator],
        c2: Union[list, Generator],
        weights1: Optional[Union[list, Generator]] = None,
        weights2: Optional[Union[list, Generator]] = None,
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
            :param weights1: the list of weights on each value of c1
            :type weights1: Optional[Union[list, Generator]]
            :param weights2: the list of weights on each value of c2
            :type weights2: Optional[Union[list, Generator]]


            Returns
            ---------------------------
            :return: how closely (0 is best) the sets c1, c2 satisfy the constraint
            :rtype: float
        '''
        # Get counts (i.e., distributions) for set1 and set2. We build the
        # vocabulary organically as it is seen. If there are sets at the
        # beginning that don't cover the vocabulary, we might underestimate the
        # true KL-divergence, and these values are not technically comparable
        # accross iterations.
        c1 = list(c1)
        c2 = list(c2)
        weights1 = list(weights1) if weights1 is not None else weights1
        weights2 = list(weights2) if weights2 is not None else weights2
        c1_counts = {v: self.smooth for v in self.vocab}
        c2_counts = {v: self.smooth for v in self.vocab}
        c1_total = self.smooth * len(self.vocab)
        c2_total = self.smooth * len(self.vocab)
        iterator1 = zip(weights1, c1) if weights1 is not None else zip([1.0]*len(c1), c1)
        iterator2 = zip(weights2, c2) if weights2 is not None else zip([1.0]*len(c2), c2)
        for w, item in iterator1:
            try:
                for i in item:
                    # if i wasn't seen in the vocab,
                    # add it with count of self.smooth to both c1 and c2 counts
                    if i not in self.vocab:
                        self.vocab.add(i)
                        c1_total += self.smooth
                        c2_total += self.smooth
                        c1_counts[i] = self.smooth
                        c2_counts[i] = self.smooth
                    c1_counts[i] += w
                    c1_total += w
            except TypeError:
                if item not in self.vocab:
                    self.vocab.add(item)
                    c1_total += self.smooth
                    c2_total += self.smooth
                    c1_counts[item] = self.smooth
                    c2_counts[item] = self.smooth
                c1_counts[item] += w
                c1_total += w
        for w, item in iterator2:
            try:
                for i in item:
                    if i not in self.vocab:
                        self.vocab.add(i)
                        c2_total += self.smooth
                        c1_total += self.smooth
                        c2_counts[i] = self.smooth
                        c1_counts[i] = self.smooth
                    c2_counts[i] += w
                    c2_total += w
            except TypeError:
                if item not in self.vocab:
                    self.vocab.add(item)
                    c2_total += self.smooth
                    c1_total += self.smooth
                    c2_counts[item] = self.smooth
                    c1_counts[item] = self.smooth
                c2_counts[item] += w
                c2_total += w

        # Normalize each count by the total count to get a distribution
        c1_dist = np.array(
            [v for k, v in sorted(c1_counts.items(), key=lambda x: x[0])]
        ) / c1_total
        c2_dist = np.array(
            [v for k, v in sorted(c2_counts.items(), key=lambda x: x[0])]
        ) / c2_total

        #c1_dist = self.stat(c1)
        #c2_dist = self.stat(c2)
        # Return the appropriate direction kl
        if self.direction == "forward":
            return np.dot(c1_dist, np.log(c1_dist) - np.log(c2_dist))
        if self.direction == "reverse":
            return np.dot(c2_dist, np.log(c2_dist) - np.log(c1_dist))
        if self.direction == "symmetric":
            return 0.5 * (
                np.dot(c1_dist, np.log(c1_dist) - np.log(c2_dist)) +
                np.dot(c2_dist, np.log(c2_dist) - np.log(c1_dist))
            )
        raise ValueError(f"An invalid direction {self.direction} was likely"
            f" used. Please choose from ['forward', 'reverse', 'symmetric'"
        )

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
        c1 = list(c1)
        weights1 = list(weights1) if weights1 is not None else weights1
        c1_counts = {v: self.smooth for v in self.vocab}
        c1_total = self.smooth * len(self.vocab)
        iterator = zip(weights1, c1) if weights1 is not None else zip([1.0]*len(c1), c1)
        for w, item in iterator:
            try:
                for i in item:
                    # if i wasn't seen in the vocab,
                    # add it with count of self.smooth to both c1 and c2 counts
                    if i not in self.vocab:
                        self.vocab.add(i)
                        c1_total += self.smooth
                        c1_counts[i] = self.smooth
                    c1_counts[i] += w
                    c1_total += w
            except TypeError:
                if item not in self.vocab:
                    self.vocab.add(item)
                    c1_total += self.smooth
                    c1_counts[item] = self.smooth
                c1_counts[item] += w
                c1_total += w
        c1_dist = np.array(
            [v for k, v in sorted(c1_counts.items(), key=lambda x: x[0])]
        ) / c1_total
        return c1_dist

    def set_vocab(self, vocab: Union[set, list]):
        '''
            Summary:
                Set the domain (vocab) used when computing KL-divergence

            Inputs
            -----------------
            :param vocab: the domain
            :type vocab: Union[set, list] 
        '''
        if isinstance(vocab, set):
            self.vocab = vocab
        elif isinstance(vocab, list):
            self.vocab = set(vocab)
