from typing import Union, Generator, Optional
from nachos.constraints.kl import KL
from nachos.constraints import register
from typing import Any


@register('kl_tuple')
class KLTuple(KL):
    r'''
        Summary:
            Defines the constraint on the categorical distribution over values
            in two datasets. The cost of mismatched is computed as the sum 
            of the kl-divergense between each set and a specified categorical
            distributions.

            .. math::
                cost = KL\left(p \vert\vert q_{d1} \right) + KL\left(p \vert\vert q_{d2}\right)

            where

            .. math::
                q_{d1}, q_{d2}

            are the learned distributions. Users can also specify to use the
            forward, reverse, or symmetric (jeffry's) variants. See 
            
                nachos/constraints/kl.py

            for more information.

            To match a specific distribution means that we assume a vocabulary
            is available ahead of time. Therefore, this constraints cannot
            work if a vocabulary is not supplied.

            The distribution should be specified in the configuration dictionary
            as a tuple (or list) of lists of values specifying the desired
            marginal probability of each category in the same order as vocab.
    '''
    @classmethod
    def build(cls, conf:dict):
            return cls(*conf['values']['dist'],
            smooth=conf['values']['smooth'],
            direction=conf['values']['direction'],
            vocab=conf['values']['vocab'],
        )

    def __init__(self, 
        s1_dist: Any,
        s2_dist: Any,
        smooth: float = 0.000001,
        direction: str = 'forward',
        vocab: Optional[list] = None,
    ):
        assert vocab is not None, "Vocabulary must be set for kl_tuple."
        super().__init__(smooth=smooth, direction=direction, vocab=vocab)
        self.s1_dist = s1_dist
        self.s2_dist = s2_dist
    
    def __call__(self,
        c1: Union[list, Generator],
        c2: Union[list, Generator],
        weights1: Optional[Union[list, Generator]] = None,
        weights2: Optional[Union[list, Generator]] = None,
    ) -> float:
        '''
            Summary:
                Given a tuple

                .. math::
                    \mu = \left(p_1, p_2\right)

                compute
                
                .. math::
                    KL\left(p1 \vert\vert d1\right) + KL\left(p2 \vert\vert d2\right)

                where

                .. math::
                    d1, d2

                are the empirical densities generated from c1 and c2.
            

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
        c1_dist = self.stat(c1, weights1)
        c2_dist = self.stat(c2, weights2)

        if self.direction == "forward":
            return (
                np.dot(self.s1_dist, np.log(self.s1_dist) - np.log(c1_dist))
                + np.dot(self.s2_dist, np.log(self.s2_dist) - np.log(c2_dist))
            )
        if self.direction == "reverse":
            return (
                np.dot(c1_dist, np.log(c1_dist) - np.log(self.s1_dist))
                + np.dot(c2_dist, np.log(c2_dist) - np.log(self.s2_dist))
            )
        if self.direction == "symmetric":
            return (
                0.5 * (
                    np.dot(self.s1_dist, np.log(self.s1_dist) - np.log(c1_dist)) +
                    np.dot(c1_dist, np.log(c1_dist) - np.log(self.s1_dist))
                )
                + 0.5 * (
                    np.dot(self.s2_dist, np.log(self.s2_dist) - np.log(c2_dist)) +
                    np.dot(c2_dist, np.log(c2_dist) - np.log(self.s2_dist))
                )
            )
        raise ValueError(f"An invalid direction {self.direction} was likely"
            f" used. Please choose from ['forward', 'reverse', 'symmetric'"
        )
