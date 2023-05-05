from nachos.constraints.abstract_constraint import AbstractConstraint
from nachos.constraints import register
from typing import Union, Generator, Optional


@register('sum')
class Sum(AbstractConstraint):
    '''
        Summary:
            Defines the constraint on the mean value of a factor. The constraint
            is that the mean for two datasets should be close to a specified
            value.
    '''
    def __call__(self,
        c1: Union[list, Generator],
        c2: Union[list, Generator],
        weights1: Optional[Union[list, Generator]] = None,
        weights2: Optional[Union[list, Generator]] = None,
    ) -> float:
        r'''
            Summary:
                Computes

                .. math::
                    \lvert \sum c1 - \sum c2 \rvert

            Inputs
            -----------------------
            :param c1: the list of values to constrain associated with dataset 1
            :type c1: Union[list, Generator]
            :param c2: the list of values to constrain associated with dataset 2
            :type c2: Union[list, Generator]
            :param weights1: the list of weights on each value of c1
            :type weights1: Optional[Union[list, Generator]]
            :param weights2: the list of weights on each value of c2
            :type weights2: Optional[Union[list, Generator]]


            Returns
            -----------------------
            :return: the constraint score (how close the constraints are met)
            :rtype: float
        '''
        c1, c2 = list(c1), list(c2)
        weights1, weights2 = list(weights1), list(weights2)
        return abs(self.stat(c1, weights1) - self.stat(c2, weights2))

    def stat(self,
        c1: Union[list, Generator],
        weights1: Optional[Union[list, Generator]],
    ) -> float:
        '''
            Summary:
                computes the sum of the values in c1.

            Inputs
            ------------------
            :param c1: the list of values over which to compute the sum
            :type c1: Union[list, Generator]
            :param weights1: the list of weights on each value of c1
            :type weights1: Optional[Union[list, Generator]]

            Returns
            -------------------
            :return: The statistic, (weighted) sum, of c1
            :rtype: float 
        '''
        c1 = list(c1)
        # for multivalued problems, reduce values in c1
        if weights1 is not None:
            weights1 = list(weights1)
            return float(sum(self.reduce(c, weights=w) for w, c in zip(weights1, c1)))
        else:
            return float(sum(self.reduce(c) for c in c1))
