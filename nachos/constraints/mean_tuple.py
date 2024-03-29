from nachos.constraints.mean import Mean
from nachos.constraints import register
from typing import Any, Generator, Union, Optional


@register('mean_tuple')
class MeanTuple(Mean):
    '''
        Summary:
            Defines the constraint on the mean value of a factor. The constraint
            is that the mean for two datasets should be close to a specified
            value.
    '''
    @classmethod
    def build(cls, conf: dict):
        return cls(*conf['values'], reduction=conf['reduction'])

    def __init__(self, s1_mean: Any, s2_mean: Any, reduction: str = 'mean'):
        super().__init__(reduction=reduction)
        self.s1_mean = s1_mean
        self.s2_mean = s2_mean

    def __call__(self,
        c1: Union[list, Generator],
        c2: Union[list, Generator],
        weights1: Optional[Union[list, Generator]] = None,
        weights2: Optional[Union[list, Generator]] = None,
    ) -> float:
        r'''
            Summary:
                Given a tuple

                .. math::
                    \mu = \left(\mu_1, \mu_2\right)

                compute

                .. math::
                    \lvert \frac{1}{|c1|} \sum c1 - \mu_1 \rvert + \lvert \frac{1}{|c2|} \sum c2 - \mu_2 \rvert

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
        return abs(self.stat(c1, weights1) - self.s1_mean) + abs(self.stat(c2, weights2) - self.s2_mean)
