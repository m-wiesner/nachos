from nachos.constraints.sum import Sum
from nachos.constraints import register
from typing import Union, Generator, Optional


@register('sum_tuple')
class SumTuple(Sum):
    '''
        Summary:
            Defines the constraint on the sum of a factor. The constraint
            is that the sum for two datasets should be close to a specified
            value. The cost of deviating is the absolute difference
            between the actual and desired sum raised to a power (specified).
            The sum of these terms for both splits is the final cost.
    '''
    @classmethod
    def build(cls, conf: dict):
        '''
            Summary:
                :param conf: A dictionary containing any necessary parameters,
                    most of which are passed through a yaml configurations file.
                    This method assumes that conf has the following structure.
                    conf = {values: {values: [m1, m2], power: 2.0}, reduction='min'}
        '''
        return cls(*conf['values']['values'],
            power=conf['values']['power'],
            reduction=conf['reduction']
        )

    def __init__(self, s1_sum: float, s2_sum: float,
        reduction: str = 'mean',
        power: float = 1.0
    ):
        super().__init__(reduction=reduction)
        self.s1_sum = s1_sum
        self.s2_sum = s2_sum
        self.power = power

    def __call__(self,
        c1: Union[list, Generator],
        c2: Union[list, Generator],
        weights1: Optional[Union[list, Generator]],
        weights2: Optional[Union[list, Generator]],
    ) -> float:
        r'''
            Summary:
                Computes

                .. math::
                    \lvert \sum c1 - \mu_1\rvert^p + \lvert \sum c2 - \mu_2 \rvert^p

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
        return (
            abs(self.stat(c1, weights1) - self.s1_sum)**self.power
          + abs(self.stat(c2, weights2) - self.s2_sum)**self.power
        )
