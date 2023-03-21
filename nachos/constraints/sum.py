from nachos.constraints.abstract_constraint import AbstractConstraint
from nachos.constraints import register
from typing import Union, Generator


@register('sum')
class Sum(AbstractConstraint):
    '''
        Summary:
            Defines the constraint on the mean value of a factor. The constraint  
            is that the mean for two datasets should be close to a specified
            value. 
    '''
    @classmethod
    def build(cls, conf: dict):
        return cls(*conf['sum_tuple'])

    def __init__(self, s1_sum: float, s2_sum: float):
        super().__init__()
        self.s1_sum = s1_sum
        self.s2_sum = s2_sum

    def __call__(self,
        c1: Union[list, Generator],
        c2: Union[list, Generator],
    ) -> float:
        '''
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
            
            Returns
            -----------------------
            :return: the constraint score (how close the constraints are met)
            :rtype: float 
        '''
        return abs(self.stat(c1) - self.s1_sum) + abs(self.stat(c2) - self.s2_sum)

    def stat(self, c1: Union[List, Generator]) -> float:
        '''
            Summary:
                computes the sum of the values in c1.

            Inputs
            ------------------
            :param c1: the list of values over which to compute the sum 
        '''
        c1 = list(c1)
        # for multivalued problems, average values in c1
        return float(sum(sum(c) / len(c) for c in c1))
