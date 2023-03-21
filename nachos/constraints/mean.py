from nachos.constraints.abstract_constraint import AbstractConstraint
from nachos.constraints import register
from typing import Union, Generator


@register('mean')
class Mean(AbstractConstraint):
    '''
        Summary:
            Defines a constraint on the mean value of a factor. The constraint
            is that the mean between two Datasets (defined by the Dataset class)
            should be the same. This class just computes the difference between
            the means and returns that as a float. Instead of working with the
            Dataset class directly, this class works on the constraint values
            in that class.
    '''
    @classmethod
    def build(cls, conf: dict):
        return cls()

    def __call__(self,
        c1: Union[list, Generator],
        c2: Union[list, Generator],
    ) -> float:
        '''
            Summary:
                Computes 
                
                .. math::
                    \lvert \frac{1}{|c1|} \sum c1 - \frac{1}{|c2|} \sum c2 \rvert
            
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
        return abs(self.stat(c1) - self.stat(c2))
    
    def stat(self, c1: Union[list, Generator]) -> float:
        '''
            Summary:
                computes the mean of the values in c1.

            Inputs
            ------------------
            :param c1: the list of values over which to compute the mean 
        '''
        c1 = list(c1)
        # for multivalued problems, average values in c1
        return float(sum(sum(c) / len(c) for c in c1)) / len(c1)
