from typing import Union, Generator
from abc import ABC, abstractmethod


class AbstractConstraint(ABC):
    '''
        Summary:
            Defines the abstract base class for all constraints used for
            splitting datasets. By default, values for constraints can be
            multivalued. This can occur for instance when there are two speakers
            speaking in a single recording with different genders. Or if there
            are multiple durations of speech specified for each speaker
            in a recording. 
            
            Some constraint methods can work directly on the
            sets, while other require a specific reduction method. This
            reduction method is the only parameter passed to the base class.
            Other constraints may require specifying additional parameters.
    '''
    @classmethod
    def build(cls, conf: dict):
        return cls(reduction=conf['reduction'])

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    @abstractmethod
    def __call__(self,
        c1: Union[list, Generator],
        c2: Union[list, Generator]
    ) -> float:
        raise NotImplementedError
    
    def reduce(self, s: set) -> float:
        '''
            Summary:
                When a constraint is set valued, how do we reduce the
                constraint to be a single float? The reduction method
                specified determines this.

                Values for reduction can be:
                    - 'mean'
                    - 'sum'
                    - 'min'
                    - 'max'

                They are specified in the constraint_reduction field of the 
                yaml configuration file passed to the build method of this
                class. 
            
            :param s: The set of floats to be reduced
            :type s: set

            :return: the reduced valued
            :rtype: float
        '''
        if self.reduction == 'mean':
            return sum(s) / len(s)
        if self.reduction == 'sum': 
            return sum(s)
        if self.reduction == 'min':
            return min(s)
        if self.reduction == 'max':
            return max(s)
