from typing import Union, Generator
from abc import ABC, abstractmethod


class AbstractConstraint(ABC):
    @classmethod
    def build(cls, conf: dict):
        return cls(reduction=conf['constraint_reduction'])

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
