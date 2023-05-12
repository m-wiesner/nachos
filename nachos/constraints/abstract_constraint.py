from typing import Union, Generator, Optional
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
        c2: Union[list, Generator],
        weights1: Optional[Union[list, Generator]] = None,
        weights2: Optional[Union[list, Generator]] = None,
    ) -> float:
        raise NotImplementedError
    
    def reduce(self, s: Union[set,list], weights: Optional[list] = None) -> float:
        '''
            Summary:
                When a constraint is set valued, how do we reduce the
                constraint to be a single float? The reduction method
                specified determines this. Sometimes, we combine multiple
                records, for instance when creating a new record from all
                records belonging to the same connected component. In these
                cases, we may have a list of sets, each with a different
                possible weight. We may want to do a weighted reduction
                which is why we pass the weights argument around.

                Values for reduction can be:
                    - 'mean'
                    - 'sum'
                    - 'min'
                    - 'max'
                    - 'concat'

                They are specified in the constraint_reduction field of the 
                yaml configuration file passed to the build method of this
                class. 'concat', is useful when we have multivalued constraints
                that we have combined and want to use the kl-divergence
                constraint. Each set needs to be combined with a 
                different weight, so a tuple is created for each value with the
                associated weight.
            
            :param s: The set, or list of sets of values to be reduced
            :type s: set
            :param weights: the weights to apply during reduction
            :type weights: list[float]

            :return: the reduced valued
            :rtype: float
        '''
        if self.reduction == 'mean':
            if weights is not None:
                if isinstance(s, list):
                    return sum(w*sum(s_) for s_, w in zip(s, weights)) / sum(weights)
                elif isinstance(s, set):
                    return sum(w*s_ for s_, w in zip(s, weights)) / sum(weights)
            else:
                if isinstance(s, list):
                    return sum(sum(s_) for s_ in s) / len(s)
                elif isinstance(s, set):
                    return sum(s) / len(s)
        if self.reduction == 'sum': 
            if weights is not None:
                if isinstance(s, list):
                    return sum(w*sum(s_) for s_, w in zip(s, weights))   
                elif isinstance(s, set):
                    return sum(w*s_ for s_, w in zip(s, weights))
            else:
                if isinstance(s, list):
                    return sum(sum(s_) for s_ in s)
                elif isinstance(s, set):
                    return sum(s)
        if self.reduction == 'min':
            multiplier = sum(weights) if weights is not None else 1.0
            if isinstance(s, list):
                return multiplier * min([min(s_) for s_ in s])
            elif isinstance(s, set):
                return multiplier * min(s)
        if self.reduction == 'max':
            multiplier = sum(weights) if weights is not None else 1.0
            if isinstance(s, list):
                return multiplier * max([max(s_) for s_ in s])
            elif isinstance(s, set):
                return multiplier * max(s)
        if self.reduction == 'concat':
            if weights is not None:
                if isinstance(s, list):
                    return [(i, w) for s_, w in zip(s, weights) for i in s_]
                elif isinstance(s, set):
                    assert len(weights) == 1, f"Length of {weights} was {len(weights)}, but expected 1."
                    return [(s_, weights[0]) for s_ in s]
            else:
                if isinstance(s, list):
                    return [(i, 1.0) for s_ in s for i in s_]
                elif isinstance(s, set):
                    return [(s_, 1.0) for s_ in s]
        if self.reduction == 'none':
            return s
