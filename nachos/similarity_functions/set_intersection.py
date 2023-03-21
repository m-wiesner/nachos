from typing import Union, Any, Iterable, Hashable
from nachos.similarity_functions.abstract_similarity import AbstractSimilarity
from nachos.similarity_functions import register
from nachos.utils import check_iterable


@register('set_intersection')
class SetIntersection(AbstractSimilarity):
    @classmethod
    def build(cls, conf: dict):
        return cls()

    def __call__(self, f: Any, g: Any) -> float:
        '''
            Summary:
                Computes the similarity between inputs f and g. f, g are assumed to
                be multi-valued objects, i.e., represent sets of values. We use the
                size of the intersection of the elements as the similiarity.
            Inputs
            -------------------------------
            :param f: a value to compare
            :type f: Union[Any, Iterable]. I.e., a set or something which can be
                converted to a set
            :param g: a value to compare
            :type g: Union[Any, Iterable]. I.e., a set or something which can be
                converted to a set
    
            Returns
            -----------------------------------
            :return: returns the similarity score
            :rtype: float
        '''
        # Both objects are just values --> Use Boolean comparison
        if not check_iterable(f) and not check_iterable(g):
            return float(f == g)
        # One object is a value the other is a set --> convert value to
        # iterable, then set and then compute size of set intersection
        elif check_iterable(f) and not check_iterable(g):
            g = [g]
        elif not check_iterable(f) and check_iterable(g):
            f = [f]
        # At this point we know that f, g are both iterable. If their elements
        # are hashable, convert the iterable objects to sets.
        elif isinstance(next(iter(f)), Hashable) and not isinstance(next(iter(g)), Hashable):     
            f = set(f)
        elif not isinstance(next(iter(f)), Hashable) and isinstance(next(iter(g)), Hashable):
            g = set(g)
        elif isinstance(next(iter(f)), Hashable) and isinstance(next(iter(g)), Hashable):
            f, g = set(f), set(g)
        # At this point, we know the elements of f, and g are not hashable. We
        # assume they are iterable, flatten them and convert them to sets.
        elif isinstance(next(iter(f)), Iterable) and isinstance(next(iter(g)), Iterable):
            f = set().union(*f) 
            g = set().union(*g)
        return float(len(f.intersection(g)))  
