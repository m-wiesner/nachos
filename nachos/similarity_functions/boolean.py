from typing import Any
from nachos.similarity_functions.abstract_similarity import AbstractSimilarity
from nachos.similarity_functions import register


@register('boolean')
class Boolean(AbstractSimilarity):
    '''
        Summary:
            This class defines the boolean similarity between points. It
            assumes the points are categorical. It is overloaded to allow for
            sets of inputs, in which case the similarity (True or False) is
            decided by examining whether any element in the set is equal to 
            any element in the other set. 
    '''
    @classmethod
    def build(cls, conf: dict):
        return cls()
    
    def __call__(self, f: Any, g: Any) -> bool:
        '''
            Summary:
                Computes the similarity bewtween f and g. Similarity is binary
                a binary value. f, g can be any object though the intention is for
                them to be categorical values that can be compared for equality.
            Inputs
            ----------------------------------
            :param f: a value (categorical) to be compared
            :param g: a value (categorical) to be compared
            
            Returns
            ---------------------------------
            :return: the boolean similarity between f and g
            :rtype: bool
        '''
        return float(f == g)
