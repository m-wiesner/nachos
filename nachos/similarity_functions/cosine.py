from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nachos.similarity_functions.abstract_similarity import AbstractSimilarity
from nachos.similarity_functions import register


@register('cosine')
class Cosine(AbstractSimilarity):
    '''
        Summary:
            Defines the (thresholded) cosine similarity between two points.
            Each points are expected to be ndarrays. The cosine similarity
            is computed using the sklearn pairwise metrics package. If all
            pairwise distances are desired, then the ndarray can be Nxd, where
            N specifies the number of data points. 

            Using N > 1 is useful when defining similarities on sets, which 
            this similarity function is automatically designed to do. It
            returns the largest pairwise similarity between any elements of the
            sets being compared.
    '''
    @classmethod
    def build(cls, conf: dict):
        return cls(conf['cosine_thresh'])

    def __init__(self, t: float):
        super().__init__()
        self.thresh = t

    def __call__(self, f: set, g: set) -> float:
        '''
            Summary:
                Computes the thresholded cosine similarity between inputs f, g.
                f, g are assumed to be real valued vectors, generally representing
                embeddings which have been whitened.
            Inputs  
            ---------------------------------------
            :param f: an ndarray representing a set of vectors to compare
            :type f: set
            :param g: an ndarray representing a set of vectors to compare
            :type g: set
            
            Returns
            --------------------------------------
            :return: returns the similarity score
            :rtype: float
        '''
        # cast sets of arrays to ndarray 
        f, g = np.array(list(f)), np.array(list(g))
        
        # In the case of a set of vectors being compared to a single vector, or
        # another set, we use \sup_{x, y} d(x, y)
        sim = cosine_similarity(f, g).max()
        return float(sim if sim > self.thresh else 0)
