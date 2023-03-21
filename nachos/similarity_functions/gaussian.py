import numpy as np
from nachos.similarity_functions.abstract_similarity import AbstractSimilarity
from nachos.similarity_functions import register


@register('gaussian')
class Gaussian(AbstractSimilarity):
    @classmethod
    def build(cls, conf: dict):
        return cls(conf['Gaussian_thresh'])

    def __init__(self, t: float):
        super().__init__()
        self.thresh = t

    def __call__(self, f: set, g: set) -> float:
        '''
            Summary:
                Computes the thresholded similarity score between inputs f, g.
                f, g are assumed to be real valued scalars, and the similarity is
                the Gaussian similarity between the values assuming unit variance.
            Inputs
            ------------------------------------
            :param f: a float representing a real value to compare
            :type f: float
            :param g: a float representing a real value to compare
            :type g: float

            Returns
            --------------------------------------
            :return: returns the similarity score
            :rtype: float
        '''
        # If there is more than one real value in f, g, we just take the average
        sim = np.exp(-(float(sum(f)/len(g)) - float(sum(g)/len(g)))**2)
        return float(sim if sim > self.thresh else 0)
