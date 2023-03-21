from typing import Any
from abc import ABC, abstractmethod


class AbstractSimilarity(ABC):
    @classmethod
    @abstractmethod
    def build(cls, conf: dict):
        raise NotImplementedError

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, f: Any, g: Any) -> float:
        raise NotImplementedError 
