from typing import Union, Generator
from abc import ABC, abstractmethod


class AbstractConstraint(ABC):
    @classmethod
    @abstractmethod
    def build(cls, conf: dict):
        raise NotImplementedError

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self,
        c1: Union[list, Generator],
        c2: Union[list, Generator]
    ) -> float:
        raise NotImplementedError 
