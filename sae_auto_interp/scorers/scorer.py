from abc import ABC, abstractmethod

from ..features.features import FeatureRecord

class Scorer(ABC):
    
    @abstractmethod
    def __call__(
        self,
        record: FeatureRecord
    ):
        pass