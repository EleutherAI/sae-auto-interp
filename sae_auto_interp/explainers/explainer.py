from dataclasses import dataclass

from ..features.features import Example, FeatureRecord  
from typing import List
import asyncio

from abc import ABC, abstractmethod

import asyncio
import orjson
import os
import aiofiles
from typing import List, Callable, Awaitable


@dataclass
class ExplainerInput:
    train_examples: List[Example]
    record: FeatureRecord 

@dataclass
class ExplainerResult:
    explainer_type: str = ""
    input: str = ""
    response: str = ""
    explanation: str = ""

class Explainer(ABC):

    @abstractmethod
    def __call__(
        self,
        explainer_in: ExplainerInput
    ) -> ExplainerResult:
        pass