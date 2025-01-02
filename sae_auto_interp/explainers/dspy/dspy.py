import re
from ..explainer import Explainer, ExplainerResult
from ...logger import logger
from typing import Optional
from dspy import LM
from ...features.features import FeatureRecord


class DSPyExplainer(Explainer):
    name = "default"

    def __init__(self,
                 client: Optional[LM],
                 tokenizer,
                 **generation_kwargs,
                 ):
        self.client = client
        self.tokenizer = tokenizer
        self.generation_kwargs = generation_kwargs

    def __call__(self, record: FeatureRecord):
        pass