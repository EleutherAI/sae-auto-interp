import re
from ..explainer import Explainer, ExplainerResult
from ...logger import logger
from typing import Optional, List
from pydantic import BaseModel
import dspy
from ...features.features import FeatureRecord


class DSPyExplainer(Explainer):
    name = "default"

    def __init__(self,
                 client: Optional[dspy.LM],
                 tokenizer,
                 **generation_kwargs,
                 ):
        self.client = client
        self.tokenizer = tokenizer
        self.generation_kwargs = generation_kwargs
        self.explainer = dspy.ChainOfThought(Explanations)

    async def __call__(self, record: FeatureRecord):
        tokenizer = self.tokenizer
        records = record.train
        records = [
            FeatureExample(
                text=tokenizer.decode(ex.tokens),
                max_activation=max(ex.activations.tolist()),
                tokens=tokenizer.batch_decode(ex.tokens),
                tokens_and_activations=list(zip(tokenizer.batch_decode(ex.tokens), ex.activations.tolist()))
            )
            for ex in records
        ]
        result = self.explainer(feature_examples=records, lm=self.client)
        return ExplainerResult(
            record,
            result.explanation
        )


class FeatureExample(BaseModel):
    """A text snippet that is an example that activates a feature."""
    
    text: str
    max_activation: float
    tokens: list[str]
    tokens_and_activations: list[tuple[str, float]]


class Explanations(dspy.Signature):
    """
    You are a meticulous AI researcher conducting an important investigation into patterns found in language. Your task is to analyze text and provide an explanation that thoroughly encapsulates possible patterns found in it.
    Guidelines:

    You will be given a list of text examples on which special words are selected and between delimiters like <<this>>. If a sequence of consecutive tokens all are important, the entire sequence of tokens will be contained between delimiters <<just like this>>. How important each token is for the behavior is listed after each example in parentheses.

    - Try to produce a concise final description. Simply describe the text features that are common in the examples, and what patterns you found.
    - If the examples are uninformative, you don't need to mention them. Don't focus on giving examples of important tokens, but try to summarize the patterns found in the examples.
    - Do not mention the marker tokens (<< >>) in your explanation.
    """

    feature_examples: List[FeatureExample] = dspy.InputField(desc="A list of examples that activate the feature.")
    special_words: List[str] = dspy.OutputField(desc="Find the special words that are selected in the examples and list a couple of them. Search for patterns in these words, if there are any. Don't list more than 5 words.")
    shared_features: str = dspy.OutputField(desc="Write down general shared features of the text examples. This could be related to the full sentence or to the words surrounding the marked words.")
    hypothesis: str = dspy.OutputField(desc="Formulate an hypothesis.")
    explanation: str = dspy.OutputField(desc="A concise single-sentence explanation of the feature.")    
