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
        self.explainer = dspy.Predict(Explanations)
        self.explainer = dspy.LabeledFewShot().compile(
            self.explainer,
            trainset=TRAINSET,
        )

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
    # special_words: List[str] = dspy.OutputField(desc="Find the special words that are selected in the examples and list a couple of them. Search for patterns in these words, if there are any. Don't list more than 5 words.")
    shared_features: List[str] = dspy.OutputField(desc="Write down general shared features of the text examples. This could be related to the full sentence or to the words surrounding the marked words.")
    hypothesis: str = dspy.OutputField(desc="Formulate an hypothesis.")
    explanation: str = dspy.OutputField(desc="A concise single-sentence explanation of the feature. This is the final output of the model and is what will be graded.")    


TRAINSET = [
    dspy.Example(
        feature_examples=[
            FeatureExample(
                text="and he was over the moon to find",
                max_activation=9.0,
                tokens=["over", "the", "moon"],
                tokens_and_activations=[("over", 5), ("the", 6), ("moon", 9)],
            ),
            FeatureExample(
                text="it was better than meets the eye",
                max_activation=7.0,
                tokens=["than", "meets", "the", "eye"],
                tokens_and_activations=[
                    ("than", 4),
                    ("meets", 7),
                    ("the", 2),
                    ("eye", 5),
                ],
            ),
            FeatureExample(
                text="I was also over the moon when",
                max_activation=8.0,
                tokens=["over", "the", "moon"],
                tokens_and_activations=[("over", 6), ("the", 5), ("moon", 8)],
            ),
            FeatureExample(
                text="it's more than meets the eye",
                max_activation=6.0,
                tokens=["than", "meets", "the", "eye"],
                tokens_and_activations=[
                    ("than", 3),
                    ("meets", 6),
                    ("the", 1),
                    ("eye", 4),
                ],
            ),
        ],
        shared_features=[
            "The examples contain common idioms.",
            "The activating tokens are parts of common idioms.",
            "The text examples all convey positive sentiment.",
        ],
        hypothesis="The activation values are the highest for the more common idioms.",
        explanation="Common idioms in text conveying positive sentiment.",
    ),
    dspy.Example(
        feature_examples=[
            FeatureExample(
                text="a river is wide but the ocean is wider. The ocean",
                max_activation=8.0,
                tokens=["er"],
                tokens_and_activations=[("er", 8)],
            ),
            FeatureExample(
                text='every year you get taller," she',
                max_activation=2.0,
                tokens=["er"],
                tokens_and_activations=[("er", 2)],
            ),
            FeatureExample(
                text="the hole was smaller but deeper than the",
                max_activation=9.0,
                tokens=["er", "er"],
                tokens_and_activations=[("er", 9), ("er", 9)],
            ),
        ],
        shared_features=[
            "The activating token appeared at the end of a comparative adjective.",
            "The comparative adjectives describe size.",
        ],
        hypothesis="The activation values are higher when comparing physical sizes more directly.",
        explanation='The token "er" at the end of a comparative adjective describing size.',
    ),
    dspy.Example(
        feature_examples=[
            FeatureExample(
                text='something happening inside my house", he',
                max_activation=7.0,
                tokens=["house"],
                tokens_and_activations=[("house", 7)],
            ),
            FeatureExample(
                text='presumably was always contained in a box", according',
                max_activation=9.0,
                tokens=["a", "box"],
                tokens_and_activations=[("a", 5), ("box", 9)],
            ),
            FeatureExample(
                text='people were coming into the smoking area". However he',
                max_activation=4.0,
                tokens=["smoking", "area"],
                tokens_and_activations=[("smoking", 2), ("area", 4)],
            ),
            FeatureExample(
                text='Patrick: "why are you getting in the way?" Later,',
                max_activation=4.0,
                tokens=["way", "?"],
                tokens_and_activations=[("way", 4), ("?", 2)],
            ),
        ],
        shared_features=[
            "The activating tokens are things that one can be in (literally or figuratively).",
            "The activating token is followed by a quotation mark, suggesting it occurs within speech.",
        ],
        hypothesis="The activation values are highest for distinctive objects or spaces.",
        explanation="Nouns representing distinct objects that contain something, often preceding a quotation mark.",
    ),
]
