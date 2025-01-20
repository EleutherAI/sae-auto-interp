
from typing import List, Literal, Union, Optional

import dspy
from asyncer import asyncify
from pydantic import ValidationError, field_validator
from transformers import PreTrainedTokenizer

from ...explainers.dspy import (
    DSPyFeatureExample,
    feature_record_generator_to_feature_example_list,
)
from ...clients import DSPy
from ...features import FeatureRecord
from ...logger import logger
from .classifier import Classifier
from .sample import ClassifierOutput, Sample


class ExampleClassifier(dspy.Signature):
    """You are an intelligent and meticulous linguistics researcher.

    You will be given a certain feature of text, such as "male pronouns" or "text with negative sentiment". You will be given a few examples of text that contain this feature. Portions of the sentence which strongly represent this feature are between tokens << and >>.

    Some examples might be mislabeled. Your task is to determine if every single token within << and >> is correctly labeled. Consider that all provided examples could be correct, none of the examples could be correct, or a mix. An example is only correct if every marked token is representative of the feature

    For each example in turn, return true if the sentence is correctly labeled or false if the tokens are mislabeled. You must return your response in a valid Python list. Give probabilities for each example. Never output None."""
    
    feature_description: str = dspy.InputField(desc="Feature explanation")
    train_examples: List[Union[str, DSPyFeatureExample]] = dspy.InputField(
        desc="Training examples", default=[]
    )
    feature_examples: List[Union[str, DSPyFeatureExample]] = dspy.InputField(
        desc="Test examples"
    )
    is_feature: List[Literal[0, 1]] = dspy.OutputField(desc="Whether the example is correctly labeled")
    # is_feature_probabilities: List[float] = dspy.OutputField(desc="Predicted probabilities for each example")

    @field_validator("is_feature", mode="after")
    @classmethod
    def check_length(cls, v, info):
        if len(v) != len(info.data["feature_examples"]):
            raise ValueError("Length of is_feature and feature_examples must be the same")
        return v


class NoncomposableDSPyClassifier(Classifier):
    def __init__(
        self,
        client: dspy.LM,
        module: dspy.Module,
        tokenizer: PreTrainedTokenizer,
        verbose: bool,
        batch_size: int,
        log_prob: bool,
        n_aux_examples: int = 0,
        **generation_kwargs,
    ):
        self.client = client
        self.module = module
        self.tokenizer = tokenizer
        self.verbose = verbose
        
        self.n_aux_examples = n_aux_examples
        self.batch_size = batch_size
        self.generation_kwargs = generation_kwargs
        self.log_prob = log_prob

    async def _generate(
        self, explanation: str, batch: list[Sample]
    ) -> list[ClassifierOutput]:
        """
        Generate predictions for a batch of samples.
        """

        batched_input = dict(
            feature_description=explanation,
            train_examples=feature_record_generator_to_feature_example_list(
                [batch[0].record], extract_record="train", tokenizer=self.tokenizer
            )[: self.n_aux_examples],
            feature_examples=[sample.text for sample in batch],
        )
        try:
            result = await asyncify(self.module)(**batched_input, lm=self.client)
            ExampleClassifier.model_validate(ExampleClassifier(
                # you can see the Javascript in this one
                **result.toDict(),
                **batched_input
            ))
        except ValidationError as e:
            logger.error(f"DSPy validation error: {e}")
            results = []
            for sample in batch:
                data = sample.data
                data.prediction = -1
                data.correct = False
                results.append(data)
            return results
        results = []
        correct = []
        response = []
        for i, sample in enumerate(batch):
            data = sample.data
            prediction = result.is_feature[i]
            data.prediction = prediction
            data.correct = prediction == data.ground_truth
            correct.append(data.ground_truth)
            response.append(prediction)
            if self.log_prob:
                data.probability = result.is_feature_probabilities[i]
                data.conditional_probability = 1
            results.append(data)

            if self.verbose:
                result.text = sample.text
        return results


class DSPyClassifier(NoncomposableDSPyClassifier):
    def __init__(self, classifier,
                 module = None,
                 batch_size: Optional[int] = None, cot: bool = False,
                 few_shot: bool = True, n_aux_examples: int = 0
                 ):
        if batch_size is None:
            batch_size = classifier.batch_size
        assert isinstance(classifier.client, DSPy)
        client = classifier.client.client
        if module is None:
            module = dspy_classifier_module(cot=cot, few_shot=few_shot)
        self.classifier_module = module
        self.base_classifier = classifier
        super().__init__(
            client,
            self.classifier_module,
            tokenizer=classifier.tokenizer,
            verbose=classifier.verbose,
            batch_size=batch_size,
            log_prob=classifier.log_prob,
            n_aux_examples=n_aux_examples,
            **classifier.generation_kwargs,
        )
    
    def _prepare(self, record: FeatureRecord) -> list[list[Sample]]:
        return self.base_classifier._prepare(record)


def dspy_classifier_module(cot: bool = False, few_shot: bool = True):
    module = (dspy.Predict if not cot else dspy.ChainOfThought)(ExampleClassifier)
    if few_shot:
        module = dspy.LabeledFewShot().compile(
            module,
            trainset=TRAINSET,
        )
    return module


TRAINSET = [
    dspy.Example(
        feature_description="Words related to American football positions, specifically the tight end position.",
        feature_examples=[
            "<|endoftext|>Getty ImagesĊĊPatriots tight end Rob Gronkowski had his bossâĢĻ",
            "names of months used in The Lord of the Rings:ĊĊâĢľâĢ¦the",
            "Media Day 2015ĊĊLSU defensive end Isaiah Washington (94) speaks to the",
            "shown, is generally not eligible for ads. For example, videos about recent tragedies,",
            "line, with the left side âĢĶ namely tackle Byron Bell at tackle and guard Amini"
        ],
        is_feature=[True, False, True, False, True],
        is_feature_probabilities=[0.95, 0.05, 0.85, 0.1, 0.75]
    ),
    dspy.Example(
        feature_description="The word \"guys\" in the phrase \"you guys\".",
        feature_examples=[
            "enact an individual health insurance mandate?âĢĿ, Pelosi's response was to dismiss both",
            "birth control access<|endoftext|> but I assure you women in Kentucky aren't laughing as they struggle",
            "du Soleil Fall Protection Program with construction requirements that do not apply to theater settings because",
            "Ċ<|endoftext|> distasteful. Amidst the slime lurk bits of Schadenfre",
            "the<|endoftext|>ľI want to remind you all that 10 days ago (director Massimil"
        ],
        is_feature=[False, False, False, False, False],
        is_feature_probabilities=[0.1, 0.1, 0.1, 0.1, 0.1]
    ),
    dspy.Example(
        feature_description="\"of\" before words that start with a capital letter.",
        feature_examples=[
            "climate, TomblinâĢĻs Chief of Staff Charlie Lorensen said.Ċ",
            "no wonderworking relics, no true Body and Blood of Christ, no true Baptism",
            "ĊĊDeborah Sathe, Head of Talent Development and Production at Film London,",
            "ĊĊIt has been devised by Director of Public Prosecutions (DPP)",
            "and fair investigation not even include the Director of Athletics? Â· Finally, we believe the"
        ],
        is_feature=[True, True, True, True, True],
        is_feature_probabilities=[0.9, 0.95, 0.9, 0.95, 0.95]
    ),
]
