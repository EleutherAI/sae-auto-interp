"""
WIP:
A dspy.Module-based pipeline for explanation and scoring.
Adapter for use from outside code, functions for training and evaluation.
"""

from itertools import repeat
from more_itertools import chunked as batched
import random
import dspy
import dspy.evaluate
from .explainers.dspy import dspy_explainer_module, DSPyFeatureExample
from .features import FeatureRecord
from typing import List, Iterable, Tuple
from .scorers.classifier.dspy_classifier import dspy_classifier_module


class DSPyClassifierPipeline(dspy.Module):
    def __init__(self, explainer_cot: bool = False, scorer_cot: bool = False, batch_size: int = 5):
        self.explainer = dspy_explainer_module(cot=explainer_cot)
        self.scorer = dspy_classifier_module(cot=scorer_cot)
        self.batch_size = batch_size
    
    def forward(self, feature_examples: List[DSPyFeatureExample], test_examples: List[str]) -> dspy.Prediction:
        explanation = self.explainer(feature_examples=feature_examples)
        predictions = []
        for batch in batched(feature_examples, self.batch_size):
            prediction_batch = self.scorer(feature_description=explanation, feature_examples=batch)
            if len(prediction_batch.is_feature) != len(batch):
                raise ValueError("Scorer returned wrong number of predictions")
            predictions.extend(prediction_batch.is_feature)
        return dspy.Prediction(is_feature=predictions)


def fuzz_corrupt_feature_example(
    feature_example: DSPyFeatureExample,
    do_fuzz: bool = True,
    threshold: float = 0.2,
):
    activated_indices = [i for i, (_, act) in enumerate(feature_example.tokens_and_activations) if act > threshold]
    indices_we_can_sample_from = set(range(len(feature_example.tokens))) - set(activated_indices)
    random_indices = random.sample(indices_we_can_sample_from, len(activated_indices))
    selected_indices = random_indices if do_fuzz else activated_indices

    # new_activations = feature_example.tokens_and_activations[:]
    # for i in activated_indices:
    #     new_activations[i] = (new_activations[i][0], 0.0)
    # for i in random_indices:
    #     new_activations[i] = (new_activations[i][0], random.random() * feature_example.max_activation)
    
    text = []
    left, right = "<<", ">>"
    i = 0
    while i < len(feature_example.tokens):
        if i in selected_indices:
            text.append(left)
            while i < len(feature_example.tokens) and i in selected_indices:
                text.append(feature_example.tokens[i])
                i += 1
            text.append(right)
        else:
            text.append(feature_example.tokens[i])
            i += 1
    print("".join(text))
    return "".join(text)
    
    # return DSPyFeatureExample(
    #     text=feature_example.text,
    #     max_activation=feature_example.max_activation,
    #     tokens=feature_example.tokens,
    #     tokens_and_activations=new_activations,
    # )


def split_explanation_scoring(
    source_set: Iterable[FeatureRecord],
    tokenizer
):
    train_records = feature_record_generator_to_feature_example_list(
        source_set, tokenizer, for_train=True
    )
    test_records = feature_record_generator_to_feature_example_list(
        source_set, tokenizer, for_train=False
    )
    all_train_records, all_test_records, all_test_labels = [], [], []
    for train_records, test_records in zip(train_records, test_records):
        test_records = random.sample(test_records, len(test_records))
        test_records_corrupted = [
            fuzz_corrupt_feature_example(x, do_fuzz=True) for x in test_records[: len(test_records) // 2]
        ]
        test_records_uncorrupted = [
            fuzz_corrupt_feature_example(x, do_fuzz=False) for x in test_records[len(test_records) // 2 :]
        ]
        test_labels = [0] * len(test_records_corrupted) + [1] * len(
            test_records_uncorrupted
        )
        test_records = test_records_corrupted + test_records_uncorrupted
        test_records, test_labels = zip(*random.sample(list(zip(test_records, test_labels)), len(test_records)))
        
        all_train_records.append(train_records)
        all_test_records.append(test_records)
        all_test_labels.append(test_labels)
    return all_train_records, all_test_records, all_test_labels


def evaluate_classifier_pipeline(
    training_set: Iterable[FeatureRecord],
    tokenizer,
    model: dspy.LM,
    seed=2,
    **pipeline_kwargs,
):
    random.seed(seed)
    train_records, test_records, test_labels = split_explanation_scoring(training_set, tokenizer)
    base_classifier = DSPyClassifierPipeline(**pipeline_kwargs)
    base_classifier.set_lm(model)
    predictions = dspy.Parallel()(list(zip(repeat(base_classifier), zip(train_records, test_records))))
    # print(predictions)
    accuracies = []
    for test_label, prediction in zip(test_labels, predictions):
        correct = [p == l for p, l in zip(prediction.is_feature, test_label)]
        # print(sum(correct) / len(correct))
        accuracies.append(sum(correct) / len(correct))
    return sum(accuracies) / len(accuracies)


# def train_classifier_pipeline(
#     training_set: Iterable[FeatureRecord],
#     tokenizer,
#     prompt_model: dspy.LM,
#     task_model: dspy.LM,
#     **pipeline_kwargs,
# ):
#     train_records, test_records, test_labels = split_explanation_scoring(training_set, tokenizer)
#     base_classifier = DSPyClassifierPipeline(**pipeline_kwargs)
#     dspy.evaluate.answer_exact_match()
#     dspy.MIPROv2(metric=feature_accuracy, prompt_model=prompt_model, task_model=task_model)


def feature_record_generator_to_feature_example_list(
    feature_record_generator: Iterable[FeatureRecord],
    tokenizer,
    for_train: bool = True,
) -> List[List[DSPyFeatureExample]]:
    """Convert a generator of FeatureRecords to a list of lists of DSPyFeatureExamples for use in the DSPy pipeline.

    Args:
        feature_record_generator (Iterable[FeatureRecord]): the generator of FeatureRecords to convert
            May be obtained from a features.FeatureLoader
        tokenizer (Tokenizer): the tokenizer to use for decoding tokens
        for_train (bool, optional): Whether to use train or test records. Defaults to True.

    Returns:
        List[List[DSPyFeatureExample]]: The list of lists of DSPyFeatureExamples corresponding to each of the input features.
    """
    return [
        [
            DSPyFeatureExample(
                text=tokenizer.decode(ex.tokens),
                max_activation=max(ex.activations.tolist()),
                tokens=tokenizer.batch_decode(ex.tokens),
                tokens_and_activations=list(
                    zip(tokenizer.batch_decode(ex.tokens), ex.activations.tolist())
                ),
            )
            for ex in (record.train if for_train else (y for x in record.test for y in x))
        ]
        for record in feature_record_generator
    ]
