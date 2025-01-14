"""
WIP:
A dspy.Module-based pipeline for explanation and scoring.
Adapter for use from outside code, functions for training and evaluation.
"""

from itertools import repeat

import dspy.evaluate.evaluate
from more_itertools import chunked as batched
import random
import dspy
import dspy.evaluate
from .explainers.dspy import dspy_explainer_module, DSPyFeatureExample
from .features import FeatureRecord
from typing import List, Iterable, Literal
from .scorers.classifier.dspy_classifier import dspy_classifier_module


class DSPyClassifierPipeline(dspy.Module):
    def __init__(self, explainer_cot: bool = False, classifier_cot: bool = False,
                 explainer_few_shot: bool = True, classifier_few_shot: bool = True, batch_size: int = 5):
        self.explainer = dspy_explainer_module(cot=explainer_cot, few_shot=explainer_few_shot)
        self.classifier = dspy_classifier_module(
            cot=classifier_cot, few_shot=classifier_few_shot
        )
        self.batch_size = batch_size
    
    def forward(self, feature_examples: List[DSPyFeatureExample], test_examples: List[str]) -> dspy.Prediction:
        explanation = self.explainer(feature_examples=feature_examples[:10]).explanation    
        predictions = []
        for batch in batched(test_examples, self.batch_size):
            prediction_batch = self.classifier(feature_description=explanation, feature_examples=batch)
            if len(prediction_batch.is_feature) != len(batch):
                raise ValueError("Classifier returned wrong number of predictions")
            predictions.extend(prediction_batch.is_feature)
        return dspy.Prediction(is_feature=predictions)


def fuzz_corrupt_feature_example(
    feature_example: DSPyFeatureExample,
    do_fuzz: bool = True,
    return_counts: bool = False,
    threshold: float = 0.2,
    n_incorrect = None,
    insert_brackets: bool = True,
):
    activated_indices = [i for i, (_, act) in enumerate(feature_example.tokens_and_activations) if act > threshold]
    if return_counts:
        return len(activated_indices)
    indices_we_can_sample_from = set(range(len(feature_example.tokens))) - set(activated_indices)
    if n_incorrect is None:
        n_incorrect = len(activated_indices)
    random_indices = random.sample(indices_we_can_sample_from, min(n_incorrect, len(indices_we_can_sample_from)))
    selected_indices = random_indices if do_fuzz else activated_indices

    # new_activations = feature_example.tokens_and_activations[:]
    # for i in activated_indices:
    #     new_activations[i] = (new_activations[i][0], 0.0)
    # for i in random_indices:
    #     new_activations[i] = (new_activations[i][0], random.random() * feature_example.max_activation)
    
    text = []
    if insert_brackets:
        left, right = "<<", ">>"
    else:
        left, right = "", ""
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
    return "".join(text)
    
    # return DSPyFeatureExample(
    #     text=feature_example.text,
    #     max_activation=feature_example.max_activation,
    #     tokens=feature_example.tokens,
    #     tokens_and_activations=new_activations,
    # )


def split_classification_scoring(
    source_set: Iterable[FeatureRecord],
    method: Literal["fuzz", "pseudo_fuzz", "detect"],
    tokenizer
):
    train_records = feature_record_generator_to_feature_example_list(
        source_set, "train", tokenizer
    )
    test_records = feature_record_generator_to_feature_example_list(
        source_set, "test", tokenizer
    )
    random_records = feature_record_generator_to_feature_example_list(
        source_set, "random", tokenizer
    )
    all_train_records, all_test_records, all_test_labels = [], [], []
    for train_records, test_records, random_records in zip(
        train_records, test_records, random_records
    ):
        test_records = random.sample(test_records, len(test_records))
        if method == "fuzz":
            test_records_corrupted = [
                fuzz_corrupt_feature_example(x, do_fuzz=True, insert_brackets=method != "detect") for x in test_records[: len(test_records) // 2]
            ]
        num_tokens = [fuzz_corrupt_feature_example(x, do_fuzz=False, return_counts=True) for x in test_records]
        avg_num_tokens = int(sum(num_tokens) / len(num_tokens))
        test_records_uncorrupted = [
            fuzz_corrupt_feature_example(x, do_fuzz=False)
            for x in test_records[len(test_records) // 2 :]
        ]
        if method == "pseudo_fuzz":
            test_records_corrupted = [
                fuzz_corrupt_feature_example(x, do_fuzz=True, n_incorrect=avg_num_tokens, insert_brackets=True) for x in random_records[: len(test_records) // 2]
            ]
        elif method == "detect":
            test_records_corrupted = [
                fuzz_corrupt_feature_example(x, do_fuzz=False, n_incorrect=avg_num_tokens, insert_brackets=False) for x in random_records[: len(test_records) // 2]
            ]
        test_labels = [0] * len(test_records_corrupted) + [1] * len(
            test_records_uncorrupted
        )
        test_records = test_records_corrupted + test_records_uncorrupted
        test_records, test_labels = zip(*random.sample(list(zip(test_records, test_labels)), len(test_records)))
        
        all_train_records.append(train_records)
        all_test_records.append(test_records)
        all_test_labels.append(test_labels)
    train_records, test_records, test_labels = all_train_records, all_test_records, all_test_labels
    
    return [
        dspy.Example(
            feature_examples=tr,
            test_examples=te,
            label=la,
        ).with_inputs("feature_examples", "test_examples")
        for tr, te, la in zip(train_records, test_records, test_labels)
    ]


def accuracy_score(labels, predictions, trace=None):
    if not isinstance(labels, (list, tuple)):
        labels = labels.label
    if predictions is None:
        return 0.0
    return sum([p == l for p, l in zip(predictions.is_feature, labels)]) / len(labels)


def evaluate_classifier_pipeline(
    evaluation_set: Iterable[FeatureRecord],
    tokenizer,
    model: dspy.LM,
    seed=2,
    method: str = "pseudo_fuzz",
    classifier=None,
    **pipeline_kwargs,
):
    random.seed(seed)
    evalset = split_classification_scoring(evaluation_set, method, tokenizer)
    if classifier is None:
        classifier = DSPyClassifierPipeline(**pipeline_kwargs)
    classifier.set_lm(model)
    # predictions = dspy.Parallel()(list(zip(repeat(classifier), zip(train_records, test_records))))
    return dspy.evaluate.Evaluate(
        devset=evalset,
        display_table=False,
        display_progress=True,
        max_errors=float("inf"),
    )(classifier, metric=accuracy_score)
    # accuracies = []
    # for test_label, prediction in zip(test_labels, predictions):
    #     accuracies.append(accuracy_score(test_label, prediction))
    # return sum(accuracies) / len(accuracies)


def train_classifier_pipeline(
    training_set: Iterable[FeatureRecord],
    tokenizer,
    model: dspy.LM,
    seed=2,
    method: str = "pseudo_fuzz",
    eval_loader=None,
    **pipeline_kwargs,
):
    random.seed(seed)
    trainset = split_classification_scoring(training_set, method, tokenizer)
    base_classifier = DSPyClassifierPipeline(**pipeline_kwargs)
    base_classifier.set_lm(model)
    dspy.configure(lm=model)
    # optimizer = dspy.MIPROv2(metric=accuracy_score, prompt_model=model, task_model=model,
    #                      metric_threshold=0.7, auto="medium")
    optimizer = dspy.BootstrapFewShotWithRandomSearch(metric=accuracy_score, metric_threshold=0.7, num_candidate_programs=4)
    classifier = optimizer.compile(base_classifier, trainset=trainset,
                            **(dict(valset=split_classification_scoring(eval_loader, method, tokenizer)) if eval_loader is not None else {})
                               )
    return classifier
    


def feature_record_generator_to_feature_example_list(
    feature_record_generator: Iterable[FeatureRecord],
    extract_record: Literal["train", "test", "random"],
    tokenizer,
) -> List[List[DSPyFeatureExample]]:
    """Convert a generator of FeatureRecords to a list of lists of DSPyFeatureExamples for use in the DSPy pipeline.

    Args:
        feature_record_generator (Iterable[FeatureRecord]): the generator of FeatureRecords to convert
            May be obtained from a features.FeatureLoader
        tokenizer (Tokenizer): the tokenizer to use for decoding tokens

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
            for ex in (record.train if extract_record == "train" else (y for x in record.test for y in x) if extract_record == "test" else record.random_examples)
        ]
        for record in feature_record_generator
    ]
