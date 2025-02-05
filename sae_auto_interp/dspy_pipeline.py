"""
WIP:
A dspy.Module-based pipeline for explanation and scoring.
Adapter for use from outside code, functions for training and evaluation.
"""


import random
from itertools import repeat
from typing import Iterable, List, Literal

import concurrent.futures
from dspy.utils.parallelizer import ParallelExecutor
import dspy
import dspy.evaluate
import dspy.evaluate.evaluate
from more_itertools import chunked as batched

from .explainers.dspy import (
    DSPyFeatureExample,
    dspy_explainer_module,
    feature_record_generator_to_feature_example_list,
)
from .features import FeatureRecord
from .logger import logger
from .scorers.classifier.dspy_classifier import dspy_classifier_module


class DSPyClassifierPipeline(dspy.Module):
    def __init__(self, explainer_cot: bool = False, classifier_cot: bool = False,
                 explainer_few_shot: bool = True, classifier_few_shot: bool = True,
                 batch_size: int = 5, n_aux_examples: int = 0,
                 drop_out_explainer_prob: float = 0.0,
                 ignore_errors: bool = True):
        self.explainer = dspy_explainer_module(cot=explainer_cot, few_shot=explainer_few_shot)
        self.classifier = dspy_classifier_module(
            cot=classifier_cot, few_shot=classifier_few_shot
        )
        self.batch_size = batch_size
        self.n_aux_examples = n_aux_examples
        self.drop_out_explainer_prob = drop_out_explainer_prob
        self.ignore_errors = ignore_errors
    
    def forward(self, feature_examples: List[DSPyFeatureExample], test_examples: List[str]) -> dspy.Prediction:
        if random.random() > self.drop_out_explainer_prob:
            explanation = self.explainer(feature_examples=feature_examples[:10]).explanation
        else:
            explanation = ""

        predictions = []

        # for batch in batched(test_examples, self.batch_size):
        #     prediction_batch = self.classifier(
        #         feature_description=explanation,
        #         train_examples=feature_examples[: self.n_aux_examples],
        #         feature_examples=batch,
        #     )
        #     if len(prediction_batch.is_feature) != len(batch):
        #         if self.ignore_errors:
        #             logger.error("Classifier returned wrong number of predictions")
        #             predictions.extend([0] * len(batch))
        #         else:
        #             raise ValueError("Classifier returned wrong number of predictions")
        #     else:
        #         predictions.extend(prediction_batch.is_feature)
        # return dspy.Prediction(is_feature=predictions)
        
        def classify(batch):
            prediction_batch = self.classifier(
                feature_description=explanation,
                train_examples=feature_examples[: self.n_aux_examples],
                feature_examples=batch,
            )
            if len(prediction_batch.is_feature) != len(batch):
                if self.ignore_errors:
                    logger.error("Classifier returned wrong number of predictions")
                    return [0] * len(batch)
                else:
                    raise ValueError("Classifier returned wrong number of predictions")
            else:
                return prediction_batch.is_feature
        # predictions_nested = dspy.Parallel()(list(zip(repeat(classify), batched(test_examples, self.batch_size))))

        # executor = ParallelExecutor(
        #     num_threads=8,
        #     max_errors=float("inf") if self.ignore_errors else 10,
        #     disable_progress_bar=True,
        # )
        # predictions_nested = executor.execute(classify, list(batched(test_examples, self.batch_size)))

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = [
                executor.submit(classify, batch)
                for batch in batched(test_examples, self.batch_size)
            ]
            predictions_nested = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        predictions = [p for sublist in predictions_nested for p in sublist]
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

    # new_activations = feature_example.tokens_and_activations[:]
    # for i in activated_indices:
    #     new_activations[i] = (new_activations[i][0], 0.0)
    # for i in random_indices:
    #     new_activations[i] = (new_activations[i][0], random.random() * feature_example.max_activation)

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
        num_tokens = [
            fuzz_corrupt_feature_example(x, do_fuzz=False, return_counts=True)
            for x in test_records
        ]
        avg_num_tokens = int(sum(num_tokens) / len(num_tokens))
        if method == "fuzz":
            test_records_corrupted = [
                fuzz_corrupt_feature_example(
                    x,
                    do_fuzz=True,
                    insert_brackets=True,
                    n_incorrect=avg_num_tokens,
                )
                for x in test_records[: len(test_records) // 2]
            ]
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
        return_all_scores=True,
        num_threads=1,
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
    optimizer_method: Literal["none", "mipro", "bootstrap"] = "mipro",
    eval_loader=None,
    **pipeline_kwargs,
):
    random.seed(seed)
    trainset = split_classification_scoring(training_set, method, tokenizer)
    base_classifier = DSPyClassifierPipeline(**pipeline_kwargs)
    base_classifier.set_lm(model)
    dspy.configure(lm=model)
    if optimizer_method.startswith("mipro"):
        optimizer = dspy.MIPROv2(metric=accuracy_score, prompt_model=model, task_model=model, metric_threshold=0.7,
                                 auto="light" if "_" not in optimizer_method else optimizer_method.split("_")[1])
    elif optimizer_method == "bootstrap":
        optimizer = dspy.BootstrapFewShotWithRandomSearch(
            metric=accuracy_score,
            metric_threshold=0.7,
            num_candidate_programs=12,
            max_errors=float("inf"),
            num_threads=8,
        )
    if optimizer_method == "none":
        classifier = base_classifier
    else:
        classifier = optimizer.compile(base_classifier, trainset=trainset,
                                **(dict(valset=split_classification_scoring(eval_loader, method, tokenizer)) if eval_loader is not None else {}),
                                **(dict(minibatch_size=4, requires_permission_to_run=False) if optimizer_method == "mipro" else {}),
                                # **(dict(max_demos=4) if optimizer_method == "bootstrap" else {})
                                )
    return classifier
    
