import re
import random
import multiprocessing
from multiprocessing import Process, Queue

from .prompts import get_detection_template
from ... import det_cfg
from ...clients import get_client
from ..scorer import Scorer, ScorerResult, ScorerInput
from dataclasses import dataclass
from ...autoencoders.features import Feature, FeatureRecord
import json
from collections import defaultdict
import numpy as np
from transformers import AutoTokenizer
import logging
import time

@dataclass
class Sample:
    feature_index: int
    distance: float
    example: str
    marked: bool = False

class NeighborDetectionScorer(Scorer):

    def __init__(
        self,
        model,
        provider,
        test_model_dir
    ):
        super().__init__(validate=True)
        self.name = "neighbor_detection" 
        self.model = model
        self.provider = provider
        self.tokenizer = AutoTokenizer.from_pretrained(test_model_dir)

    def create_samples(self, layer_index, neighbors):
        samples = []
        for feature_index, distance in neighbors:
            record = FeatureRecord.load(
                feature=Feature(layer_index, feature_index),
                tokenizer=self.tokenizer,
                top_logits=None,
                nearest_neighbors=None
            )
            top_examples = record.examples[:det_cfg.n_examples]
            for example in top_examples:
                samples.append(Sample(
                    feature_index = feature_index, 
                    distance=distance,
                    example=example.text 
                ))

        return samples

    def __call__(
            self, 
            scorer_in: ScorerInput,
            echo=False
    ) -> ScorerResult:
        random.seed(det_cfg.seed)
        
        neighboring_examples = self.create_samples(
            scorer_in.record.feature.layer_index,
            scorer_in.record.nearest_neighbors
        )

        test_example_text = [example.text for example in scorer_in.test_examples]

        test_examples = [
            Sample(
                feature_index=scorer_in.record.feature.feature_index,
                distance=0.0,
                example=example_text
            ) 
            for example_text in test_example_text
        ]

        mixed_examples = neighboring_examples + test_examples
        self.total_examples = len(mixed_examples)
        random.shuffle(mixed_examples)

        example_batches = [
            mixed_examples[i:i+det_cfg.batch_size] 
            for i in range(0, len(mixed_examples), det_cfg.batch_size)
        ]

        # Create a queue to store results
        result_queue = Queue()

        # Create and start processes
        processes = []
        for batch in example_batches:
            p = Process(target=self.query_wrapper, args=(batch, scorer_in.explanation, echo, result_queue))
            processes.append(p)
            p.start()

        # Wait for all processes to complete
        for p in processes:
            p.join()

        # Collect results from the queue
        shared_results = []
        while not result_queue.empty():
            shared_results.append(result_queue.get())

        # Aggregate results
        inputs = []
        responses = []
        results = []

        for result in shared_results:
            inputs.append(result['prompt'])
            responses.append(result['response'])
            results.extend(result['results'])

        results = self.postprocess(scorer_in.record, results)
        inputs.insert(0, test_example_text)

        return ScorerResult(
            input=inputs,
            response=responses,
            score=results
        )

    def query_wrapper(self, batch, explanation, echo, result_queue):
        # Create a new client for each process
        client = get_client(self.provider, self.model)
        time.sleep(1)
        
        examples = ""
        for i, example in enumerate(batch):
            examples += f"Example {i}: {example.example}\n"

        prompt, response, results = self.query(
            batch,
            explanation, 
            examples, 
            echo=echo,
            client=client
        )

        # Put results in the queue
        result_queue.put({
            'prompt': prompt,
            'response': response,
            'results': results,
        })

    def postprocess(self, record, results):
        results_dict = defaultdict(lambda: 0)

        false_positive = 0
        true_positive = 0
        false_negative = 0
        true_negative = 0

        for result in results:
            if result.feature_index != record.feature.feature_index:
                name = f"f{result.feature_index}_d{result.distance:.3f}"
                if result.marked:
                    results_dict[name] += 1
                    false_positive += 1
                else:
                    results_dict[name] += 0
                    true_negative += 1
                
            elif result.feature_index == record.feature.feature_index:
                name = f"f{result.feature_index}_d{0.000}"
                if result.marked:
                    results_dict[name] += 1
                    true_positive += 1
                else:
                    results_dict[name] += 0
                    false_negative += 1

        results_dict["false_positive"] = false_positive
        results_dict["true_positive"] = true_positive
        results_dict["false_negative"] = false_negative
        results_dict["true_negative"] = true_negative
        
        return dict(results_dict)
                
            
    def query(self, batch, explanation, examples, echo=False, client=None, max_retries=3):
        if client is None:
            client = get_client(self.provider, self.model)

        prompt = get_detection_template(examples, explanation)

        attempts = 0
        while attempts < max_retries:
            response = client.generate(
                prompt,
                {
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"}
                },
            )

            try:
                selections = json.loads(response)
                break  # Exit the loop if JSON is successfully parsed
            except json.JSONDecodeError:
                attempts += 1
                print(f"Attempt {attempts}: Invalid JSON response, retrying...")
                if attempts == max_retries:
                    print(f"Max retries reached. Last response: {response}")
                    raise  # Raise the exception if max retries are reached
                time.sleep(1)  # Optional: Add a delay between retries

        results = []

        for i, (_, mark) in enumerate(selections.items()):
            if mark == 1:
                batch[i].marked = True
                results.append(batch[i])
            else:
                results.append(batch[i])

        return examples, "", results