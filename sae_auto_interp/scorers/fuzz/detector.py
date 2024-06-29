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
    example: str
    max_activation: float
    quantile: int
    marked: bool = False

class DetectionScorer(Scorer):

    def __init__(
        self,
        model,
        provider,
        test_model_dir
    ):
        super().__init__(validate=True)
        self.name = "detection_half"
        self.model = model
        self.provider = provider
        self.tokenizer = AutoTokenizer.from_pretrained(test_model_dir)

    def build_samples(self, test_examples):

        all_samples = []

        for quantile, example_set in enumerate(test_examples):
            for example in example_set:
                all_samples.append(
                    Sample(
                        example=example.text,
                        max_activation=max(example.activations),
                        quantile=quantile
                    )
                )
        
        return all_samples

    def __call__(
            self, 
            scorer_in: ScorerInput,
            echo=False
    ) -> ScorerResult:
        random.seed(det_cfg.seed)
        
        test_examples = self.build_samples(scorer_in.test_examples)

        random.shuffle(test_examples)

        example_batches = [
            test_examples[i:i+det_cfg.batch_size] 
            for i in range(0, len(test_examples), det_cfg.batch_size)
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
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        return ScorerResult(
            input="",
            response="",
            score=results
        )

    def query_wrapper(self, batch, explanation, echo, result_queue):
        # Create a new client for each process
        client = get_client(self.provider, self.model)
        time.sleep(1)
        
        examples = ""
        for i, example in enumerate(batch):
            examples += f"Example {i}: {example.example}\n"

        results = self.query(
            batch,
            explanation, 
            examples, 
            echo=echo,
            client=client
        )

        # Put results in the queue
        result_queue.put(results)
            
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

        return results