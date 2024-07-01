from .prompts import get_gen_scorer_template
from ... import gen_cfg
from ..scorer import Scorer, ScorerResult, ScorerInput
import time
from ...clients import get_client
import json

import numpy as np

class GenerationScorer(Scorer):

    def __init__(
        self,
        model,
        provider,
        test_model,
        ae_dict
    ):
        super().__init__(validate=False)
        self.name = "generation"

        self.test_model = test_model

        self.client = get_client(provider, model)
        self.ae_dict = ae_dict

    def get_llm_examples(self, explanation, echo=False, max_retries=3):

        generation_args = {
            "max_tokens" : 3000,
            "temperature" : gen_cfg.temperature,
        }

        prompt = get_gen_scorer_template(explanation, gen_cfg.n_tests)
        
        attempts = 0
        while attempts < max_retries:

            response = self.client.generate(
                prompt,
                generation_args
            )

            try:
                example_list = json.loads(response)
                break
            except:
                attempts += 1
                print(f"Failed to generate examples. Attempt {attempts}/{max_retries}")
                if attempts == max_retries:
                    print("Failed to generate examples. Returning empty string.")
                    raise
                time.sleep(1)

        return prompt, response, list(example_list.values())
    
    def get_stats(self, record):

        max_activations = [
            max(example.activations)
            for example in record.examples
        ]

        std = np.std(max_activations)
        mean = np.mean(max_activations)

        return std, mean

    def __call__(
        self, 
        scorer_in: ScorerInput,
        echo=False
    ) -> ScorerResult:

        prompt, response, examples = self.get_llm_examples(
            scorer_in.explanation, 
            echo=echo
        )

        layer_index = scorer_in.record.feature.layer_index
        feature_index = scorer_in.record.feature.feature_index

        with self.test_model.trace(examples, scan=False, validate=False):
            acts = self.test_model.transformer.h[layer_index].output[0]
            latents, _ = self.ae_dict[layer_index].encode(acts)
            latents = latents[:,:,feature_index]
            latents[:,0] = 0.
            latents.save()

        std, mean = self.get_stats(scorer_in.record)
        results = [{
            "example" : example,
            "latents" : _latents,
            # Numpy flaot 64 not serializable by orjson
            "z_score" : float((max(_latents) - mean) / std),
            "max_activation" : max(_latents),
        } for example, _latents in zip(examples, latents.tolist())]
        
        del latents

        return ScorerResult(
            input="explanation",
            response=response,
            score={
                "results" : results,
                "max_activation" : scorer_in.record.max_activation,
            }
        )