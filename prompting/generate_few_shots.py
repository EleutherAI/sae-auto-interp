# %%
import asyncio
from nnsight import LanguageModel
from tqdm import tqdm
import os
import json

os.environ["CONFIG_PATH"] = "configs/caden_gpt2.yaml"

from sae_auto_interp.clients import get_client
from sae_auto_interp.scorers import ScorerInput, FuzzingScorer
from sae_auto_interp.utils import load_tokenized_data, execute_model
from sae_auto_interp.features import FeatureRecord
from sae_auto_interp.experiments import sample_top_and_quantiles
from sae_auto_interp.logger import logger

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
tokens = load_tokenized_data(model.tokenizer)

raw_features_path = "raw_features"
processed_features_path = "processed_features"
explanations_dir = "explanations/simple_local_70b"
scorer_out_dir = "scores/fuzz_70b/simple_local_70b_q10_nt2"
top_features_path = "top_features.json"

def load_explanation(feature):
    explanations_path = f"{explanations_dir}/{feature}.txt"

    with open(explanations_path, "r") as f:
        explanation = f.read()

    return explanation

def load_scores(feature):
    scores_path = f"{scorer_out_dir}/{feature}.json"

    with open(scores_path, "r") as f:
        scores = json.load(f)

    return scores

with open(top_features_path, "r") as f:
    best_features = json.load(f)


# %%

scorer_inputs  = []

scorer = FuzzingScorer(None, echo=True, get_prompts=True)

for layer in range(0,12,2):
    selected_features = [int(i) for i in best_features[str(layer)]]

    records = FeatureRecord.from_tensor(
        tokens,
        layer,
        tokenizer=model.tokenizer,
        selected_features=selected_features,
        raw_dir= raw_features_path,
        processed_dir=processed_features_path,
        n_random=10,
        min_examples=150,
        max_examples=10000
    )

    for record in tqdm(records):

        try:
            explanation = load_explanation(record.feature)

            record.examples = record.examples[100:]
            _, test, extra = sample_top_and_quantiles(
                record=record,
                n_train=0,
                n_test=5,
                n_quantiles=4,
                seed=22,
                n_extra=10
            )

        except Exception as e:
            logger.error(f"Failed while sampling for {record.feature}: {e}") 
            continue

        scorer_inputs.append(
            ScorerInput(
                record=record,
                test_examples=test,
                explanation=explanation,
                random_examples=record.random,
                extra_examples=extra
            )
        )
    
# %%async

import random
random.seed(22)

few_shot_examples = {}

for input in scorer_inputs:

    clean_batches, fuzzed_batches = await scorer(input)

    _some_clean = random.sample(clean_batches[0], 5)
    _some_fuzzed = random.sample(fuzzed_batches[0], 5)

    some_clean = []

    for s in _some_clean:
        point = {
            "text" : s.text,
            "score" : 1 if s.ground_truth else 0
        }
        some_clean.append(point)

    some_fuzzed = []

    for s in _some_fuzzed:
        point = {
            "text" : s.text,
            "score" : 1 if s.ground_truth else 0
        }
        some_fuzzed.append(point)
        
    few_shot_examples[f"{input.record.feature}"] = {
        "explanation": input.explanation,
        "clean": some_clean,
        "fuzzed": some_fuzzed
    }

# %%

with open("few_shot_examples.json", "w") as f:
    json.dump(few_shot_examples, f, indent=2)