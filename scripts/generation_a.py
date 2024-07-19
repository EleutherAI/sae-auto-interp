import asyncio
import random
from nnsight import LanguageModel
import os
from sae_auto_interp.scorers import GenerationScorer, ScorerInput
from sae_auto_interp.clients import get_client, execute_model
from sae_auto_interp.utils import load_tokenized_data
from sae_auto_interp.features import FeatureRecord, Feature

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
tokens = load_tokenized_data(model.tokenizer)

raw_features_path = "raw_features"
explanations_dir = "results/explanations/simple"
scorer_out_dir = "results/scores/generation_a"
random.seed(22)

# Moving this to a utils file in the future btw
def load_explanation(feature, explanations_dir):
    explanations_path = f"{explanations_dir}/{feature}.txt"

    with open(explanations_path, "r") as f:
        explanation = f.read()

    if type(explanation) == dict:
        explanation = explanation["result"]

    return explanation

scorer_inputs = []

def to_feature(string):
    string = string.replace(".txt", "").replace("feature", "") 
    return string.split("_")

for file in os.listdir(explanations_dir):
    module, feature = to_feature(file)

    feature = Feature(module, feature)

    explanation = load_explanation(
        feature,
        explanations_dir=explanations_dir
    )

    scorer_inputs.append(
        ScorerInput(
            explanation=explanation,
            record=FeatureRecord(feature),
            test_examples=None
        )
    )

client = get_client("outlines", "meta-llama/Meta-Llama-3-8B-Instruct")

scorer = GenerationScorer(
    client
)

asyncio.run(
    execute_model(
        scorer, 
        scorer_inputs,
        output_dir=scorer_out_dir,
        record_time=True
    )
)
