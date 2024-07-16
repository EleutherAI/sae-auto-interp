# %%

import asyncio
from nnsight import LanguageModel
from tqdm import tqdm

from sae_auto_interp.clients import get_client,execute_model
from sae_auto_interp.scorers import ScorerInput, FuzzingScorer
from sae_auto_interp.utils import load_tokenized_data
from sae_auto_interp.features import FeatureRecord
from sae_auto_interp.features.sampling import sample_top_and_quantiles
from sae_auto_interp.logger import logger

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
tokens = load_tokenized_data(model.tokenizer)

raw_features_path = "raw_features"
processed_features_path = "processed_features"
explanations_dir = "results/explanations/simple"
scorer_out_dir = "results/scores/simple"

def load_explanation(feature):
    explanations_path = f"{explanations_dir}/layer{feature.layer_index}_feature{feature.feature_index}.txt"

    with open(explanations_path, "r") as f:
        explanation = f.read()

    if type(explanation) == dict:
        explanation = explanation["result"]

    return explanation

scorer_inputs = []



for layer in range(0,12,2):

    module_path = f".transformer.h.{layer}"
    records = FeatureRecord.from_tensor(
        tokens,
        module_path,
        selected_features=list(range(20)),
        raw_dir= raw_features_path,
        sampler=sample_top_and_quantiles,
        min_examples=120,
        max_examples=10000,
        n_random=10
    )

    for record in tqdm(records):
        try:
            explanation = load_explanation(record.feature)

        except Exception as e:
            logger.error(f"Failed while sampling for {record.feature}: {e}") 
            continue  

        extra = [batch[5:] for batch in record.test]
        test = [batch[:5] for batch in record.test]

        scorer_inputs.append(
            ScorerInput(
                record=record,
                test_examples=test,
                explanation=explanation,
                random_examples=record.random_examples,
                extra_examples=sum(extra, [])
            )
        )


# client = get_client("openrouter", "anthropic/claude-3-haiku", api_key=openrouter_key)
client = get_client("outlines", "casperhansen/llama-3-70b-instruct-awq")

scorer = FuzzingScorer(client, tokenizer=model.tokenizer, echo=False)

asyncio.run(
    execute_model(
        scorer, 
        scorer_inputs,
        output_dir=scorer_out_dir,
        record_time=True
    )
)