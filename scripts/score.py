import asyncio

from sae_auto_interp.clients import get_client
from sae_auto_interp.scorers import ScorerInput, FuzzingScorer
from sae_auto_interp.utils import get_samples, load_tokenized_data, execute_model
from sae_auto_interp.features import FeatureRecord
from sae_auto_interp.experiments import sample_top_and_quantiles
from sae_auto_interp.logger import logger
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

tokens = load_tokenized_data(tokenizer)
samples = get_samples(features_per_layer=200)

raw_features_path = "raw_features"
processed_features_path = "processed_features"
explanations_dir = "saved_explanations/cot"


def load_explanation(feature):
    explanations_path = f"{explanations_dir}/layer{feature.layer_index}_feature{feature.feature_index}.txt"

    with open(explanations_path, "r") as f:
        explanation = f.read()

    return explanation

scorer_inputs = []

for layer in [0]:
    records = FeatureRecord.from_tensor(
        tokens,
        layer,
        tokenizer=tokenizer,
        selected_features=samples[layer],
        raw_dir= raw_features_path,
        processed_dir=processed_features_path,
        min_examples=300,
        max_examples=2000
    )
    
    for record in records:

        explanation = load_explanation(record.feature)
        try:
            _, test = sample_top_and_quantiles(
                record=record,
                n_train=10,
                train_population=50,
                n_test_per_quantile=5,
                n_quantiles=4,
                seed=22,
            )
        except ValueError:
            continue

        scorer_inputs.append(
            ScorerInput(
                record=record,
                test_examples=test,
                explanation=explanation
            )
        )

client = get_client("local", "casperhansen/llama-3-70b-instruct-awq")
scorer = FuzzingScorer(client)
scorer_out_dir = "saved_scores/no_few_shot"

asyncio.run(
    execute_model(
        scorer, 
        scorer_inputs,
        output_dir=scorer_out_dir,
    )
)