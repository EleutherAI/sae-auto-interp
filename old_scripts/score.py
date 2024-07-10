import asyncio
from nnsight import LanguageModel
from tqdm import tqdm
from sae_auto_interp.clients import get_client
from sae_auto_interp.utils import load_tokenized_data, get_samples,execute_model
from sae_auto_interp.scorers import ScorerInput, FuzzingScorer
from sae_auto_interp.features import FeatureRecord
from sae_auto_interp.experiments import sample_top_and_quantiles
from sae_auto_interp.logger import logger
import argparse

from sae_auto_interp import cache_config as CONFIG

argparser = argparse.ArgumentParser()
argparser.add_argument("--layers", type=str, default="12,14")
args = argparser.parse_args()
layers = [int(layer) for layer in args.layers.split(",") if layer.isdigit()]


model = LanguageModel("openai-community/gpt2", device_map="cpu", dispatch=True)
CONFIG.batch_len = 64
tokens = load_tokenized_data(model.tokenizer,CONFIG=CONFIG)

raw_features_path = "raw_features"
processed_features_path = "processed_features"
explanations_dir = "explanations/cot"
scorer_out_dir = "scores/oai"

samples = get_samples()

def load_explanation(explanation_dir,feature):
    explanations_path = f"{explanation_dir}/layer{feature.layer_index}_feature{feature.feature_index}.txt"

    with open(explanations_path, "r") as f:
        explanation = f.read()

    return explanation

scorer_inputs_cot = []
scorer_inputs_simple = []
for layer in layers:
    records = FeatureRecord.from_tensor(
        tokens,
        layer,
        tokenizer=model.tokenizer,
        selected_features=samples[layer],
        raw_dir= raw_features_path,
        n_random=10,
        min_examples=200,
        max_examples=2000
    )
    
    for record in tqdm(records):

        try:
            explanation_cot = load_explanation("saved_explanations/cot",record.feature)
            explanation_simple = load_explanation("saved_explanations/simple",record.feature)
            _, test, extra = sample_top_and_quantiles(
                record=record,
                n_train=20,
                n_test=5,
                n_quantiles=4,
                seed=22,
                n_extra=10
            )
            
        except Exception as e:
            logger.error(f"Failed while sampling for {record.feature}: {e}") 
            continue

        record.extra = extra

        scorer_inputs_cot.append(
            ScorerInput(
                record=record,
                test_examples=test,
                explanation=explanation_cot
            )
        )
        scorer_inputs_simple.append(
            ScorerInput(
                record=record,
                test_examples=test,
                explanation=explanation_simple
            )
        )

client = get_client("local", "casperhansen/llama-3-70b-instruct-awq", base_url="http://127.0.0.1:8001")
scorer = FuzzingScorer(client)
scorer_out_dir = "scores/cot"
print("Running scorer for cot")
asyncio.run(
    execute_model(
        scorer, 
        scorer_inputs_cot,
        output_dir=scorer_out_dir,
    )
)
scorer_out_dir = "scores/simple"
print("Running scorer for simple")
asyncio.run(
    execute_model(
        scorer, 
        scorer_inputs_cot,
        output_dir=scorer_out_dir,
    )
)