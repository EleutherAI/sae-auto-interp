import asyncio
from nnsight import LanguageModel
from tqdm import tqdm
import os

os.environ["CONFIG_PATH"] = "configs/caden_gpt2.yaml"

from sae_auto_interp.clients import get_client
from sae_auto_interp.scorers import ScorerInput, FuzzingScorer
from sae_auto_interp.utils import load_tokenized_data, execute_model
from sae_auto_interp.features import FeatureRecord
from sae_auto_interp.experiments import sample_top_and_activation_quantiles
from sae_auto_interp.logger import logger

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
tokens = load_tokenized_data(model.tokenizer)

raw_features_path = "raw_features"
processed_features_path = "processed_features"
explanations_dir = "explanations/simple_local_70b"
scorer_out_dir = "scores/fuzz_70b/simple_local_70b_q4_nt5"

def load_explanation(feature):
    explanations_path = f"{explanations_dir}/layer{feature.layer_index}_feature{feature.feature_index}.txt"

    with open(explanations_path, "r") as f:
        explanation = f.read()

    return explanation

scorer_inputs = []

for layer in range(0,12,2):
    records = FeatureRecord.from_tensor(
        tokens,
        layer,
        tokenizer=model.tokenizer,
        selected_features=list(range(20)),
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
            _, test, extra = sample_top_and_activation_quantiles(
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


# client = get_client("openrouter", "anthropic/claude-3.5-sonnet", api_key=openrouter_key)
client = get_client("local", "meta-llama/Meta-Llama-3-8B-Instruct")


for n_few_shots in range(2,9):
    scorer_out_dir = \
        f"/share/u/caden/sae-auto-interp/scores/fuzz_70b/simple_local_70b_nfew{n_few_shots}"
    os.makedirs(scorer_out_dir)

    scorer = FuzzingScorer(client, echo=False, n_few_shots=n_few_shots)

    asyncio.run(
        execute_model(
            scorer, 
            scorer_inputs,
            output_dir=scorer_out_dir,
        )
    )