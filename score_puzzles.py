# %%


import asyncio
from nnsight import LanguageModel
from tqdm import tqdm
import os
from keys import openrouter_key
os.environ["CONFIG_PATH"] = "configs/caden_gpt2.yaml"

from sae_auto_interp.clients import get_client
from sae_auto_interp.scorers import ScorerInput, FuzzingScorer
from sae_auto_interp.utils import load_tokenized_data, execute_model
from sae_auto_interp.features import FeatureRecord
from sae_auto_interp.experiments import sample_top_and_activation_quantiles
from sae_auto_interp.logger import logger
from sae_auto_interp.experiments.load_puzzles import load_puzzles

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
tokens = load_tokenized_data(model.tokenizer)

raw_features_path = "raw_features"
processed_features_path = "processed_features"
explanations_dir = "explanations/simple_local_70b"
scorer_out_dir = "scores/fuzz_70b_single/simple_local_70b_q4_nt5"

# %%

puzzles = load_puzzles("/share/u/caden/sae-auto-interp/sae_auto_interp/experiments/puzzles.json")
# %%
scorer_inputs = []

for layer in range(0,12,2):
    records = FeatureRecord.from_tensor(
        tokens,
        layer,
        tokenizer=model.tokenizer,
        selected_features=list(range(5)),
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
    break

# client = get_client("openrouter", "anthropic/claude-3-haiku", api_key=openrouter_key)
client = get_client("local", "casperhansen/llama-3-70b-instruct-awq")

scorer = FuzzingScorer(client, echo=False)

asyncio.run(
    execute_model(
        scorer, 
        scorer_inputs,
        output_dir=scorer_out_dir,
        record_time=True
    )
)