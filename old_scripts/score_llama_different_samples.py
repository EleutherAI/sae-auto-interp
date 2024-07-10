# %%

import asyncio
from nnsight import LanguageModel
from tqdm import tqdm
from sae_auto_interp.clients import get_client
from sae_auto_interp.utils import load_tokenized_data, get_samples,execute_model
from sae_auto_interp.scorers import ScorerInput, FuzzingScorer
from sae_auto_interp.features import FeatureRecord
from sae_auto_interp.experiments import sample_top_and_quantiles
from sae_auto_interp.logger import logger
import torch

model = LanguageModel("meta-llama/Meta-Llama-3-8B", device_map="cpu", dispatch=True,torch_dtype =torch.bfloat16)
tokens = load_tokenized_data(model.tokenizer)
print(tokens.shape)
raw_features_path = "raw_features_llama"

samples = get_samples(N_LAYERS=32,N_FEATURES=131072,N_SAMPLES=1000)

def load_explanation(explanation_dir,feature):
    explanations_path = f"{explanation_dir}/layer{feature.layer_index}_feature{feature.feature_index}.txt"

    with open(explanations_path, "r") as f:
        explanation = f.read()

    return explanation

scorer_inputs_explanation_10_100 = []
scorer_inputs_explanation_10_all = []
scorer_inputs_explanation_20_100 = []
scorer_inputs_explanation_20_all = []
scorer_inputs_explanation_10_all_10 = []

for layer in [4]:
    records = FeatureRecord.from_tensor(
        tokens,
        layer,
        tokenizer=model.tokenizer,
        selected_features=samples[layer],
        raw_dir= raw_features_path,
        n_random=10,
        min_examples=100,
        max_examples=2000
    )
    
    for record in tqdm(records):

        try:
            explanation_10_100 = load_explanation("saved_explanations/llama_simple_top10_100",record.feature)
            explanation_10_all = load_explanation("saved_explanations/llama_simple_top10_all",record.feature)
            explanation_20_100 = load_explanation("saved_explanations/llama_simple_top20_100",record.feature)
            explanation_20_all = load_explanation("saved_explanations/llama_simple_top20_all",record.feature)
            explanation_10_all_10 = load_explanation("saved_explanations/llama_simple_top10_all_10",record.feature)
            
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

        scorer_inputs_explanation_10_100.append(
            ScorerInput(
                record=record,
                test_examples=test,
                explanation=explanation_10_100
            )
        )
        scorer_inputs_explanation_10_all.append(
            ScorerInput(
                record=record,
                test_examples=test,
                explanation=explanation_10_all
            )
        )
        scorer_inputs_explanation_20_100.append(
            ScorerInput(
                record=record,
                test_examples=test,
                explanation=explanation_20_100
            )
        )
        scorer_inputs_explanation_20_all.append(
            ScorerInput(
                record=record,
                test_examples=test,
                explanation=explanation_20_all
            )
        )
        scorer_inputs_explanation_10_all_10.append(
            ScorerInput(
                record=record,
                test_examples=test,
                explanation=explanation_10_all_10
            )
        )

client = get_client("local", "casperhansen/llama-3-70b-instruct-awq", base_url="http://127.0.0.1:8000")
scorer = FuzzingScorer(client)
scorer_out_dir = "scores/llama_10_100"

asyncio.run(
    execute_model(
        scorer, 
        scorer_inputs_explanation_10_100,
        output_dir=scorer_out_dir,
    )
)
scorer_out_dir = "scores/llama_10_all"

asyncio.run(
    execute_model(
        scorer, 
        scorer_inputs_explanation_10_all,
        output_dir=scorer_out_dir,
    )
)

scorer_out_dir = "scores/llama_20_100"

asyncio.run(
    execute_model(
        scorer, 
        scorer_inputs_explanation_20_100,
        output_dir=scorer_out_dir,
    )
)

scorer_out_dir = "scores/llama_20_all"

asyncio.run(
    execute_model(
        scorer, 
        scorer_inputs_explanation_20_all,
        output_dir=scorer_out_dir,
    )
)

scorer_out_dir = "scores/llama_10_all_10"

asyncio.run(
    execute_model(
        scorer, 
        scorer_inputs_explanation_10_all_10,
        output_dir=scorer_out_dir,
    )
)

