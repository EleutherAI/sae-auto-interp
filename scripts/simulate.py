import asyncio
from nnsight import LanguageModel
from tqdm import tqdm

from sae_auto_interp.clients import get_client, execute_model
from sae_auto_interp.scorers import ScorerInput, OpenAISimulator
from sae_auto_interp.utils import load_tokenized_data
from sae_auto_interp.features import FeatureRecord
from sae_auto_interp.load.sampling import sample_top_and_quantiles
from sae_auto_interp.logger import logger

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
tokens = load_tokenized_data(model.tokenizer)

raw_features_path = "raw_features"
processed_features_path = "processed_features"
explanations_dir = "results/explanations/simple"
scorer_out_dir = "results/scores/simple"

def load_explanation(feature):
    explanations_path = f"{explanations_dir}/{feature}.txt"

    with open(explanations_path, "r") as f:
        explanation = f.read()

    return explanation

scorer_inputs = []

for layer in range(0,12,2):
    module_name = f".transformer.h.{layer}"

    records = FeatureRecord.from_tensor(
        tokens,
        module_name=module_name,
        selected_features=list(range(5)),
        raw_dir= raw_features_path,
        sampler=lambda rec: sample_top_and_quantiles(rec, n_test=2, n_quantiles=4),
        n_random=0,
        min_examples=80,
        max_examples=10000
    )
    
    for record in tqdm(records):

        try:
            explanation = load_explanation(record.feature)

        except Exception as e:
            logger.error(f"Failed while sampling for {record.feature}: {e}") 
            continue

        scorer_inputs.append(
            ScorerInput(
                record=record,
                test_examples=sum(record.test, []),
                explanation=explanation
            )
        )

print(len(scorer_inputs))

client = get_client("outlines", "casperhansen/llama-3-70b-instruct-awq")
scorer = OpenAISimulator(client)

asyncio.run(
    execute_model(
        scorer, 
        scorer_inputs,
        output_dir=scorer_out_dir,
        record_time=True
    )
)