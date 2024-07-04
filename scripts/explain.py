import asyncio
from tqdm import tqdm
import random
from transformers import AutoTokenizer

from sae_auto_interp.explainers import ChainOfThought, ExplainerInput
from sae_auto_interp.clients import get_client
from sae_auto_interp.utils import execute_model, load_tokenized_data
from sae_auto_interp.features import FeatureRecord
from sae_auto_interp.experiments import sample_quantiles


tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokens = load_tokenized_data(tokenizer)

raw_features_path = "raw_features"
processed_features_path = "processed_features"
explainer_out_dir = "explanations/cot"

explainer_inputs = []
random.seed(22)

for layer in range(0,12,2):
    records = FeatureRecord.from_tensor(
        tokens,
        layer,
        tokenizer=tokenizer,
        selected_features=list(range()),
        raw_dir=raw_features_path,
        processed_dir=processed_features_path,
        min_examples=200,
        max_examples=2000
    )

    for record in records:
        if not record:
            continue

        try:
            train, _ = sample_quantiles(
                record=record,
                n_train=20,
                n_quantiles=5
            )
        except:
            continue

        explainer_inputs.append(
            ExplainerInput(
                train_examples=train,
                record=record
            )
        )

client = get_client("local", "casperhansen/llama-3-70b-instruct-awq")
explainer = ChainOfThought(client)

asyncio.run(
    execute_model(
        explainer, 
        explainer_inputs,
        output_dir=explainer_out_dir,
    )
)
