import asyncio
from tqdm import tqdm
import random
from nnsight import LanguageModel
import os
from keys import openrouter_key

os.environ["CONFIG_PATH"] = "configs/caden_gpt2.yaml"

from sae_auto_interp.explainers import SimpleExplainer, ExplainerInput
from sae_auto_interp.clients import get_client
from sae_auto_interp.utils import execute_model, load_tokenized_data
from sae_auto_interp.features import FeatureRecord
from sae_auto_interp.experiments import sample_top_and_quantiles

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
tokens = load_tokenized_data(model.tokenizer)

raw_features_path = "raw_features"
explainer_out_dir = "explanations/cot"
explainer_inputs=[]
random.seed(22)
        
for layer in range(0,12,2):
    records = FeatureRecord.from_tensor(
        tokens,
        tokenizer=model.tokenizer,
        layer_index=layer,
        selected_features=list(range(50)),
        raw_dir= raw_features_path,
        max_examples=10000
    )

    for record in records:

        examples = record.examples

        if len(examples) < 150:
            continue

        train_examples = random.sample(examples[:100], 10)
        
        explainer_inputs.append(
            ExplainerInput(
                train_examples=train_examples,
                record=record
            )
        )


# client = get_client("openrouter", "meta-llama/llama-3-70b-instruct", api_key=openrouter_key)
client = get_client("local", "meta-llama/Meta-Llama-3-8B-Instruct")

explainer = SimpleExplainer(client)

explainer_out_dir = "explanations/local_simple"
asyncio.run(
    execute_model(
        explainer, 
        explainer_inputs,
        output_dir=explainer_out_dir,
    )
)

