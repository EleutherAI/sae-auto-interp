# %%
import asyncio
import random

from nnsight import LanguageModel
from sae_auto_interp.explainers import SimpleExplainer, ExplainerInput
from sae_auto_interp.clients import get_client, execute_model
from sae_auto_interp.utils import load_tokenized_data
from sae_auto_interp.features.features import FeatureRecord, load_feature_batch
from sae_auto_interp.features.utils import vis
import time
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
tokens = load_tokenized_data(model.tokenizer)

raw_features_path = "raw_features"
explainer_out_dir = "results/explanations/simple"

explainer_inputs=[]
random.seed(22)

start_time = time.time()
for layer in [0,2]:
    module_name = f".transformer.h.{layer}"

    
    records = load_feature_batch(
        tokens,
        module_name,
        selected_features=list(range(100)),
        raw_dir = raw_features_path,
        # min_examples=120,
        # max_examples=10000
    )

print(f"Time taken: {time.time() - start_time}")
# %%
    for record in records:

        examples = record.examples
        train_examples = random.sample(examples[:100], 10)
        
        explainer_inputs.append(
            ExplainerInput(
                train_examples=train_examples,
                record=record
            )
        )

    break

client = get_client("local", "meta-llama/Meta-Llama-3-8B-Instruct")

explainer = SimpleExplainer(client, tokenizer=model.tokenizer)

asyncio.run(
    execute_model(
        explainer, 
        explainer_inputs,
        output_dir=explainer_out_dir,
        record_time=True
    )
)

