# %%
import asyncio
import random
from nnsight import LanguageModel
from sae_auto_interp.explainers import SimpleExplainer, ExplainerInput
from sae_auto_interp.clients import get_client, execute_model
from sae_auto_interp.utils import load_tokenized_data
from sae_auto_interp.features import FeatureRecord
from sae_auto_interp.autoencoders import load_autoencoders

model = LanguageModel("EleutherAI/pythia-70m-deduped", device_map="auto", dispatch=True)
tokens = load_tokenized_data(model.tokenizer, seq_len=128)

submodule_dict = load_autoencoders(
    model,
    [0,1,2,3,4],
    "/share/u/caden/sae-auto-interp/sae_auto_interp/autoencoders/Sam/pythia-70m-deduped",
)

raw_features_path = "raw_features"
explainer_out_dir = "results/explanations/pythia"
explainer_inputs=[]


random.seed(22)

for module_name in submodule_dict.keys():    
    try:
        records = FeatureRecord.from_tensor(
            tokens,
            module_name,
            raw_dir = raw_features_path,
            min_examples=120,
            max_examples=10000
        )
    except:
        # Module had no features in the SFC
        continue

    for record in records:

        examples = record.examples
        train_examples = random.sample(examples[:100], 10)

        record.top_logits = None
        
        explainer_inputs.append(
            ExplainerInput(
                train_examples=train_examples,
                record=record
            )
        )

client = get_client("local", "astronomer/Llama-3-8B-Instruct-GPTQ-8-Bit")

explainer = SimpleExplainer(client, activations=True, cot=True,echo=True, tokenizer=model.tokenizer)

asyncio.run(
    execute_model(
        explainer, 
        explainer_inputs,
        output_dir=explainer_out_dir,
        record_time=True
    )
)