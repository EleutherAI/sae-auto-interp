import sys

sys.path.append("/share/u/caden/sae-auto-interp")

import asyncio
from transformer_lens import utils
from datasets import load_dataset
from nnsight import LanguageModel 
from tqdm import tqdm
from sae_auto_interp.explainers.cot.cot import ChainOfThought
from sae_auto_interp.clients import get_client
from sae_auto_interp.explainers.explainer import ExplainerInput, run_explainers
from sae_auto_interp.autoencoders.ae import load_autoencoders
from sae_auto_interp.features.tokens import load_tokens
from sae_auto_interp.sample import get_samples
from sae_auto_interp.features.features import feature_loader, Feature
from sae_auto_interp.features.stats import CombinedStat, Logits


client = get_client("local", "astronomer/Llama-3-8B-Instruct-GPTQ-8-Bit")
explainer = ChainOfThought(client)

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

ae_dict, submodule_dict, edits = load_autoencoders(
    model, 
    "/share/u/caden/sae-auto-interp/sae_auto_interp/autoencoders/oai/gpt2"
)

tokens = load_tokens


samples = get_samples()
samples = {
    int(layer) : features[:10] 
    for layer, features in samples.items() 
    if int(layer) in ae_dict.keys()
}
features = [
    Feature(
        layer_index,
        feature_index
    )
    for layer_index, features in samples.items()
    for feature_index in features

]

features_path = "/share/u/caden/sae-auto-interp/saved_records"

stats = CombinedStat(
    logits = Logits(
        model=model,
        top_k_logits=10
    ),
)    

explainer_inputs = []

for ae, records in feature_loader(
    tokens, 
    features,
    model,
    ae_dict,
    features_path,
    pipe=True
):
    stats.refresh(W_dec=ae.decoder.weight)
    stats.compute(records)

    for record in records:
        explainer_inputs.append(
            ExplainerInput(
                record,
                # TODO: Implement some sampling
                record.examples[:10]
            )
        )


explainer_out_dir = "/share/u/caden/sae-auto-interp/scripts"

async def run():
    await run_explainers(
        explainer, 
        explainer_inputs,
        output_dir=explainer_out_dir,
    )

asyncio.run(run())