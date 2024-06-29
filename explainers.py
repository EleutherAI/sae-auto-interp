import sys

sys.path.append("/share/u/caden/sae-auto-interp")

from sae_auto_interp.explainers.cot.cot import ChainOfThought
from sae_auto_interp.clients import get_client
from sae_auto_interp.explainers.explainer import ExplainerInput, run_explainers

client = get_client("local", "astronomer/Llama-3-8B-Instruct-GPTQ-8-Bit")
explainer = ChainOfThought(client)

from sae_auto_interp.autoencoders.ae import load_autoencoders
from sae_auto_interp.features.cache import FeatureCache

import asyncio


from nnsight import LanguageModel

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

ae_dict, submodule_dict, edits = load_autoencoders(
    model, 
    "/share/u/caden/sae-auto-interp/sae_auto_interp/autoencoders/oai/gpt2"
)


from transformer_lens import utils
from datasets import load_dataset
from nnsight import LanguageModel 

data = load_dataset("stas/openwebtext-10k", split="train")

tokens = utils.tokenize_and_concatenate(
    data, 
    model.tokenizer, 
    max_length=64
)   

tokens = tokens.shuffle(22)['tokens']



from sae_auto_interp.features.features import feature_loader, Feature

features_path = "/share/u/caden/sae-auto-interp/saved_records"
layer_index = 6
feature_index = 13452

feature = Feature(
    layer_index,
    feature_index
)


for ae, records in feature_loader(
    tokens, 
    [feature],
    model,
    ae_dict,
    features_path
):
    print(records)
    break


record = records[0]
record.top_logits = ["penis", "PENIS", "penis", "PENISPENIS"]

explainer_in = ExplainerInput(
    record.examples,
    record,
)


async def run():
    await run_explainers(
        explainer, 
        [explainer_in],
        "/share/u/caden/sae-auto-interp/scripts",
    )
asyncio.run(run())