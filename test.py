# %%

import os

config_path = "configs/caden_gpt2.yaml"
ae_path = "sae_auto_interp/autoencoders/oai/gpt2"
raw_features_path = "raw_features"
processed_features_path = "processed_features"
os.environ["CONFIG_PATH"] = config_path

# %%

from nnsight import LanguageModel 
from sae_auto_interp.utils import load_tokenized_data
from sae_auto_interp.autoencoders.ae import load_autoencoders
from sae_auto_interp.features import CombinedStat, FeatureRecord, Logits, Activation, QuantileSizes

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
ae_dict, submodule_dict = load_autoencoders(
    model, 
    list(range(0,12,2)),
    ae_path
)
tokens = load_tokenized_data(model.tokenizer)

# %%

stats = CombinedStat(
    logits = Logits(
        model.tokenizer, 
        W_U = model.transformer.ln_f.weight * model.lm_head.weight
    ),
    activations = Activation(
        k=100,
    ),
    quantiles=QuantileSizes()
)    



for layer, ae in ae_dict.items():

    records = FeatureRecord.from_tensor(
        tokens,
        layer,
        tokenizer=model.tokenizer,
        raw_dir=raw_features_path,
        selected_features=list(range(0,10)),
        min_examples=200,
        max_examples=2000
    )

    stats.refresh(
        W_dec=ae.decoder.weight,
    )
    stats.compute(records)


