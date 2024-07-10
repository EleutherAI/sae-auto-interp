from nnsight import LanguageModel 
import os 

os.environ["CONFIG_PATH"] = "configs/caden_gpt2.yaml"

from sae_auto_interp.utils import load_tokenized_data
from sae_auto_interp.autoencoders.ae import load_autoencoders
from sae_auto_interp.features import CombinedStat, FeatureRecord, Logits, Activation, QuantileSizes

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
submodule_dict = load_autoencoders(
    model, 
    list(range(0,12,2)),
    "sae_auto_interp/autoencoders/oai/gpt2"
)

tokens = load_tokenized_data(model.tokenizer)

raw_features_path = "raw_features"
processed_features_path = "processed_features"

W_U = model.transformer.ln_f.weight * model.lm_head.weight

stats = CombinedStat(
    # logits = Logits(
    #     model.tokenizer, 
    #     W_U = W_U
    # ),
    activations = Activation(
        k=1000,
    ),
    # quantiles=QuantileSizes()
)    

for layer, submodule in submodule_dict.items():
    ae = submodule.ae._module

    records = FeatureRecord.from_tensor(
        tokens,
        layer,
        tokenizer=model.tokenizer,
        raw_dir=raw_features_path,
        selected_features=list(range(50)),
        max_examples=10000
    )
    
    # stats.refresh(
    #     W_dec=ae.decoder.weight,
    # )
    
    stats.compute(records, save_dir=processed_features_path)