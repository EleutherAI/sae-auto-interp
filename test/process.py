import os 

os.environ["CONFIG_PATH"] = "configs/caden_gpt2.yaml"

from nnsight import LanguageModel 

from sae_auto_interp.utils import load_tokenized_data
from sae_auto_interp.autoencoders.ae import load_autoencoders
from sae_auto_interp.features import CombinedStat, FeatureRecord, Logits, Activation, QuantileSizes

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
ae_dict, submodule_dict = load_autoencoders(
    model, 
    list(range(0,12,2)),
    "sae_auto_interp/autoencoders/oai/gpt2"
)

tokens = load_tokenized_data(model.tokenizer)

raw_features_path = "raw_features"
processed_features_path = "new_processed"

W_U = model.transformer.ln_f.weight * model.lm_head.weight
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
        selected_features=list(range(0,50)),
        max_examples=10000
    )
    
    # Refresh updates a memory intensive caches for stuff like
    # umap locations or logit matrices
    stats.refresh(
        W_dec=ae.decoder.weight,
    )
    # Compute updates records with stat information
    stats.compute(records)

    # Save the processed information to the processed feature dir
    for record in records:
        record.save(processed_features_path)