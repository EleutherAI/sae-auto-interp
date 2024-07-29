from nnsight import LanguageModel 
from sae_auto_interp.utils import load_tokenized_data
from sae_auto_interp.autoencoders.ae import load_autoencoders
from sae_auto_interp.features import CombinedStat, FeatureRecord, Logits, Activation, QuantileSizes

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
submodule_dict = load_autoencoders(
    model, 
    list(range(0,12,1)),
    "saved_autoencoders/gpt2_32k",
)

tokens = load_tokenized_data(model.tokenizer)

raw_features_path = "raw_features_small"
processed_features_path = "processed_features_small"

W_U = model.transformer.ln_f.weight * model.lm_head.weight

stats = CombinedStat(
    logits = Logits(
        model.tokenizer, 
        W_U = W_U
    ),
    activations = Activation(
        k = 1000,
    ),
)    
for layer, submodule in submodule_dict.items():
    ae = submodule.ae._module

    records = FeatureRecord.from_tensor(
        tokens,
        layer,
        raw_dir=raw_features_path,
        selected_features=list(range(50)),
        max_examples=10000
    )
    
    for record in records:
        for example in record.examples:
            example.decode(model.tokenizer)

    stats.refresh(
        W_dec=ae.autoencoder.decoder.weight,
    )
    
    stats.compute(records, save_dir=processed_features_path)