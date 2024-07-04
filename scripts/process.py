from nnsight import LanguageModel 
from tqdm import tqdm

from sae_auto_interp.utils import load_tokenized_data
from sae_auto_interp.autoencoders.ae import load_autoencoders
from sae_auto_interp.features import CombinedStat, FeatureRecord, Logits

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
ae_dict, submodule_dict = load_autoencoders(
    model, 
    list(range(0,12,2)),
    "sae_auto_interp/autoencoders/oai/gpt2"
)

tokens = load_tokenized_data(model.tokenizer)


raw_features_path = "raw_features"
processed_features_path = "processed_features"

stats = CombinedStat(
    logits = Logits(
        model, 
        get_top_logits=True,
    )
)    

for layer, ae in ae_dict.items():

    records = FeatureRecord.from_tensor(
        tokens,
        layer,
        tokenizer=model.tokenizer,
        raw_dir=raw_features_path,
        selected_features=list(range(0,50)),
        min_examples=200,
        max_examples=2000
    )
    
    # Refresh updates a memory intensive caches for stuff like
    # umap locations or logit matrices
    stats.refresh(
        W_dec=ae.decoder.weight,
        # Fold final layer norm into the lm_head
        W_U=model.transformer.ln_f.weight * model.lm_head.weight
    
    )
    # Compute updates records with stat information
    stats.compute(records)

    # Save the processed information to the processed feature dir
    for record in records:
        record.save(processed_features_path)