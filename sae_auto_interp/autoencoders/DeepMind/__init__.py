import torch as t
from sae_lens import SAE
from nnsight import LanguageModel

def load_gemma_autoencoders(
    ae_layers: list[int],
    size: str = "16k",
    type: str = "res",
):
    model = LanguageModel(
        "google/gemma-2-2b", 
        torch_dtype=t.bfloat16, 
        dispatch=True, 
        attn_implementation="eager", 
        device_map="auto"
    )
    submodules = {}

    for layer in ae_layers:

        submodule = model.model.layers[layer]
    
        sae = SAE.from_pretrained(
            release=f"gemma-scope-2b-pt-{type}-canonical",
            sae_id=f"layer_{layer}/width_{size}/canonical",
            device="cuda",
        )[0].to(t.bfloat16)
        
        submodules[submodule._path] = (sae, submodule)

    return model, submodules
