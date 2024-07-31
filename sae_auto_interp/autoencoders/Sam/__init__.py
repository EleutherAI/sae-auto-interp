from ..wrapper import AutoencoderLatents
from .model import AutoEncoder

DEVICE = "cuda:0"
DICTIONARY_ID = 10
DICTIONARY_SIZE = 32_768


def _load(submodules, module, path):
    submodules[module._module_path] = module

    ae = AutoEncoder.from_pretrained(path, device=DEVICE)

    module.ae = AutoencoderLatents(
        ae, _forward=lambda x: ae.encode(x), width=DICTIONARY_SIZE
    )


def load_sam_autoencoders(
    model, ae_layers, weight_dir, modules=["embed", "mlp", "attention", "resid"]
):
    submodules = {}

    if "embed" in modules:
        _load(
            submodules,
            model.gpt_neox.embed_in,
            f"{weight_dir}/embed/{DICTIONARY_ID}_{DICTIONARY_SIZE}/ae.pt",
        )

    for i in ae_layers:
        if "mlp" in modules:
            _load(
                submodules,
                model.gpt_neox.layers[i].mlp,
                f"{weight_dir}/mlp_out_layer{i}/{DICTIONARY_ID}_{DICTIONARY_SIZE}/ae.pt",
            )

        if "attention" in modules:
            _load(
                submodules,
                model.gpt_neox.layers[i].attention,
                f"{weight_dir}/attn_out_layer{i}/{DICTIONARY_ID}_{DICTIONARY_SIZE}/ae.pt",
            )

        if "resid" in modules:
            _load(
                submodules,
                model.gpt_neox.layers[i],
                f"{weight_dir}/resid_out_layer{i}/{DICTIONARY_ID}_{DICTIONARY_SIZE}/ae.pt",
            )

        print(f"Loaded autoencoders for layer {i}")

    with model.edit(" "):
        for path, submodule in submodules.items():
            if "embed" not in path and "mlp" not in path:
                acts = submodule.output[0]
            else:
                acts = submodule.output
            submodule.ae(acts, hook=True)

    return submodules
