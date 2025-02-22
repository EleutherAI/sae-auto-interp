from nnsight import LanguageModel, NNsight
from simple_parsing import ArgumentParser
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from delphi.autoencoders import load_eai_autoencoders
from delphi.autoencoders.OpenAI.model import ACTIVATIONS_CLASSES, TopK
from delphi.autoencoders.wrapper import AutoencoderLatents
from functools import partial
from delphi.config import CacheConfig
from delphi.features import FeatureCache
from delphi.utils import load_tokenized_data
from sparsify.itda import ITDA
from pathlib import Path
import json
import os


def main(cfg: CacheConfig, args): 
    model_name = "EleutherAI/pythia-160m"
    # sae_dir = Path("../halutsae/checkpoints/itda/pythia-l9_mlp-transcoder-mean-skip-k32")
    sae_dir = Path("../halutsae/checkpoints/itda/pythia-l9_mlp-mean-k32")
    print("Using config", sae_dir.name)
    itda = ITDA.load_from_disk(sae_dir, device="cuda:0")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model = NNsight(model)
    model.tokenizer = AutoTokenizer.from_pretrained(model_name)
    submodule_dict = {}

    device = "cuda:0"
    for hookpoint in ["gpt_neox.layers.9.mlp"]:
        weight_dir = sae_dir / hookpoint
        
        # sae = Sae.load_from_disk(weight_dir, device=device).to(dtype=model.dtype)
        if not hookpoint.startswith(model.base_model_prefix):
            hookpoint = f"{model.base_model_prefix}.{hookpoint}"
        
        def _forward(itda, x):
            x_batch = x.shape[:-1]
            x = x.reshape(-1, x.shape[-1])
            out = itda(x, x)
            result = torch.zeros(x.shape[:-1] + (itda.dictionary_size,), device=out.weights.device, dtype=out.weights.dtype)
            result.scatter_(-1, out.indices, out.weights)
            return result.view(*x_batch, -1)
        
        atoms = hookpoint.split(".")
        submodule = model
        for atom in atoms:
            submodule = getattr(submodule, atom)
        submodule.ae = AutoencoderLatents(
            itda, partial(_forward, itda), width=itda.dictionary_size,
            hookpoint=hookpoint
        )

        submodule_dict[hookpoint] = submodule

    with model.edit("") as edited:
        for path, submodule in submodule_dict.items():
            if "embed" not in path and "mlp" not in path:
                acts = submodule.output[0]
            else:
                acts = submodule.output
            submodule.ae(acts, hook=True)

    tokens = load_tokenized_data(
        cfg.ctx_len,
        edited.tokenizer,
        cfg.dataset_repo,
        cfg.dataset_split,
    )
    cache = FeatureCache(
        edited, 
        submodule_dict, 
        batch_size=cfg.batch_size,
    )
    name = sae_dir.name
    cache.run(10_000_000, tokens)
    cache_save_dir = f"results/itda_cache/{name}"
    os.makedirs(cache_save_dir, exist_ok=True)

    cache.save_splits(
        n_splits=cfg.n_splits, 
        save_dir=cache_save_dir
    )
    cache.save_config(
        save_dir=cache_save_dir,
        cfg=cfg,
        model_name=model_name,
    )

if __name__ == "__main__":

    parser = ArgumentParser()
    #ctx len 256
    parser.add_arguments(CacheConfig, dest="options")
    args = parser.parse_args()
    cfg = args.options
    print(cfg)
    
    main(cfg, args)
