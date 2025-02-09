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
from sparsify import Sae
from pathlib import Path
import json
import os


def main(cfg: CacheConfig, args): 
    k = args.k
    expansion = args.expansion
    transcode = args.transcode
    pkm = args.pkm
    model_name = args.model_name
    sae_dir = Path(args.sae_dir)

    for sae_dir in sae_dir.glob("*"):
        config_path = sae_dir / "config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        if config["sae"]["expansion_factor"] != expansion:
            continue
        if config["sae"]["encoder_pkm"] != pkm:
            continue
        if config["transcode"] != transcode:
            continue
        if config["sae"]["k"] != k:
            continue
        if model_name not in config["model"]:
            continue
        model_name = config["model"]
        break
    else:
        raise FileNotFoundError("No matching SAE found")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model = NNsight(model)
    model.tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = LanguageModel(model_name, device_map="auto",
    #                     #   dispatch=True,
    #                       torch_dtype=torch.bfloat16,
    #                       trust_remote_code=True
    #                       )
    
    # submodule_dict, model = load_eai_autoencoders(
    #     model,
    #     config["layers"],
    #     str(sae_dir.resolve()),
    #     module="mlp" if transcode else "res",
    #     randomize=False,
    #     k=k
    # )
    submodule_dict = {}

    device = "cuda:0"
    for hookpoint in config["hookpoints"]:
        weight_dir = sae_dir / hookpoint
        
        sae = Sae.load_from_disk(weight_dir, device=device).to(dtype=model.dtype)
        if not hookpoint.startswith(model.base_model_prefix):
            hookpoint = f"{model.base_model_prefix}.{hookpoint}"
        
        def _forward(sae, k,x):
            encoded = sae.pre_acts(x)
            if k is not None:
                trained_k = k
            else:
                trained_k = sae.cfg.k
            topk = TopK(trained_k, postact_fn=ACTIVATIONS_CLASSES["Identity"]())
            return topk(encoded)
        
        atoms = hookpoint.split(".")
        submodule = model
        for atom in atoms:
            submodule = getattr(submodule, atom)
        submodule.ae = AutoencoderLatents(
            sae, partial(_forward, sae, k), width=sae.encoder.weight.shape[0],
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
    # name=""
    # name += model_name.replace("/", "-")
    # name += f"_x{expansion}"
    # name += f"_k-{k}"
    # name += "_trans" if transcode else ""
    # name += "_pkm" if pkm else ""
    name = sae_dir.name
    cache.run(10_000_000, tokens)
    cache_save_dir = f"results/sae_pkm/{name}"
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
    parser.add_argument("--sae_dir", type=str, default="../halutsae/pkm_saes")
    parser.add_argument("--model_name", type=str, default="pythia-160m")
    parser.add_argument("--expansion", type=int, default=64)
    parser.add_argument("--transcode", type=bool, default=False)
    parser.add_argument("--pkm", type=bool, required=True)
    parser.add_argument("--k", type=int, default=32)
    args = parser.parse_args()
    cfg = args.options
    print(cfg)
    
    main(cfg, args)
