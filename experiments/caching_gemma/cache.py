from nnsight import LanguageModel
from simple_parsing import ArgumentParser
import torch
from delphi.autoencoders import load_gemma_autoencoders
from delphi.config import CacheConfig
from delphi.features import FeatureCache
from delphi.utils import load_tokenized_data
import os


l0_dict_mlp = {
    "16k": {0:50,
            1:56,
            2:33,
            3:55,
            4:66,
            5:46,
            6:46,
            7:47,
            8:55,
            9:40,
            10:49,
            11:34,
            12:42,
            13:40,
            14:41,
            15:45,
            16:37,
            17:41,
            18:36,
            19:38,
            20: 41,
            21:34,
            22:34,
            23:73,
            24:32,
            25:72,
            26:57,
            27:52,
            28:50,
            29:49,
            30:51,
            31:43,
            32:44,
            33:48,
            34:47,
            35:46,
            36:47,
            37:53,
            38:45,
            39:43,
            40:37,
            41:58

    },
    "131k":{
    20: 41,
    24: 33,
    28: 47,
    32: 40
}}
l0_dict_res = {
    "16k": {0:35,
            1:69,
            2:67,
            3:37,
            4:37,
            5:37,
            6:47,
            7:46,
            8:51,
            9:51,
        10:57,
        11:32,
        12:33,
        13:34,
            14:35,
            15:34,
            16:39,
            17:38,
            18:37,
            19:35,
        20: 36,
        21:36,
        22: 35, 
        23: 35,       
    24: 34,
    25: 34,
    26: 35,
    27:36,
    28: 37,
    29:38,
    30:37,
    31:35,
    32: 34,
    33:34,
    34:34,
    35:34,
    36:34,
    37:34,
    38:34,
    39:34,
    40:32,
    41:52
    },
    "131k": {0:30,
             1:33,
             2:36,
             3:46,
             4:51,
             5:51,
             6:66,#Doesnt work
             7:38,
             8:41,
             9:42,
             10:47,
             11:49,
             12:52,
             13:30,#Doesnt work
             14:56,
             15:55,
             16:35,
             17:35,
             18:34,
             19:32, 
             20:34,
             21:33,
             22:32,
             23:32,
    24: 55,
    25:54,
    26:32,
    27:33,
    28: 32,
    29:33,
    30:32,
    31:52,
    32: 51,
    33:51,
    34:51,
    35:51,
    36: 51,
    37:53,
    38:53,
    39:54,
    40: 49,
    41:45,
    }
}

def main(cfg: CacheConfig,args): 
    layers = args.layers
    size = args.size
    type = args.type
    name = args.name
    random = args.random
    model = LanguageModel("google/gemma-2-9b", device_map="cuda", dispatch=True,torch_dtype="float16")
    layers = [int(layer) for layer in layers.split(",")]
    if type == "res":
        dict_l0 = l0_dict_res
    elif type == "mlp":
        dict_l0 = l0_dict_mlp
    
    submodule_dict,model = load_gemma_autoencoders(
            model,
            layers,
            {layer: dict_l0[size][layer] for layer in layers},
            size,
            type,
            random
        )
    
    tokens = load_tokenized_data(
        cfg.ctx_len,
        model.tokenizer,
        cfg.dataset_repo,
        cfg.dataset_split,
        cfg.dataset_name    
    )

    cache = FeatureCache(
        model, 
        submodule_dict, 
        batch_size=cfg.batch_size,
    )

    cache.run(10000000, tokens)
    if name not in [""]:
        name = f"_{name}"
    if random:
        name = name + "_random"

    if not os.path.exists(f"raw_features/gemma/{size}{name}"):
        os.makedirs(f"raw_features/gemma/{size}{name}")

    cache.save_splits(
        n_splits=cfg.n_splits, 
        save_dir=f"raw_features/gemma/{size}{name}"
    )

    cache.save_config(
        save_dir=f"raw_features/gemma/{size}{name}",
        cfg=cfg,
        model_name="google/gemma-2-9b"
    )

if __name__ == "__main__":

    parser = ArgumentParser()
    #ctx len 256
    parser.add_arguments(CacheConfig, dest="options")
    parser.add_argument("--layers", type=str, default="23,27")
    parser.add_argument("--size", type=str, default="16k")
    parser.add_argument("--type", type=str, default="res")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--random", action="store_true")
    args = parser.parse_args()
    cfg = args.options

    main(cfg,args)
