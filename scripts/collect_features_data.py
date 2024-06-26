import torch
from transformer_lens import  HookedTransformer

import tqdm
from argparse import ArgumentParser
from sae_auto_interp.autoencoders.model import get_autoencoder
from sae_auto_interp.features.dataset import get_available_configs, get_batches, get_config
from pathlib import Path

import json

if __name__ == "__main__":
    parser = ArgumentParser()
    #TODO: Allow for a list of layers
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Layer to collect features from, if -1 will collect from all layers",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        choices="gpt2,meta-llama/Meta-Llama-3-8B"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--number_features",
        type=int,
        default=-1,
        help="Number of features to collect, if -1 will collect all features",
    )
    parser.add_argument(
        "--dataset_configuration",
        type=str,
        choices=get_available_configs(),
    )
    parser.add_argument(
        "--file_prefix",
        type=str,
        default="",
        help="Prefix to add to the file name",
    )
    parser.add_argument(
        "--path_autoencoder",
        type=str,
        default="/mnt/ssd-1/gpaulo/SAE-Zoology/saved_autoencoders",
        help="Path to the autoencoder files",
    )

    args = parser.parse_args()
    model_name = args.model
    device = args.device
    layer = args.layer
    number_features = args.number_features
    number_examples = args.number_examples
    config = args.dataset_configuration
    config = get_config(config)
    prefix = args.prefix
    if prefix != "":
        prefix = prefix + "_"
    autoencoder_path= args.path_autoencoder

    print("Loading the model")
    model = HookedTransformer.from_pretrained(
        model_name, center_writing_weights=False, device=device, dtype="bfloat16"
    )
    tokenizer = model.tokenizer

    print("Loading the autoencoders")
    if layer == -1:
        layers = list(range(12))
        autoencoders = {
            layer: get_autoencoder(model_name, layer, device,autoencoder_path) for layer in layers
        }
    else:
        layers = [layer]
        autoencoders = {layer: get_autoencoder(model_name, layer, device,autoencoder_path)}

    mini_batches = get_batches(config,tokenizer)
    batch_size = config["batch_size"]
    ## I think we always want all the features, this is more of a debug option
    number_features = number_features
    feature_indices = list(range(number_features))

    all_non_zero_features = {layer: [] for layer in layers}
    all_indices = {layer: [] for layer in layers}
    for minibatch in tqdm.tqdm(mini_batches):
        with torch.no_grad():
            
            #TODO: Cache only the activations for the layers we are interested in
            _, model_acts = model.run_with_cache(minibatch, remove_batch_dim=False)
            for layer in layers:
                layer_acts = model_acts[f"blocks.{layer}.hook_resid_post"]
                # Get the features of the autoencoder
                features = autoencoders[layer](layer_acts)
                if number_features == -1:
                    wanted_features = features
                else:
                    wanted_features = features[:, :, feature_indices]
                non_zero_features = wanted_features[wanted_features.abs() > 1e-5]
                indices = torch.nonzero(wanted_features.abs() > 1e-5)
                all_non_zero_features[layer].append([non_zero_features.cpu()])
                all_indices[layer].append([indices.cpu()])

                # Trying to have the memory not blow up
                del layer_acts
                del features
                del wanted_features
                del non_zero_features
                del indices
            del _
            del model_acts
        torch.cuda.empty_cache()

    if "/" in model_name:
        save_name = model_name.split("/")[1]

    # Make the folder to save the features
    feature_path = Path("saved_features") / model_name
    feature_path.mkdir(parents=True, exist_ok=True)

    for layer in layers:

        # Stacking the features and indices
        feature_layers = torch.stack(
            [torch.cat(sub_list, dim=0) for sub_list in all_non_zero_features[layer]],
            dim=0,
        )
        indice_layers = torch.stack(
            [torch.cat(sub_list, dim=0) for sub_list in all_indices[layer]], dim=0
        )
        feature_layers = feature_layers.ravel()
        new_indices = torch.zeros(
            indice_layers.shape[0] * indice_layers.shape[1], indice_layers.shape[2]
        )
        for i in range(indice_layers.shape[0]):
            old_indices = indice_layers[i]
            old_indices[:, 0] = old_indices[:, 0] + i * batch_size
            new_indices[
                i * indice_layers.shape[1] : (i + 1) * indice_layers.shape[1]
            ] = old_indices

        filename = feature_path / f"{prefix}layer_{layer}_features.pt"
        with open(filename, mode="wb") as f:
            torch.save(feature_layers, f)
        filename = feature_path / f"{prefix}layer_{layer}_indices.pt"
        with open(filename, mode="wb") as f:
            torch.save(new_indices, f)
        filename = feature_path / f"{prefix}layer_{layer}_config.json"
        with open(filename, mode="w") as f:
            json.dump(config, f)
