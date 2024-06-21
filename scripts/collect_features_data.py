import torch
from transformer_lens import utils, HookedTransformer
import datasets
import tqdm
from argparse import ArgumentParser
from sae_auto_interp.models.load import get_autoencoder
import os
from pathlib import Path
token = os.getenv("HF_TOKEN")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--layer",
        type=int, choices=list(range(12))+[-1],default=-1,
    )
    parser.add_argument(
        "--model",
        type=str, default="gpt2",
    )
    parser.add_argument(
        "--device",
        type=str, default="cuda:0",
    )
    parser.add_argument(
        "--number_features",
        type=int, default=131072,
    )
    parser.add_argument(
        "--number_examples",
        type=int, default=1000,
    )
    
    args = parser.parse_args()
    model_name = args.model
    device = args.device
    layer = args.layer
    number_features = args.number_features
    number_examples = args.number_examples

    model_name = "meta-llama/Meta-Llama-3-8B"
    number_features = 100
    layer = 0
    #model = AutoModelForCausalLM.from_pretrained(model_name,cache_dir="/mnt/ssd-1/hf_cache/hub",token=token)
    
    print("Loading the model")
    model = HookedTransformer.from_pretrained(model_name, center_writing_weights=False,device=device)

    print("Loading the autoencoders")
    if layer == -1:
        layers = list(range(12))
        autoencoders = {layer:get_autoencoder(model_name,layer,device) for layer in layers}
    else:
        layers = [layer]
        autoencoders = {layer:get_autoencoder(model_name,layer,device)}
    

    print("Loading dataset")
    ## This is hardcoded for now but I should make a dataset loader
    dataset = datasets.load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train")
    
    
    keep_examples = number_examples
    dataset = dataset.select(range(keep_examples))
    tokenized_data = utils.tokenize_and_concatenate(dataset, model.tokenizer, max_length=64)
    all_tokens = tokenized_data["tokens"]

    ## This is hardcoded because this is what can fit in our gpus, can make this dynamic later
    batch_size = 8  
    mini_batches = all_tokens.split(batch_size)
    mini_batches = [batch for batch in mini_batches]
    mini_batches = mini_batches[:-1]


    ## I think we always want all the features, this is more of a debug option
    number_features = number_features
    feature_indices = list(range(number_features))
    
    all_non_zero_features= {layer:[] for layer in layers}
    all_indices = {layer:[] for layer in layers}
    for minibatch in tqdm.tqdm(mini_batches):
            with torch.no_grad():
                _,model_acts = model.run_with_cache(minibatch, remove_batch_dim=False)
                for layer in layers:
                    layer_acts = model_acts[f"blocks.{layer}.hook_resid_post"]
                    #Get the features of the autoencoder
                    features = autoencoders[layer].encode(layer_acts)
                    if number_features == 131072:
                        wanted_features = features
                    else:
                        wanted_features = features[:,:,feature_indices]
                    non_zero_features = wanted_features[wanted_features.abs()>1e-5]
                    indices = torch.nonzero(wanted_features.abs()>1e-5)
                    all_non_zero_features[layer].append([non_zero_features.cpu()])
                    all_indices[layer].append([indices.cpu()])
                    #Trying to have the memory not blow up
                    del(layer_acts)
                    del(features)
                    del(wanted_features)
                    del(non_zero_features)
                    del(indices)
                del(_)
                del(model_acts)
            torch.cuda.empty_cache()


    if "/" in model_name:
        model_name = model_name.split("/")[1]    

    feature_path = (
        Path("features")
        / model_name
    )
    feature_path.mkdir(parents=True, exist_ok=True)

    for layer in layers:
        feature_layers =  torch.stack([torch.cat(sub_list, dim=0) for sub_list in all_non_zero_features[layer]], dim=0)
        indice_layers =  torch.stack([torch.cat(sub_list, dim=0) for sub_list in all_indices[layer]], dim=0)
        feature_layers = feature_layers.ravel()
        new_indices = torch.zeros(indice_layers.shape[0]*indice_layers.shape[1],indice_layers.shape[2])
        for i in range(indice_layers.shape[0]):
            old_indices = indice_layers[i]
            old_indices[:,0] = old_indices[:,0]+i*batch_size
            new_indices[i*indice_layers.shape[1]:(i+1)*indice_layers.shape[1]] = old_indices
        
        filename = feature_path+f"/layer_{layer}_features.pt"
        with open(filename, mode="wb") as f:
            torch.save(feature_layers, f)
        filename = f"features_llama/layer_{layer}_indices.pt"
        with open(filename, mode="wb") as f:
            torch.save(new_indices, f)
    
