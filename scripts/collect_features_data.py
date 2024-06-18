import torch
from transformer_lens import utils, HookedTransformer
import datasets
import tqdm
from argparse import ArgumentParser
from sae_auto_interp.sae_models.load import get_autoencoder





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
    
    args = parser.parse_args()
    model = args.model
    device = args.device
    layer = args.layer
    number_features = args.number_features

    
    print("Loading the model")
    model = HookedTransformer.from_pretrained(model, center_writing_weights=False,device=device)

    print("Loading the autoencoders")
    if layer == -1:
        layers = list(range(12))
        autoencoders = {layer:get_autoencoder(model,layer,device) for layer in layers}
    else:
        layers = [layer]
        autoencoders = {layer:get_autoencoder(model,layer,device)}
    


    ## All of this is hardcoded for now but I should make a dataset loader
    dataset = datasets.load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train")
    keep_examples = 1000
    dataset = dataset.select(range(keep_examples))
    tokenized_data = utils.tokenize_and_concatenate(dataset, model.tokenizer, max_length=64)
    all_tokens = tokenized_data["tokens"]

    ## This is hardcoded because this is what can fit in our gpus, can make this dynamic later
    batch_size = 128  
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
                    features = autoencoders[layer].encode(layer_acts)[0]
                    if number_features == 131072:
                        wanted_features = features
                    else:
                        wanted_features = features[:,:,feature_indices]
                    non_zero_features = wanted_features[wanted_features.abs()>1e-5]
                    indices = torch.nonzero(wanted_features.abs()>1e-5)
                    all_non_zero_features[layer].append([non_zero_features.cpu()])
                    all_indices[layer].append([indices.cpu()])
                    del(layer_acts)
                    del(features)
                    del(wanted_features)
                    del(non_zero_features)
                    del(indices)
                del(_)
                del(model_acts)
            torch.cuda.empty_cache()


    for layer in layers:
        feature_layers =  torch.stack([torch.cat(sub_list, dim=0) for sub_list in all_non_zero_features[layer]], dim=0)
        indice_layers =  torch.stack([torch.cat(sub_list, dim=0) for sub_list in all_indices[layer]], dim=0)
        feature_layers = feature_layers.ravel()
        new_indices = torch.zeros(indice_layers.shape[0]*indice_layers.shape[1],indice_layers.shape[2])
        for i in range(indice_layers.shape[0]):
            old_indices = indice_layers[i]
            old_indices[:,0] = old_indices[:,0]+i*batch_size
            new_indices[i*indice_layers.shape[1]:(i+1)*indice_layers.shape[1]] = old_indices
        
        filename = f"../features/layer_{layer}_features.pt"
        with open(filename, mode="wb") as f:
            torch.save(feature_layers, f)
        filename = f"../features/layer_{layer}_indices.pt"
        with open(filename, mode="wb") as f:
            torch.save(new_indices, f)
    
