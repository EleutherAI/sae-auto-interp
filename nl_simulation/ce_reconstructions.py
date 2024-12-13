import asyncio
import glob
import json
import os
import time
from functools import partial

import numpy as np
import orjson
import pandas as pd
import torch
from nl.special_autoencoders import AutoencoderConfig, AutoencoderLatents
from nnsight import LanguageModel
from simple_parsing import ArgumentParser
from tqdm import tqdm


def read_predictions(predictions_folder):
    data = pd.read_csv(f"{predictions_folder}all_data.csv")
    return data

def get_predicted_features_top32(selected_data, features, active_features_tensor, autoencoder_features):
    # get the predicted activations for the wanted features
    wanted_predicted_activations = selected_data[selected_data["feature"].isin(features)]["predicted_activation"].tolist()
    # create a tensor with the predicted activations
    pre_predicted_features_tensor = active_features_tensor.clone()

    if len(features) != len(wanted_predicted_activations):
        return None
    
    # set the predicted activations for the wanted features
    pre_predicted_features_tensor[features] = torch.tensor(wanted_predicted_activations,
                                                            dtype=autoencoder_features.dtype,
                                                            device=autoencoder_features.device)

    # find the 32 most active latents
    top_features = torch.argsort(pre_predicted_features_tensor,descending=True)[:32]
    # create the tensor with the predicted activations
    predicted_features_tensor = torch.zeros_like(pre_predicted_features_tensor,
                                                dtype=autoencoder_features.dtype,
                                                device=autoencoder_features.device)
    predicted_features_tensor[top_features] = pre_predicted_features_tensor[top_features]
    
    return predicted_features_tensor

def compute_loss_for_features(predicted_features_tensor, model, prompt, autoencoder, real_layer_activation, layer, device):
    
    with model.trace(prompt[:-1]):
        # reconstruct the activations
        reconstructed = predicted_features_tensor@autoencoder.ae.W_dec + autoencoder.ae.b_dec
        # add the skip connections
        skip = autoencoder.ae.W_skip.to(torch.float16)
        reconstructed += real_layer_activation[0,-1,:]@skip.mT
        # set the reconstructed activations to the model
        model.gpt_neox.layers[layer].mlp.output[0,-1,:] = reconstructed
        # get the simulated output
        simulated_output = model.output.save()

    loss = torch.nn.functional.cross_entropy(simulated_output.logits[0,-1,:],
                                            torch.tensor(prompt[-1],device=device),
                                            reduction="mean")
    return loss.item()


def main(args):
    
    layer = 6
    sae_type = "SkipTranscoder"

    all_scores = pd.read_csv(f"results/scores_layer{layer}_recall_{sae_type}.csv")
    
    predictions_folder= "results/activations/8b/"
        
    device = "cuda"
    all_data = []
    for i in tqdm(range(args.start_sentence,args.start_sentence+args.num_sentences)):
        data = read_predictions(f"{predictions_folder}{i}")
        data["sentence_idx"] = i
        all_data.append(data)
    
    all_data = pd.concat(all_data)
    all_data["text"] = all_data["text"].str.replace("<<","")
    all_data["text"] = all_data["text"].str.replace(">>","")
    
    with open("results/saved_tokens.json", "r") as f:
        text_tokens = json.load(f)
    
    model = LanguageModel("EleutherAI/pythia-160m", device_map="cuda", dispatch=True,torch_dtype="float16")

    transcoder=True
    local=True
    path = "/mnt/ssd-1/nora/sae/k32-skip-32k"

    config = AutoencoderConfig(
        model_name_or_path=path,
        autoencoder_type="SAE",
        device=device,
        kwargs={"local": local, "transcoder": transcoder}
    )
    autoencoder = AutoencoderLatents.from_pretrained(config,hookpoint=f"layers.{layer}.mlp")
    autoencoder.ae = autoencoder.ae.to(device)    
    prompts = text_tokens.values()
    loss_base = []
    loss_reconstruction = []
    loss_prediction = []
    unique_features = all_data["feature"].unique()
    for i,prompt in tqdm(enumerate(prompts)):
        if i >= args.num_sentences:
            break
        
        actual_prompt = prompt[:-1]
        with model.trace(actual_prompt):
            real_layer_activation = model.gpt_neox.layers[layer].mlp.input.save()
            
            autoencoder_features = autoencoder.forward(real_layer_activation).save()
            output = model.output.save()
        base_loss = torch.nn.functional.cross_entropy(output.logits[0,-1,:],torch.tensor(prompt[-1],device=device),reduction="mean")
        
        active_features_tensor = autoencoder_features.clone()[0,-1,:]
        
        reconstruction_loss = compute_loss_for_features(active_features_tensor,model, prompt, autoencoder, real_layer_activation, layer, device)
        
        if args.random:    
            # Do multiple random selections and average the loss              
            num_trials = 5
            trial_losses = []
            num_features = max(int(args.fraction * len(unique_features)),1)
            selected_data = all_data[all_data["sentence_idx"]==i]
            for _ in range(num_trials):
                features = np.random.choice(unique_features,num_features,replace=False)
                predicted_features_tensor = get_predicted_features_top32(selected_data, features, active_features_tensor, autoencoder_features)
                if predicted_features_tensor is None:

                    continue
                    
                loss = compute_loss_for_features(predicted_features_tensor, model, prompt, 
                                            autoencoder, real_layer_activation, layer, device)
                trial_losses.append(loss)

            loss_prediction.append(np.mean(trial_losses))

        else:
            if args.fraction == 1:
                # select the data from this sentence
                selected_data = all_data[all_data["sentence_idx"]==i]
                
                # get the predictions
                predicted_activations = selected_data["predicted_activation"].tolist()
                predicted_activations = torch.tensor(predicted_activations,dtype=autoencoder_features.dtype,device=autoencoder_features.device)

                # find the 32 most active latents
                top_features_idx = torch.argsort(predicted_activations,descending=True)[:32].tolist()
                features = selected_data["feature"][top_features_idx].tolist()
                
                # create the tensor with the predicted activations
                predicted_features_tensor = torch.zeros_like(active_features_tensor,dtype=autoencoder_features.dtype,device=autoencoder_features.device)
                predicted_features_tensor[features] = predicted_activations[top_features_idx]

            elif args.fraction == -1:
                # all zeros
                predicted_features_tensor = torch.zeros_like(active_features_tensor,dtype=autoencoder_features.dtype,device=autoencoder_features.device)
            elif args.fraction == -2:
                # random latents
                features = np.random.choice(unique_features,32,replace=False)
                predicted_features_tensor = torch.zeros_like(active_features_tensor,dtype=autoencoder_features.dtype,device=autoencoder_features.device)
                predicted_features_tensor[features] = active_features_tensor.nonzero()[0].to(autoencoder_features.dtype)
            else:
                selected_data = all_data[all_data["sentence_idx"]==i]
                
                if args.fraction != 0:
                    num_features = max(int(args.fraction * len(unique_features)),1)
                else:
                    num_features = 0
                
                # get the top scoring features correspondign to the wanted fraction
                all_scores = all_scores.sort_values(by="balanced_accuracy", ascending=False)
                features = all_scores["feature"][:num_features].tolist()
                features.sort()
                
                # create the tensor with the predicted activations
                predicted_features_tensor = get_predicted_features_top32(selected_data, features, active_features_tensor, autoencoder_features)
                
                if predicted_features_tensor is None:
                    continue
            
            loss = compute_loss_for_features(predicted_features_tensor, model, prompt,
                                           autoencoder, real_layer_activation, layer, device)
            loss_prediction.append(loss)
        loss_reconstruction.append(reconstruction_loss)
        loss_base.append(base_loss.item())

      
        del real_layer_activation, autoencoder_features, output, predicted_features_tensor
    print("Fraction: ",args.fraction," Random: ",args.random)
    print(f"Loss prediction: {np.mean(loss_prediction),np.median(loss_prediction)}")
    # save the results
    name = f"{args.fraction}"
    if args.random:
        name += "_random"
    with open(f"{predictions_folder}ce_loss_{name}.json", "w") as f:
       json.dump({"loss_base":loss_base,"loss_reconstruction":loss_reconstruction,"loss_prediction":loss_prediction},f)

    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_sentences", type=int, default=1000)
    parser.add_argument("--start_sentence", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=1)
    parser.add_argument("--random", action="store_true")
    args = parser.parse_args()
    
    main(args)