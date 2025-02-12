import argparse
import asyncio
import json
import os
import random
import numpy as np
from functools import partial

import torch
from nnsight import LanguageModel
from special_autoencoders import AutoencoderConfig, AutoencoderLatents
from special_fuzz import FuzzingScorer
from torch.nn import KLDivLoss

from sae_auto_interp.clients import Offline
from sae_auto_interp.features import Example, Feature, FeatureRecord
from sae_auto_interp.pipeline import Pipe, Pipeline, process_wrapper


async def main(args):
    RESULTS_FOLDER = f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/iterative_generation/percentage/"
    device = "cuda"
   
    model = LanguageModel("google/gemma-2-9b", device_map=device, dispatch=True,torch_dtype="float16")

    config = AutoencoderConfig(
        model_name_or_path="google/gemma-scope-9b-pt-res",
        autoencoder_type="CUSTOM",
        device="cuda",
        hookpoints=[f"layer_11/width_131k/average_l0_49"],
        kwargs={"custom_name": "gemmascope"}
    )
    autoencoder = AutoencoderLatents.from_pretrained(config,hookpoint=f"layer_11/width_131k/average_l0_49")
    autoencoder.ae = autoencoder.ae.to(device)    

    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    kl_div = KLDivLoss(reduction="batchmean")

    percentages = [0.01,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.99]
    all_simulated_outputs_sample = []
    all_simulated_outputs_top = []
    all_kl_div_simulated_top = []
    all_kl_div_simulated_sample = []
    all_active_features = []
    all_active_features_top = []
    all_active_features_sample = []
    
    for seedprompt in ["The sun was", "I'm a", "This text was", "The quick brown" , "There are always people"]:
        prompt = seedprompt
        normal_outputs = []
        simulated_outputs_top = {}
        active_features_top = {}
        simulated_outputs_sample = {}
        active_features_sample = {}
        kl_div_simulated_sample = {}
        kl_div_simulated_top = {}
        correct_features = []
        for i in range(15):
            
            with model.trace(prompt):

                real_layer_activation = model.model.layers[11].output[0].save()
                autoencoder_reconstruction = autoencoder.ae.forward(real_layer_activation).save()
                autoencoder_features = autoencoder.ae.encode(autoencoder_reconstruction).save()
                output = model.output.save()

            #greedy decoding

            real_token = output.logits[0].argmax(dim=-1)[-1]
            normal_outputs.append(prompt + model.tokenizer.decode(real_token))

            
            active_features = autoencoder_features[0,-1,:]
            non_zero_features = active_features!=0
            active_features_idx = torch.nonzero(non_zero_features).squeeze().tolist()
            active_features = active_features[non_zero_features]
            # sort active_features tensor
            _,sorted_active_features = torch.sort(active_features, descending=True)
            # sort active_features_idx, using the sorted_active_features    
            sorted_active_features_idx = [active_features_idx[i] for i in sorted_active_features]
            correct_features.append(sorted_active_features_idx)
            prediction_per_percentage_top = {}
            prediction_per_percentage_sample = {}
            kl_div_per_percentage_top = {}
            kl_div_per_percentage_sample = {}
            active_per_percentage_top = {}
            active_per_percentage_sample = {}

            random_permutation = torch.randperm(len(sorted_active_features_idx))
            for percentage in percentages:
                quantity = max(1,int(len(sorted_active_features_idx)*percentage))
                # top percentage of active features
                selected_features_top = sorted_active_features_idx[:quantity]
                # sample percentage of active features
                sampled_features_idx = random_permutation[:quantity]
                selected_features_sample = torch.tensor(sorted_active_features_idx)[sampled_features_idx].tolist()

                active_per_percentage_top[percentage] = selected_features_top
                active_per_percentage_sample[percentage] = selected_features_sample

                random_filler_features = torch.randperm(131072)[:len(active_features)].tolist()
                selected_features_top.extend(random_filler_features)
                selected_features_sample.extend(random_filler_features)
                
                selected_features_top = selected_features_top[:len(active_features)]
                selected_features_sample = selected_features_sample[:len(active_features)]

                # create a tensor like autoencoder_features but with the selected features
                selected_features_tensor_sample = autoencoder_features.clone()
                selected_features_tensor_sample[0,-1,:] = torch.zeros_like(selected_features_tensor_sample[0,-1,:])
                selected_features_tensor_sample[0,-1,selected_features_sample] = active_features[sorted_active_features]
                selected_features_tensor_sample = selected_features_tensor_sample.to(device)

                # top tensor
                selected_features_tensor_top = autoencoder_features.clone()
                selected_features_tensor_top[0,-1,:] = torch.zeros_like(selected_features_tensor_top[0,-1,:])
                selected_features_tensor_top[0,-1,selected_features_top] = active_features[sorted_active_features]
                selected_features_tensor_top = selected_features_tensor_top.to(device)


                with model.trace(prompt):
                    reconstructed_top = autoencoder.ae.decode(selected_features_tensor_top)
                    model.model.layers[11].output[0][0,1:,:] = reconstructed_top[0,1:,:]
                    simulated_output_top = model.output.save()

                with model.trace(prompt):   
                    reconstructed_sample = autoencoder.ae.decode(selected_features_tensor_sample)
                    model.model.layers[11].output[0][0,1:,:] = reconstructed_sample[0,1:,:]
                    simulated_output_sample = model.output.save()

            
                #greedy decoding
                simulated_token_sample = simulated_output_sample.logits[0].argmax(dim=-1)[-1]
                
                simulated_token_top = simulated_output_top.logits[0].argmax(dim=-1)[-1]
                
                kl_div_sample = kl_div(simulated_output_sample.logits[0].log_softmax(dim=-1),output.logits[0].softmax(dim=-1)).item()
                kl_div_top = kl_div(simulated_output_top.logits[0].log_softmax(dim=-1),output.logits[0].softmax(dim=-1)).item()

                prediction_per_percentage_sample[percentage] = model.tokenizer.decode(simulated_token_sample)
                prediction_per_percentage_top[percentage] = model.tokenizer.decode(simulated_token_top)
                kl_div_per_percentage_sample[percentage] = kl_div_sample
                kl_div_per_percentage_top[percentage] = kl_div_top
           
           
            for percentage in percentages:
                if percentage not in simulated_outputs_sample:
                    simulated_outputs_sample[percentage] = []
                if percentage not in simulated_outputs_top:
                    simulated_outputs_top[percentage] = []
                if percentage not in kl_div_simulated_sample:
                    kl_div_simulated_sample[percentage] = []
                if percentage not in kl_div_simulated_top:
                    kl_div_simulated_top[percentage] = []
                if percentage not in active_features_top:
                    active_features_top[percentage] = []
                if percentage not in active_features_sample:
                    active_features_sample[percentage] = []
                simulated_outputs_sample[percentage].append(prediction_per_percentage_sample[percentage])
                simulated_outputs_top[percentage].append(prediction_per_percentage_top[percentage])
                kl_div_simulated_sample[percentage].append(kl_div_per_percentage_sample[percentage])
                kl_div_simulated_top[percentage].append(kl_div_per_percentage_top[percentage])
                active_features_top[percentage].append(active_per_percentage_top[percentage])
                active_features_sample[percentage].append(active_per_percentage_sample[percentage])
            
        
            print(f"KL divergence top (99%): {kl_div_simulated_top[0.99][-1]}")
            print(f"KL divergence top (50%): {kl_div_simulated_top[0.5][-1]}")
            print(f"KL divergence top (1%): {kl_div_simulated_top[0.01][-1]}")
            print(f"KL divergence sample (99%): {kl_div_simulated_sample[0.99][-1]}")
            print(f"KL divergence sample (50%): {kl_div_simulated_sample[0.5][-1]}")
            print(f"KL divergence sample (1%): {kl_div_simulated_sample[0.01][-1]}")

            print(prompt + simulated_outputs_sample[0.99][-1])
            print(prompt + simulated_outputs_sample[0.5][-1])
            print(prompt + simulated_outputs_sample[0.01][-1])
            print(prompt + simulated_outputs_top[0.99][-1])
            print(prompt + simulated_outputs_top[0.5][-1])
            print(prompt + simulated_outputs_top[0.01][-1])

            prompt = prompt + model.tokenizer.decode(real_token)
        all_simulated_outputs_sample.append(simulated_outputs_sample)
        all_simulated_outputs_top.append(simulated_outputs_top)
        all_kl_div_simulated_sample.append(kl_div_simulated_sample)
        all_kl_div_simulated_top.append(kl_div_simulated_top)
        all_active_features.append(correct_features)
        all_active_features_top.append(active_features_top)
        all_active_features_sample.append(active_features_sample)

    with open(f"{RESULTS_FOLDER}/iterative_generation_outputs.json", "w") as f:
        json.dump({"simulated_outputs_sample": all_simulated_outputs_sample, "simulated_outputs_top": all_simulated_outputs_top}, f)
    with open(f"{RESULTS_FOLDER}/iterative_generation_kl_div.json", "w") as f:
        json.dump({"kl_div_simulated_sample": all_kl_div_simulated_sample, "kl_div_simulated_top": all_kl_div_simulated_top}, f)
    with open(f"{RESULTS_FOLDER}/iterative_generation_active_features.json", "w") as f:
        json.dump({"all_active_features": all_active_features, "active_features_sample": all_active_features_sample, "active_features_top": all_active_features_top}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="8b")
    args = parser.parse_args()
    asyncio.run(main(args))
