import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from full_sim.special_autoencoders import AutoencoderConfig, AutoencoderLatents
from nnsight import LanguageModel
from simple_parsing import ArgumentParser
from tqdm import tqdm


def merge_scores(SCORES_FOLDER):
    temp_data = pd.read_csv(f"{SCORES_FOLDER}all_data.csv")
    # check if there's more than 500 rows
    if len(temp_data) > 500:
        return temp_data
    else:
        all_files = glob.glob(os.path.join(SCORES_FOLDER, "*.txt"))
        all_results = []
        for file in all_files:
            with open(file, "r") as f:
                data = json.load(f)
                all_results.append(data)
        
        all_data = pd.DataFrame(all_results)
        all_data.to_csv(f"{SCORES_FOLDER}all_data.csv",index=False)
        
        return all_data

def get_predicted_features_top32(selected_data, features, active_features_tensor, autoencoder_features):
    features_in = selected_data["feature"].isin(features)
    wanted_predicted_activations = selected_data[features_in]["activation"].tolist()
    pre_predicted_features_tensor = active_features_tensor.clone()
    wanted_predicted_activations = [0 if np.isnan(x) else x for x in wanted_predicted_activations]
    #non-zero elements
    features_in_tensor = selected_data[features_in]["feature"].unique().tolist()
    non_zero_before = torch.nonzero(pre_predicted_features_tensor).squeeze().tolist()
    pre_predicted_features_tensor[features_in_tensor] = torch.tensor(wanted_predicted_activations,
                                                            dtype=autoencoder_features.dtype,
                                                            device=autoencoder_features.device)
    top_features = torch.argsort(pre_predicted_features_tensor,descending=True)[:32]
    # overlap between top_features and non_zero_before
    #overlap = len(set(top_features.tolist()) & set(non_zero_before))
    #print(overlap/32)
    predicted_features_tensor = torch.zeros_like(pre_predicted_features_tensor,
                                                dtype=autoencoder_features.dtype,
                                                device=autoencoder_features.device)
    predicted_features_tensor[top_features] = pre_predicted_features_tensor[top_features]
    
    return predicted_features_tensor

def compute_loss_for_features(predicted_features_tensor, model, prompt, autoencoder, real_layer_activation, layer, device):
    
    with model.trace(prompt[:-1]):
        reconstructed = predicted_features_tensor@autoencoder.ae.W_dec + autoencoder.ae.b_dec
        skip = autoencoder.ae.W_skip.to(torch.float16)
        reconstructed += real_layer_activation[0,-1,:]@skip.mT
        model.gpt_neox.layers[layer].mlp.output[0,-1,:] = reconstructed
        simulated_output = model.output.save()

    loss = torch.nn.functional.cross_entropy(simulated_output.logits[0,-1,:],
                                            torch.tensor(prompt[-1],device=device),
                                            reduction="mean")
    return loss.item()


def main(args):
    
    layer = 6
    sae_type = "SkipTranscoder"

    all_scores = pd.read_csv(f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/transcoders/results/scores_layer{layer}_recall_{sae_type}.csv")
    
    if args.model_size == "8b":
        predictions_folder= f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/transcoders/results/kl_div_activations/8b/{sae_type}/"
        
    elif args.model_size == "70b":
        predictions_folder= f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/transcoders/results/kl_div_activations/70b/{args.sae_type}/"
        
    device = "cuda"
    if args.cheating:
        all_data = pd.read_csv("cleaned_data_cheating.csv")
    else:
        all_data = pd.read_csv("cleaned_data.csv")
    if args.normalized:
        all_data["activation"] = all_data["normalized_activation"]
    else:
        all_data["activation"] = all_data["predicted_activation"]
    
    

    with open(f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/transcoders/results/saved_tokens.json", "r") as f:
        text_tokens = json.load(f)
    
    model = LanguageModel("EleutherAI/pythia-160m", device_map="cuda", dispatch=True,torch_dtype="float16")

    transcoder=True
    local=True
    path = "/mnt/ssd-1/nora/sae-ckpts/k32-skip-32k"

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
    sentence_groups = dict(tuple(all_data.groupby("sentence_idx")))
    print(args.fraction)
    for fraction in args.fraction:
        for i,prompt in tqdm(enumerate(prompts)):
            if i >= args.num_sentences:
                break
            if i in sentence_groups:
                selected_data = sentence_groups[i]
                if len(selected_data) < 30000:
                    continue
            else:
                continue
        
            actual_prompt = prompt[:-1]
            text = model.tokenizer.decode(actual_prompt)
            with model.trace(actual_prompt):
                real_layer_activation = model.gpt_neox.layers[layer].mlp.input.save()
                
                autoencoder_features = autoencoder.forward(real_layer_activation).save()
                output = model.output.save()
            ba_loss = torch.nn.functional.cross_entropy(output.logits[0,-1,:],torch.tensor(prompt[-1],device=device),reduction="mean")
            
            active_features_tensor = autoencoder_features.clone()[0,-1,:]
            
            re_loss = compute_loss_for_features(active_features_tensor,model, prompt, autoencoder, real_layer_activation, layer, device)
            
            if args.sample:                    # Do multiple random selections and average the loss
                num_features = max(int(fraction * len(unique_features)),1)
                features = np.random.choice(unique_features,num_features,replace=False)
                predicted_features_tensor = get_predicted_features_top32(selected_data, features, active_features_tensor, autoencoder_features)
                if predicted_features_tensor is None:

                    continue
                    
                loss = compute_loss_for_features(predicted_features_tensor, model, prompt, 
                                            autoencoder, real_layer_activation, layer, device)
                

                loss_prediction.append(loss)

            else:
                if args.random:
                    # random features
                    features = np.random.choice(unique_features,32,replace=False)
                    predicted_features_tensor = torch.zeros_like(active_features_tensor,dtype=autoencoder_features.dtype,device=autoencoder_features.device)
                    predicted_features_tensor[features] = active_features_tensor.nonzero()[0].to(autoencoder_features.dtype)
                elif args.zero_out:
                    num_features = max(int(fraction * len(unique_features)),1)
                    features = np.random.choice(unique_features,num_features,replace=False)
                    active_features_tensor[features] = 0
                    predicted_features_tensor = active_features_tensor

                else:
                    
                    if fraction != 0:
                        num_features = max(int(fraction * len(unique_features)),1)
                    else:
                        num_features = 0
                    
                    all_scores = all_scores.sort_values(by="balanced_accuracy", ascending=False)
                    features = all_scores["feature"][:num_features].tolist()
                    features.sort()
                    predicted_features_tensor = get_predicted_features_top32(selected_data, features, active_features_tensor, autoencoder_features)
                    if predicted_features_tensor is None:
                        continue
                loss = compute_loss_for_features(predicted_features_tensor, model, prompt,
                                            autoencoder, real_layer_activation, layer, device)
                #print(loss)
                loss_prediction.append(loss)
            loss_reconstruction.append(re_loss)
            #print(re_loss)
            loss_base.append(ba_loss.item())
            #print(ba_loss.item())
        
            del real_layer_activation, autoencoder_features, output, predicted_features_tensor
        # save the results
        name = f"{fraction}"
        if args.sample:
            name += "_sample"
        if args.random:
            name += "_random"
        if args.zero_out:
            name += "_zero_out"
        if args.normalized:
            name += "_normalized"
        if args.cheating:
            name += "_cheating"
        print("Name: ",name)
        print(f"Loss base: {np.mean(loss_base),np.median(loss_base)}")
        print(f"Loss reconstruction: {np.mean(loss_reconstruction),np.median(loss_reconstruction)}")
        print(f"Loss prediction: {np.mean(loss_prediction),np.median(loss_prediction)}")
        
        with open(f"{predictions_folder}ce_loss_{name}.json", "w") as f:
            json.dump({"loss_base":loss_base,"loss_reconstruction":loss_reconstruction,"loss_prediction":loss_prediction},f)

    
if __name__ == "__main__":
    parser = ArgumentParser()
    #parser.add_argument("--sentence_idx", type=int, default=0)
    parser.add_argument("--model_size", type=str, default="8b")
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--num_sentences", type=int, default=11000)
    parser.add_argument("--start_sentence", type=int, default=1000)
    parser.add_argument("--fraction", type=float, nargs="+", default=[1])
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--normalized", action="store_true")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--zero_out", action="store_true")
    parser.add_argument("--cheating", action="store_true")
    args = parser.parse_args()
    #sentence_idx = args.sentence_idx

    main(args)