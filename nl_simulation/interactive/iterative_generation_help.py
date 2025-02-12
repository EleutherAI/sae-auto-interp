import argparse
import asyncio
import json
import os
import random
import numpy as np
from functools import partial

import torch
from nnsight import LanguageModel
from lib.autoencoders import AutoencoderConfig, AutoencoderLatents
from lib.fuzz import FuzzingScorer
from torch.nn import KLDivLoss

from sae_auto_interp.clients import Offline
from sae_auto_interp.features import Example, Feature, FeatureRecord
from sae_auto_interp.pipeline import Pipe, Pipeline, process_wrapper


def get_feature_records(prompt,explanations,model,active_features):
    feature_records = []
    tokens = model.tokenizer.encode(prompt)
    activations = torch.zeros(len(tokens))
    for i in active_features:
        if str(i) not in explanations:
            continue
        explanation = explanations[str(i)]
        feature = Feature(f".model.layer_11", i)
        example = Example(tokens, activations)
        feature_record = FeatureRecord(feature)
        feature_record.explanation = explanation
        feature_record.test = [example]
        feature_records.append(feature_record)
        
    random_features = np.random.choice(np.arange(131072), size=1000, replace=False)
    for i in random_features:
        if str(i) not in explanations:
            continue
        explanation = explanations[str(i)]
        feature = Feature(f".model.layer_11", i)
        example = Example(tokens, activations)
        feature_record = FeatureRecord(feature)
        feature_record.explanation = explanation
        feature_record.test = [example]
        feature_records.append(feature_record)
    return feature_records

async def feature_generator(feature_records):
    for feature_record in feature_records:
            yield feature_record


async def get_prediction(prompt,explanations,client,model,active_features):
    feature_records = get_feature_records(prompt,explanations,model,active_features)

    global_result = {}
    def scorer_postprocess(result):
         probability = result.score[0].probability
         prediction = result.score[0].prediction
         feature = str(result.record.feature).split("feature")[1]
         global_result[feature] = [probability,prediction]
         
    fuzz_scorer = FuzzingScorer(client, model.tokenizer, finetuned=False)
    scorer_pipe = Pipe(process_wrapper(fuzz_scorer, postprocess=partial(scorer_postprocess)))
    pipeline = Pipeline(
        feature_generator(feature_records),
        scorer_pipe,
    )
    await pipeline.run(100000)
    return global_result


async def main(args):
    if args.model_size == "8b":
        client = Offline("meta-llama/Meta-Llama-3.1-8B-Instruct",max_memory=0.8,max_model_len=5120,num_gpus=4,batch_size=5000)
        RESULTS_FOLDER = f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/iterative_generation/8b-help/"
    elif args.model_size == "70b":
        client = Offline("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",max_memory=0.8, max_model_len=5120,num_gpus=2)   
        RESULTS_FOLDER = f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/iterative_generation/70b-help/"
    device = "cuda:4"
   
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


    with open(f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/explanations_131k/model.layers.11_feature.json", "r") as f:
            #load json file
            quantile_explanations = json.load(f)

    with open(f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/explanations_131k_top/model.layers.11_feature.json", "r") as f:
            top_explanations = json.load(f)

    all_normal_outputs = []
    all_reconstructed_outputs = []
    all_simulated_outputs_quantile = []
    all_simulated_outputs_top = []
    all_random_outputs = []
    all_kl_div_reconstructed = []
    all_kl_div_simulated_quantile = []
    all_kl_div_simulated_top = []
    all_kl_div_random = []	
    all_fraction_correct_quantile = []
    all_fraction_correct_top = []
    all_fraction_correct_active_quantile = []
    all_fraction_correct_active_top = []
    all_correct_active_top = []
    all_correct_active_quantile = []
    for seedprompt in ["The sun was", "I'm a", "This text was", "The quick brown" , "There are always people"]:
        prompt = seedprompt

        normal_outputs = []
        reconstructed_outputs = []
        simulated_outputs_quantile = []
        simulated_outputs_top = []
        random_outputs = []
        running_kl_div_reconstructed = []
        running_kl_div_simulated_quantile = []
        running_kl_div_simulated_top = []
        running_kl_div_random = []
        running_fraction_correct_quantile = []
        running_fraction_correct_top = []
        running_fraction_correct_active_quantile = []
        running_fraction_correct_active_top = []
        correct_active_top = []
        correct_active_quantile = []
        for i in range(15):
            

            with model.trace(prompt):

                real_layer_activation = model.model.layers[11].output[0].save()
                autoencoder_reconstruction = autoencoder.ae.forward(real_layer_activation).save()
                autoencoder_features = autoencoder.ae.encode(autoencoder_reconstruction).save()
                output = model.output.save()

            #greedy decoding

            real_token = output.logits[0].argmax(dim=-1)[-1]
            normal_outputs.append(model.tokenizer.decode(real_token))

            with model.trace(prompt):
                model.model.layers[11].output[0][0,1:,:] = autoencoder_reconstruction[0,1:,:]
                reconstructed_output = model.output.save()

            #greedy decoding
            reconstructed_token = reconstructed_output.logits[0].argmax(dim=-1)[-1]
            reconstructed_outputs.append(model.tokenizer.decode(reconstructed_token))
            active_features = autoencoder_features[0,-1,:]
            non_zero_features = active_features!=0
            active_features_idx = torch.nonzero(non_zero_features).squeeze().tolist()
            active_features = active_features[non_zero_features]
            # sort active_features tensor
            _,sorted_active_features = torch.sort(active_features, descending=True)
            # sort active_features_idx, using the sorted_active_features    
            sorted_active_features_idx = [active_features_idx[i] for i in sorted_active_features]
            
            result_quantile = await get_prediction(prompt,quantile_explanations,client,model,sorted_active_features_idx)
            result_top = await get_prediction(prompt,top_explanations,client,model,sorted_active_features_idx)
            
            
            sorted_result_quantile = sorted(result_quantile.items(), key=lambda x: x[1][0], reverse=True)
            sorted_result_top = sorted(result_top.items(), key=lambda x: x[1][0], reverse=True)
            # Compute the fraction of correct predictions
            # if feature is in active_features and prediction is 1, it is a correct prediction
            # if it is not in active_features and prediction is 0, it is a correct prediction
            # otherwise it is an incorrect prediction
            true_positives_quantile = sum(1 for k, v in sorted_result_quantile if int(k) in active_features_idx and v[1] == 1)
            true_negatives_quantile = sum(1 for k, v in sorted_result_quantile if int(k) not in active_features_idx and v[1] == 0)
            true_positives_top = sum(1 for k, v in sorted_result_top if int(k) in active_features_idx and v[1] == 1)
            true_negatives_top = sum(1 for k, v in sorted_result_top if int(k) not in active_features_idx and v[1] == 0)
            correct_predictions_quantile = true_positives_quantile + true_negatives_quantile
            correct_predictions_top = true_positives_top + true_negatives_top
            total_predictions_quantile = len(sorted_result_quantile)
            total_predictions_top = len(sorted_result_top)
            fraction_correct_quantile = correct_predictions_quantile / total_predictions_quantile
            fraction_correct_top = correct_predictions_top / total_predictions_top
            print(f"Fraction of correct predictions (quantile): {fraction_correct_quantile}")
            print(f"Fraction of correct predictions (top): {fraction_correct_top}")

            # Compute the fraction of correct predictions for the active features
            correct_predictions_active_idx_quantile = [int(i) for i, v in sorted_result_quantile if int(i) in active_features_idx and v[1] == 1]
            correct_predictions_active_idx_top = [int(i) for i, v in sorted_result_top if int(i) in active_features_idx and v[1] == 1]
            correct_active_quantile.append(correct_predictions_active_idx_quantile)
            correct_active_top.append(correct_predictions_active_idx_top)

            incorrect_predictions_active_idx_quantile = [int(i) for i, v in sorted_result_quantile if int(i) in active_features_idx and v[1] == 0]
            incorrect_predictions_active_idx_top = [int(i) for i, v in sorted_result_top if int(i) in active_features_idx and v[1] == 0]
            
            correct_predictions_active_quantile = len(correct_predictions_active_idx_quantile)
            fraction_correct_active_quantile = correct_predictions_active_quantile / len(active_features)
            print(f"Fraction of correct predictions for the active features (quantile): {fraction_correct_active_quantile}")
            
            correct_predictions_active_top = len(correct_predictions_active_idx_top)
            fraction_correct_active_top = correct_predictions_active_top / len(active_features)
            print(f"Fraction of correct predictions for the active features (top): {fraction_correct_active_top}")

            # The features that the model got right will be used. Then we sample the rest from the features it didn't get right
            selected_features_quantile = [i for i in correct_predictions_active_idx_quantile]
            selected_features_top = [i for i in correct_predictions_active_idx_top]
            correct_activations_quantile = autoencoder_features[0,-1,selected_features_quantile]
            correct_activations_top = autoencoder_features[0,-1,selected_features_top]
            
            # create a tensor like autoencoder_features but with the selected features
            selected_features_tensor_quantile = autoencoder_features.clone()
            selected_features_tensor_quantile[0,-1,:] = torch.zeros_like(selected_features_tensor_quantile[0,-1,:])
            selected_features_tensor_quantile[0,-1,selected_features_quantile] = correct_activations_quantile
            selected_features_tensor_quantile = selected_features_tensor_quantile.to(device)

            # top tensor
            selected_features_tensor_top = autoencoder_features.clone()
            selected_features_tensor_top[0,-1,:] = torch.zeros_like(selected_features_tensor_top[0,-1,:])
            selected_features_tensor_top[0,-1,selected_features_top] = correct_activations_top
            selected_features_tensor_top = selected_features_tensor_top.to(device)

            # sample the rest from the false positives
            false_positives_idx_quantile = [int(i) for i, v in sorted_result_quantile if int(i) not in active_features_idx and v[1] == 1]
            false_positives_idx_top = [int(i) for i, v in sorted_result_top if int(i) not in active_features_idx and v[1] == 1]
            
            false_positives_idx_quantile = false_positives_idx_quantile[:len(active_features)-len(selected_features_quantile)]
            false_positives_idx_top = false_positives_idx_top[:len(active_features)-len(selected_features_top)]

            #wrong active features
            wrong_activations_quantile = autoencoder_features[0,-1,incorrect_predictions_active_idx_quantile]
            wrong_activations_top = autoencoder_features[0,-1,incorrect_predictions_active_idx_top]
            
            false_features_tensor_quantile = autoencoder_features.clone()
            false_features_tensor_quantile[0,-1,:] = torch.zeros_like(false_features_tensor_quantile[0,-1,:])
            false_features_tensor_quantile[0,-1,incorrect_predictions_active_idx_quantile] = wrong_activations_quantile
            false_features_tensor_quantile = false_features_tensor_quantile.to(device)

            false_features_tensor_top = autoencoder_features.clone()
            false_features_tensor_top[0,-1,:] = torch.zeros_like(false_features_tensor_top[0,-1,:])
            false_features_tensor_top[0,-1,incorrect_predictions_active_idx_top] = wrong_activations_top
            false_features_tensor_top = false_features_tensor_top.to(device)    

            # random tensor
            selected_features_tensor_random = autoencoder_features.clone()
            selected_features_tensor_random[0,-1,:] = torch.zeros_like(selected_features_tensor_random[0,-1,:])
            random_idx = torch.randperm(131072)
            selected_features_tensor_random[0,-1,random_idx[:len(active_features)]] = active_features
            selected_features_tensor_random = selected_features_tensor_random.to(device)

            with model.trace(prompt):
                reconstructed_quantile = autoencoder.ae.decode(selected_features_tensor_quantile)
                model.model.layers[11].output[0][0,1:,:] = reconstructed_quantile[0,1:,:]
                simulated_output_quantile = model.output.save()

            with model.trace(prompt):
                reconstructed_top = autoencoder.ae.decode(selected_features_tensor_top)
                model.model.layers[11].output[0][0,1:,:] = reconstructed_top[0,1:,:]
                simulated_output_top = model.output.save()

            with model.trace(prompt):
                reconstructed = autoencoder.ae.decode(selected_features_tensor_random)
                model.model.layers[11].output[0][0,1:,:] = reconstructed[0,1:,:]
                simulated_output_random = model.output.save()

            
            #greedy decoding
            simulated_token_quantile = simulated_output_quantile.logits[0].argmax(dim=-1)[-1]
            simulated_outputs_quantile.append(model.tokenizer.decode(simulated_token_quantile))

            simulated_token_top = simulated_output_top.logits[0].argmax(dim=-1)[-1]
            simulated_outputs_top.append(model.tokenizer.decode(simulated_token_top))

            simulated_token_random = simulated_output_random.logits[0].argmax(dim=-1)[-1]
            random_outputs.append(model.tokenizer.decode(simulated_token_random))

            
            # compute kl divergence
            kl_div_reconstructed = kl_div(reconstructed_output.logits[0].log_softmax(dim=-1),output.logits[0].softmax(dim=-1)).item()
            kl_div_simulated_quantile = kl_div(simulated_output_quantile.logits[0].log_softmax(dim=-1),output.logits[0].softmax(dim=-1)).item()
            kl_div_simulated_top = kl_div(simulated_output_top.logits[0].log_softmax(dim=-1),output.logits[0].softmax(dim=-1)).item()
            kl_div_random = kl_div(simulated_output_random.logits[0].log_softmax(dim=-1),output.logits[0].softmax(dim=-1)).item()
            
            running_kl_div_reconstructed.append(kl_div_reconstructed)
            running_kl_div_simulated_quantile.append(kl_div_simulated_quantile)
            running_kl_div_simulated_top.append(kl_div_simulated_top)
            running_kl_div_random.append(kl_div_random)
            
            running_fraction_correct_quantile.append(fraction_correct_quantile)
            running_fraction_correct_top.append(fraction_correct_top)
            running_fraction_correct_active_quantile.append(fraction_correct_active_quantile)
            running_fraction_correct_active_top.append(fraction_correct_active_top)
        
            print(f"KL divergence (reconstructed): {kl_div_reconstructed}")
            print(f"KL divergence (simulated quantile): {kl_div_simulated_quantile}")
            print(f"KL divergence (simulated top): {kl_div_simulated_top}")
            print(f"KL divergence (random): {kl_div_random}")
        
            print(prompt + normal_outputs[-1])
            print(prompt + reconstructed_outputs[-1])
            print(prompt + simulated_outputs_quantile[-1])
            print(prompt + simulated_outputs_top[-1])
            print(prompt + random_outputs[-1])

            prompt = prompt + model.tokenizer.decode(real_token)



        all_normal_outputs.append(normal_outputs)
        all_reconstructed_outputs.append(reconstructed_outputs)
        all_simulated_outputs_quantile.append(simulated_outputs_quantile)
        all_simulated_outputs_top.append(simulated_outputs_top)
        all_random_outputs.append(random_outputs)
        # compute average kl divergence
        avg_kl_div_reconstructed = sum(running_kl_div_reconstructed) / len(running_kl_div_reconstructed)
        avg_kl_div_simulated_quantile = sum(running_kl_div_simulated_quantile) / len(running_kl_div_simulated_quantile)
        avg_kl_div_simulated_top = sum(running_kl_div_simulated_top) / len(running_kl_div_simulated_top)
        avg_kl_div_random = sum(running_kl_div_random) / len(running_kl_div_random)
        all_kl_div_reconstructed.append(running_kl_div_reconstructed)
        all_kl_div_simulated_quantile.append(running_kl_div_simulated_quantile)
        all_kl_div_simulated_top.append(running_kl_div_simulated_top)
        all_kl_div_random.append(running_kl_div_random)

        print(f"Average KL divergence (reconstructed): {avg_kl_div_reconstructed}")
        print(f"Average KL divergence (simulated quantile): {avg_kl_div_simulated_quantile}")
        print(f"Average KL divergence (simulated top): {avg_kl_div_simulated_top}")
        print(f"Average KL divergence (random): {avg_kl_div_random}")

        all_fraction_correct_quantile.append(running_fraction_correct_quantile)
        all_fraction_correct_top.append(running_fraction_correct_top)
        all_fraction_correct_active_quantile.append(running_fraction_correct_active_quantile)
        all_fraction_correct_active_top.append(running_fraction_correct_active_top)

        all_correct_active_top.append(correct_active_top)
        all_correct_active_quantile.append(correct_active_quantile)
        

    with open(f"{RESULTS_FOLDER}/iterative_generation_outputs.json", "w") as f:
        json.dump({"normal_outputs": all_normal_outputs, "reconstructed_outputs": all_reconstructed_outputs, "simulated_outputs_quantile": all_simulated_outputs_quantile, "simulated_outputs_top": all_simulated_outputs_top, "random_outputs": all_random_outputs}, f)
    with open(f"{RESULTS_FOLDER}/iterative_generation_kl_div.json", "w") as f:
        json.dump({"kl_div_reconstructed": all_kl_div_reconstructed, "kl_div_simulated_quantile": all_kl_div_simulated_quantile, "kl_div_simulated_top": all_kl_div_simulated_top, "kl_div_random": all_kl_div_random}, f)
    with open(f"{RESULTS_FOLDER}/iterative_generation_fraction_correct.json", "w") as f:
        json.dump({"fraction_correct_quantile": all_fraction_correct_quantile, "fraction_correct_top": all_fraction_correct_top, "fraction_correct_active_quantile": all_fraction_correct_active_quantile, "fraction_correct_active_top": all_fraction_correct_active_top}, f)
    with open(f"{RESULTS_FOLDER}/iterative_generation_active_features.json", "w") as f:
        json.dump({"correct_active_top": all_correct_active_top, "correct_active_quantile": all_correct_active_quantile}, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="8b")
    args = parser.parse_args()
    asyncio.run(main(args))
