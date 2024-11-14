import asyncio
import json
import os
import time
from functools import partial

import numpy as np
import orjson
import torch
from nnsight import LanguageModel
from safetensors.numpy import load_file
from simple_parsing import ArgumentParser
from nl.fuzz import FuzzingScorer

from sae_auto_interp.clients import Offline
from sae_auto_interp.features.features import Example, Feature, FeatureRecord
from sae_auto_interp.pipeline import Pipe, Pipeline, process_wrapper
from sae_auto_interp.utils import load_tokenized_data
import glob
from nl.autoencoders import AutoencoderConfig, AutoencoderLatents

import pandas as pd

def build_feature_record(selected_tokens ,activation,explanation=None,order=-1,sentence_idx=-1,feature=0):
    feature = Feature("layer", feature)
    activations = torch.zeros_like(selected_tokens,dtype=torch.float32) # The real activations don't matter
    activations[-1] = activation
    example = Example(selected_tokens, activations)
    feature_record = FeatureRecord(feature)
    feature_record.explanation = explanation
    feature_record.test = [example]
    feature_record.sentence_idx = sentence_idx
    feature_record.order = order
    return feature_record

def create_records(sentence_idx, tokens, locations, activation, all_explanations,all_scores, window_size,SCORES_FOLDER):
    window_idx = np.random.randint(window_size+1,256-window_size)
    selected_window = tokens[sentence_idx][window_idx-window_size:window_idx]
    
    selected_sentence_idx = locations[:,0]==sentence_idx
    sentence_locations = locations[selected_sentence_idx]
    sentence_activation = activation[selected_sentence_idx]
    
    token_idx = sentence_locations[:,1]==window_idx-1 # selected tokens does not include the last token
    locations_at_token = sentence_locations[token_idx]
    activation_at_token = sentence_activation[token_idx]
    active_features = locations_at_token[:,2]

    # sort the active features by the activation
    sorted_active_features = active_features[torch.argsort(activation_at_token)]
    sorted_activation_at_token = activation_at_token[torch.argsort(activation_at_token)]
    feature_records = []
    # Create records for the active features
    for idx,active_feature in enumerate(sorted_active_features):
        if str(active_feature.item()) not in all_explanations:
            continue
        if str(active_feature.item()) in all_scores:
            score = "Score: "+str(all_scores[str(active_feature.item())])
        else:
            score = ""
        explanation = all_explanations[str(active_feature.item())]
        explanation = explanation+score
        feature_record = build_feature_record(selected_window,explanation=explanation,activation=sorted_activation_at_token[idx],order=idx,sentence_idx=sentence_idx,feature=active_feature.item())
        feature_records.append(feature_record)
    os.makedirs(SCORES_FOLDER+f"{sentence_idx}", exist_ok=True)
    return feature_records


async def feature_generator(tokens, locations, activation, all_explanations,all_scores, window_size, number_of_sentences, start_sentence, SCORES_FOLDER):
    for sentence_idx in range(start_sentence,start_sentence+number_of_sentences):
        feature_records = create_records(sentence_idx, tokens, locations, activation, all_explanations,all_scores, window_size, SCORES_FOLDER)
        for feature_record in feature_records:
            yield feature_record
# Make postprocess for the fuzzing scorer
def scorer_postprocess(result,output_folder):
    prediction = result.score[0].prediction
    sentence_idx = result.record.sentence_idx
    feature = str(result.record.feature).split("feature")[-1]
    activations = result.record.test[0].activations[-1]
    saving_result = {"feature":int(feature),"prediction":prediction,"text":result.score[0].text,"activation":activations.tolist()}
    with open(f"{output_folder}{sentence_idx}/{feature}.txt", "wb") as f:
            f.write(orjson.dumps(saving_result))
    return result
    
def coalest_results(SCORES_FOLDER):
    all_files = glob.glob(os.path.join(SCORES_FOLDER, "*.txt"))
    all_results = []
    for file in all_files:
        with open(file, "r") as f:
            data = json.load(f)
            all_results.append(data)
    
    all_data = pd.DataFrame(all_results)
    return all_data


def main(args):
    
    model = LanguageModel("google/gemma-2-9b", device_map="cpu", dispatch=True,torch_dtype="float16")
    layer = 11
    all_locations = []
    all_activations = []
    ranges = ["0_26213","26214_52427","52428_78642","78643_104856","104857_131071"]
    for valid_range in ranges:
        split_data = load_file(f"/mnt/ssd-1/gpaulo/SAE-Zoology/raw_features/gemma/131k/.model.layers.{layer}/{valid_range}.safetensors")
        activations = torch.tensor(split_data["activations"])
        locations = torch.tensor(split_data["locations"].astype(np.int64))
        locations[:,2] = locations[:,2]+int(valid_range.split("_")[0])
        all_locations.append(locations)
        all_activations.append(activations)


    locations = torch.cat(all_locations)
    activation = torch.cat(all_activations)
    if args.explanation == "quantiles":
        with open(f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/explanations_131k/model.layers.{layer}_feature.json", "r") as f:
            #load json file
            all_explanations = json.load(f)
    else:
        with open(f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/explanations_131k_top/model.layers.{layer}_feature.json", "r") as f:
            #load json file
            all_explanations = json.load(f)
    all_scores={}
    if args.score == "fuzz":
        with open(f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/full_simulation/formated_scores/gemma/131k/res/fuzz_layer_{layer}.json", "r") as f:
            #load json file
            all_scores = json.load(f)
    if args.score == "recall":
        with open(f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/full_simulation/formated_scores/gemma/131k/res/recall_layer_{layer}.json", "r") as f:
            #load json file
            all_scores = json.load(f)

    tokens = load_tokenized_data(
            256,
            model.tokenizer,
            "EleutherAI/rpj-v2-sample",
            "train[:1%]",
            "",    
            "raw_content"
        )

    if args.model_size == "8b":
        client = Offline("meta-llama/Meta-Llama-3.1-8B-Instruct",max_memory=0.8,max_model_len=5120,num_gpus=2,batch_size=4000)
        SCORES_FOLDER= f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/full_simulation/kl_div/8b{args.score}{args.soft}/{args.window_size}/{args.explanation}/"
        device = "cuda:3"
    elif args.model_size == "70b":
        client = Offline("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",max_memory=0.75, max_model_len=4096,num_gpus=4)
        SCORES_FOLDER= f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/full_simulation/kl_div/70b{args.score}/{args.window_size}/{args.explanation}/"
        device = "cuda:4"
    elif args.model_size == "8b-quantiles_top5":
        client = Offline("quantiles_top5",max_memory=0.8,max_model_len=5120,num_gpus=2,batch_size=4000)
        SCORES_FOLDER= f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/full_simulation/kl_div/8b-quantiles_top5{args.score}/{args.window_size}/{args.explanation}/"
        device = "cuda:3"
    elif args.model_size == "8b-top_top5":
        client = Offline("top_top5",max_memory=0.8,max_model_len=5120,num_gpus=2,batch_size=4000)
        SCORES_FOLDER= f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/full_simulation/kl_div/8b-top_top5{args.score}/{args.window_size}/{args.explanation}/"
        device = "cuda:3"
    else:
        raise ValueError("Model size not supported")
    start = time.time()
    window_size = args.window_size
    np.random.seed(42)

        
    if args.model_size == "8b-quantiles_top5" or args.model_size == "8b-lora-top":
        finetuned = True
    else:
        finetuned = False
    if args.score != "":
        score = True
    else:
        score = False
    if args.soft == "":
        soft = False
    else:
        soft = True

    fuzz_scorer = FuzzingScorer(client, model.tokenizer,contexts=False, finetuned=finetuned,score=score,soft=soft)
    # make folder for the scores
            
    scorer_pipe = Pipe(process_wrapper(fuzz_scorer, postprocess=partial(scorer_postprocess,output_folder=SCORES_FOLDER)))
    pipeline = Pipeline(
        feature_generator(tokens, locations, activation, all_explanations,all_scores, window_size, args.num_sentences, args.start_sentence, SCORES_FOLDER),
        scorer_pipe,
    )

    asyncio.run(pipeline.run(10000))
    all_data = []
    for i in range(args.start_sentence,args.start_sentence+args.num_sentences):
        data = coalest_results(f"{SCORES_FOLDER}{i}")
        all_data.append(data)
    all_data = pd.concat(all_data)
    end = time.time()
    print(f"Time taken: {end-start}")
    model = LanguageModel("google/gemma-2-9b", device_map=device, dispatch=True,torch_dtype="float16")

    config = AutoencoderConfig(
        model_name_or_path="google/gemma-scope-9b-pt-res",
        autoencoder_type="CUSTOM",
        device=device,
        hookpoints=[f"layer_11/width_131k/average_l0_49"],
        kwargs={"custom_name": "gemmascope"}
    )
    autoencoder = AutoencoderLatents.from_pretrained(config,hookpoint=f"layer_11/width_131k/average_l0_49")
    autoencoder.ae = autoencoder.ae.to(device)    
    prompts = all_data["text"].unique()
    kl_divergences = []
    kl_divergences_prediction = []
    for prompt in prompts:
        tokenized_prompt = model.tokenizer.encode(prompt,add_special_tokens=True)
        with model.trace(tokenized_prompt):
            real_layer_activation = model.model.layers[11].output[0].save()
            autoencoder_features = autoencoder.ae.encode(real_layer_activation).save()
            output = model.output.save()
                        
        selected_features_tensor = autoencoder_features.clone()
        selected_data = all_data[all_data["text"]==prompt]
        features = selected_data["feature"].tolist()
        activations = selected_data["activation"].tolist()
        # get the activations sorted by the features
        selected_features_tensor[0,-1,features] = torch.tensor(activations,dtype=autoencoder_features.dtype,device=autoencoder_features.device)
        with model.trace(tokenized_prompt):
            reconstructed = autoencoder.ae.decode(selected_features_tensor)
            model.model.layers[11].output[0][0,1:,:] = reconstructed[0,1:,:]
            simulated_output = model.output.save()

        # compute the kl divergence at token -2 
        kl_divergence = torch.nn.functional.kl_div(output.logits[0,-1,:].unsqueeze(0).log_softmax(dim=-1),simulated_output.logits[0,-1,:].unsqueeze(0).softmax(dim=-1),reduction="batchmean")
        
        kl_divergences.append(kl_divergence.item())
        #print(f"KL divergence reconstruction: {kl_divergence.item()}")
        
        correct_features = selected_data[selected_data["prediction"]==1]["feature"].tolist()
        correct_activations = selected_data[selected_data["prediction"]==1]["activation"].tolist()
        #print(len(correct_features)/len(features))
        
    
        selected_features_tensor = autoencoder_features.clone()
        selected_features_tensor[0,-1,:] = torch.zeros_like(selected_features_tensor[0,-1,:])
        selected_features_tensor[0,-1,correct_features] = torch.tensor(correct_activations,dtype=autoencoder_features.dtype,device=autoencoder_features.device)
        # select random indices to be active 
        #wrong_activations = selected_data[selected_data["prediction"]==0]["activation"].tolist()
        #print(wrong_activations)
            
        #random_indices = torch.randperm(autoencoder_features.shape[2])[:len(wrong_activations)]
        #selected_features_tensor[0,-1,random_indices] = torch.tensor(wrong_activations,dtype=autoencoder_features.dtype,device=autoencoder_features.device)

        with model.trace(prompt):
            reconstructed = autoencoder.ae.decode(selected_features_tensor)
            model.model.layers[11].output[0][0,1:,:] = reconstructed[0,1:,:]
            simulated_output = model.output.save()

        kl_divergence = torch.nn.functional.kl_div(output.logits[0,-1,:].unsqueeze(0).log_softmax(dim=-1),simulated_output.logits[0,-1,:].unsqueeze(0).softmax(dim=-1),reduction="batchmean")
        #print(f"KL divergence prediction: {kl_divergence.item()}")
        kl_divergences_prediction.append(kl_divergence.item())
        del real_layer_activation, autoencoder_features, output, simulated_output, selected_features_tensor
    print(f"KL divergence reconstruction: {np.mean(kl_divergences),np.median(kl_divergences)}")
    print(f"KL divergence prediction: {np.mean(kl_divergences_prediction),np.median(kl_divergences_prediction)}")
    # save the results
    with open(f"{SCORES_FOLDER}kl_divergences.json", "w") as f:
        json.dump({"reconstruction":kl_divergences,"prediction":kl_divergences_prediction},f)

    
if __name__ == "__main__":
    parser = ArgumentParser()
    #parser.add_argument("--sentence_idx", type=int, default=0)
    parser.add_argument("--explanation",type=str,default="quantiles")
    parser.add_argument("--model_size", type=str, default="8b")
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--num_sentences", type=int, default=100)
    parser.add_argument("--start_sentence", type=int, default=0)
    parser.add_argument("--score", type=str, default="")
    parser.add_argument("--soft", type=str, default="")
    args = parser.parse_args()
    #sentence_idx = args.sentence_idx

    main(args)