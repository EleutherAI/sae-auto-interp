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
from tqdm import tqdm

from sae_auto_interp.clients import Offline
from sae_auto_interp.config import FeatureConfig, ExperimentConfig
from sae_auto_interp.features import FeatureDataset
from sae_auto_interp.features.constructors import default_constructor
from sae_auto_interp.features.samplers import sample
from sae_auto_interp.features.features import Example, Feature, FeatureRecord
from sae_auto_interp.pipeline import Pipe, Pipeline, process_wrapper
from sae_auto_interp.utils import load_tokenized_data
import glob
from full_sim.special_autoencoders import AutoencoderConfig, AutoencoderLatents
from full_sim.special_simulator import Simulator

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

def create_records(sentence_idx, tokens, locations, activation, all_explanations,all_scores,high_scores, window_size,SCORES_FOLDER):
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
    index_sort = torch.argsort(activation_at_token)
    sorted_active_features = active_features[index_sort]
    sorted_activation_at_token = activation_at_token[index_sort]
    feature_records = []
    # Create records for the active features

    for latent in high_scores:

        if latent not in sorted_active_features:
            activation = 0
            order = -1
        else:
            order = torch.where(sorted_active_features==latent)[0][0].item()
            activation = sorted_activation_at_token[order]
            
        if str(latent) not in all_explanations:
            continue
        explanation = all_explanations[str(latent)]
        if str(latent) in all_scores:
            score = "Score: "+str(all_scores[str(latent)])
        else:
            score = ""
        explanation = explanation+score
        feature_record = build_feature_record(selected_window,explanation=explanation,activation=activation,order=order,sentence_idx=sentence_idx,feature=latent)
        feature_records.append(feature_record)
    os.makedirs(SCORES_FOLDER+f"{sentence_idx}", exist_ok=True)
    return feature_records


async def feature_generator(tokens, locations, activation, all_explanations,all_scores,high_scores, window_size, number_of_sentences, start_sentence, SCORES_FOLDER):
    for sentence_idx in range(start_sentence,start_sentence+number_of_sentences):
        feature_records = create_records(sentence_idx, tokens, locations, activation, all_explanations,all_scores,high_scores, window_size, SCORES_FOLDER)
        for feature_record in feature_records:
            yield feature_record
# Make postprocess for the fuzzing scorer
def scorer_postprocess(result,output_folder,quantile_stats):
    
    predicted_quantile = result.score[0].predicted_quantile
    activation = result.score[0].activation.item()
    expected_quantile = result.score[0].expected_quantile
    sentence_idx = result.record.sentence_idx
    order = result.record.order 
    text = result.score[0].text
    feature = int(str(result.record.feature).split("feature")[-1])
    quantile_info = quantile_stats[feature]
    predicted_activation = 0
    # linear interpolation between the two closest quantiles
    for idx,quantile in enumerate(quantile_info):
        if expected_quantile < idx:
            predicted_activation = quantile_info[idx-1]+(expected_quantile-idx+1)*(quantile_info[idx]-quantile_info[idx-1])
            break
    # print the types of the variables
    try:
        saving_result = {
            #"text": str(text),
            "feature": int(feature),
            "predicted_quantile": float(predicted_quantile),
            "activation": float(activation),
            "order": int(order),
            "expected_quantile": float(expected_quantile),
            "predicted_activation": float(predicted_activation)
        }
    except:
        print(f"Error with {sentence_idx} {feature}")
        saving_result = {
            #"text": "Error",
            "feature": -1,
            "predicted_quantile": -1,
            "activation": -1,
            "order": -1,
            "expected_quantile": -1,
            "predicted_activation": -1
        }
    with open(f"{output_folder}{sentence_idx}/{feature}.txt", "wb") as f:
            f.write(orjson.dumps(saving_result))
    return result
    
def merge_scores(SCORES_FOLDER):
    all_files = glob.glob(os.path.join(SCORES_FOLDER, "*.txt"))
    all_results = []
    for file in all_files:
        with open(file, "r") as f:
            data = json.load(f)
            all_results.append(data)
    
    all_data = pd.DataFrame(all_results)
    all_data.to_csv(f"{SCORES_FOLDER}all_data.csv",index=False)
    return all_data

def get_quantile_stats(locations,activations,high_scores):
    index_sort = torch.argsort(locations[:,2])
    sorted_locations = locations[index_sort]
    sorted_activations = activations[index_sort]
    
    quantile_stats = {}
    start_idx = 0
    for latent in tqdm(high_scores):
        #start_idx = torch.searchsorted(sorted_locations[:,2], latent)
        end_idx = torch.searchsorted(sorted_locations[:,2], latent + 1)
        # Get all activations for this feature
        feature_activation = sorted_activations[start_idx:end_idx]
        #feature_locations = sorted_locations[start_idx:end_idx]
        sorted_feature_activations = torch.sort(feature_activation,descending=True)[0]
        #print("Sorting activations took: ",time.time()-start)
        linearly_spaced_activations = torch.linspace(0,sorted_feature_activations[0],10).tolist()
        quantile_stats[latent] = linearly_spaced_activations
        start_idx = end_idx
    return quantile_stats


def main(args):
    
    model = LanguageModel("EleutherAI/pythia-160m", device_map="cpu", dispatch=True,torch_dtype="float16")
    layer = 6
    all_locations = []
    all_activations = []
    ranges = ["0_6552","6553_13106","13107_19659","19660_26213","26214_32767"]

    sae_type = "SAE"
    for valid_range in ranges:
        split_data = load_file(f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/transcoders/raw_features/pythia_pile/{sae_type}/.gpt_neox.layers.{layer}.mlp/{valid_range}.safetensors")
        activations = torch.tensor(split_data["activations"])
        locations = torch.tensor(split_data["locations"].astype(np.int64))
        locations[:,2] = locations[:,2]+int(valid_range.split("_")[0])
        all_locations.append(locations)
        all_activations.append(activations)


    locations = torch.cat(all_locations)
    activation = torch.cat(all_activations)
    with open(f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/transcoders/results/explanations_layer{layer}_{sae_type}.json", "r") as f:
            #load json file
            all_explanations = json.load(f)
   
    all_scores = pd.read_csv(f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/transcoders/results/scores_layer{layer}_recall_{sae_type}.csv")
    
    
    tokens = load_tokenized_data(
            256,
            model.tokenizer,
            "monology/pile-uncopyrighted",
            "train[:1%]",
            "",    
            "text"
        )

    # sort by feature
    #all_scores = all_scores.sort_values(by="feature", ascending=True)
    #high_scores = all_scores["feature"][:500].tolist()
    high_scores = all_scores["feature"].tolist()
    
    quantile_stats = get_quantile_stats(locations.cuda(),activation.cuda(),high_scores)
    
    
    if args.model_size == "8b":
        client = Offline("meta-llama/Meta-Llama-3.1-8B-Instruct",max_memory=0.9,max_model_len=5120,num_gpus=2,batch_size=5000)
        SCORES_FOLDER= f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/transcoders/results/kl_div_activations/8b/{sae_type}/"
        
    elif args.model_size == "70b":
        client = Offline("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",max_memory=0.8, max_model_len=2048,num_gpus=4,batch_size=100,enforce_eager=True,prefix_caching=True)
        SCORES_FOLDER= f"/mnt/ssd-1/gpaulo/SAE-Zoology/extras/transcoders/results/kl_div_activations/70b/{sae_type}/"
        
    else:
        raise ValueError("Model size not supported")
    start = time.time()
    window_size = args.window_size
    np.random.seed(42)


    fuzz_scorer = Simulator(client, model.tokenizer,log_prob=True,verbose=True)
    # make folder for the scores
            
    scorer_pipe = Pipe(process_wrapper(fuzz_scorer, postprocess=partial(scorer_postprocess,output_folder=SCORES_FOLDER,quantile_stats=quantile_stats)))
    pipeline = Pipeline(
        feature_generator(tokens, locations, activation, all_explanations,all_scores,high_scores, window_size, args.num_sentences, args.start_sentence, SCORES_FOLDER),
        scorer_pipe,
    )

    asyncio.run(pipeline.run(10000))
    all_data = []
    for i in tqdm(range(args.start_sentence,args.start_sentence+args.num_sentences)):
        data = merge_scores(f"{SCORES_FOLDER}{i}")
        all_data.append(data)
    all_data = pd.concat(all_data)
    end = time.time()
    print(f"Time taken: {end-start}")
    
    
if __name__ == "__main__":
    parser = ArgumentParser()
    #parser.add_argument("--sentence_idx", type=int, default=0)
    parser.add_argument("--model_size", type=str, default="8b")
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--num_sentences", type=int, default=1000)
    parser.add_argument("--start_sentence", type=int, default=0)
    args = parser.parse_args()
    #sentence_idx = args.sentence_idx

    main(args)