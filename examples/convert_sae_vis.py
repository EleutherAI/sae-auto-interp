#%%
from sae_dashboard.sae_vis_data import SaeVisConfig, SaeVisData
from sae_dashboard.feature_data import FeatureData
from sae_dashboard.components import FeatureTablesData, LogitsHistogramData, ActsHistogramData
from sae_dashboard.components import SequenceMultiGroupData, SequenceGroupData, SequenceData
from sae_dashboard.sae_vis_runner import SaeVisRunner
from sae_dashboard.utils_fns import FeatureStatistics
from sae_dashboard.data_parsing_fns import get_logits_table_data
# %%
from itertools import batched
from argparse import Namespace
from delphi.config import ExperimentConfig, FeatureConfig
from delphi.features import FeatureDataset, FeatureLoader
from delphi.features.constructors import default_constructor
from delphi.features.samplers import sample
from functools import partial
import torch
import numpy as np
from sae_dashboard.utils_fns import ASYMMETRIC_RANGES_AND_PRECISIONS
from tqdm.auto import tqdm

args = Namespace(
    module=".model.layers.16.router",
    feature_options=FeatureConfig(),
    features=100,
    model="monet_cache_converted/850m"
)
module = args.module
feature_cfg = args.feature_options
n_features = args.features  
start_feature = 0
sae_model = args.model
feature_dict = {f"{module}": torch.arange(start_feature,start_feature+n_features)}
dataset = FeatureDataset(
    raw_dir=f"results/{args.model}",
    cfg=feature_cfg,
    modules=[module],
    features=feature_dict,
)


def set_record_buffer(record, buffer_output):
    record.buffer = buffer_output
loader = FeatureLoader(dataset, constructor=set_record_buffer, sampler=lambda x: x, transform=lambda x: x)
#%%
tokens = dataset.buffers[0].load()[3]
n_sequences, max_seq_len = tokens.shape
#%%

cfg = SaeVisConfig(
    hook_point=args.module,
    minibatch_size_tokens=dataset.cache_config["ctx_len"],
    features=[],
    # batch_size=dataset.cache_config["batch_size"],
)
layout = cfg.feature_centric_layout
feature_data_dict = {}

ranges_and_precisions = ASYMMETRIC_RANGES_AND_PRECISIONS
quantiles = []
for r, p in ranges_and_precisions:
    start, end = r
    step = 10**-p
    quantiles.extend(np.arange(start, end - 0.5 * step, step))
quantiles_tensor = torch.tensor(quantiles, dtype=torch.float32)
feature_stats = FeatureStatistics()
# supposed_feature = 0
for i, record in enumerate(tqdm(loader, total=args.features)):
    # if record.buffer.locations[0, 2].item() > supposed_feature:
    #     for _ in range(supposed_feature, record.buffer.locations[0, 2].item()):
    #         feature_stats.update(FeatureStatistics(
    #             max=[0],
    #             frac_nonzero=[0],
    #             skew=[0],
    #             kurtosis=[0],
    #             quantile_data=[quantiles_tensor.new_zeros(quantiles_tensor.shape).unsqueeze(0).tolist()],
    #             quantiles=quantiles + [1.0],
    #             ranges_and_precisions=ranges_and_precisions
    #         ))
    #         supposed_feature += 1
    #     continue
    # https://github.com/jbloomAus/SAEDashboard/blob/main/sae_dashboard/utils_fns.py
    buffer = record.buffer
    activations, locations = buffer.activations, buffer.locations
    _max = activations.max()
    nonzero_mask = activations.abs() > 1e-6
    nonzero_acts = activations[nonzero_mask]
    frac_nonzero = nonzero_mask.sum() / (n_sequences * max_seq_len)
    quantile_data = torch.quantile(activations.float(), quantiles_tensor)
    skew = torch.mean((activations - activations.mean())**3) / (activations.std()**3)
    kurtosis = torch.mean((activations - activations.mean())**4) / (activations.std()**4)
    feature_stats.update(FeatureStatistics(
        max=[_max.item()],
        frac_nonzero=[frac_nonzero.item()],
        skew=[skew.item()],
        kurtosis=[kurtosis.item()],
        quantile_data=[quantile_data.unsqueeze(0).tolist()],
        quantiles=quantiles + [1.0],
        ranges_and_precisions=ranges_and_precisions
    ))
    
    logit_vector = torch.linspace(0, 1e-3, 100, dtype=torch.float32)
    
    feature_id = record.buffer.locations[0, 2].item()
    feature_data = FeatureData()
    feature_data.feature_tables_data = FeatureTablesData()
    feature_data.logits_histogram_data = LogitsHistogramData.from_data(
        data=logit_vector.to(
            torch.float32
        ),  # need this otherwise fails on MPS
        n_bins=layout.logits_hist_cfg.n_bins,  # type: ignore
        tickmode="5 ticks",
        title=None,
    )
    feature_data.acts_histogram_data = ActsHistogramData.from_data(
        data=nonzero_acts.to(torch.float32),
        n_bins=layout.act_hist_cfg.n_bins,
        tickmode="5 ticks",
        title=f"ACTIVATIONS<br>DENSITY = {frac_nonzero:.3%}",
    )
    feature_data.logits_table_data = get_logits_table_data(
        logit_vector=logit_vector,
        n_rows=layout.logits_table_cfg.n_rows,  # type: ignore
    )
    # feature_data.sequence_data = sequence_data_generator.get_sequences_data(
    #     feat_acts=masked_feat_acts,
    #     # feat_logits=logits[i],
    #     resid_post=torch.tensor([]),  # no longer used
    #     # feature_resid_dir=feature_resid_dir[i],
    # )
    feature_data_dict[feature_id] = feature_data
    # supposed_feature += 1

feature_list = feature_dict[module].tolist()
cfg.features = feature_list
#%%
n_quantiles = 5
experiment_cfg = ExperimentConfig(
    n_examples_train=25,
    example_ctx_len=16,
    n_quantiles=n_quantiles,
    train_type="quantiles"
)
sampler = partial(sample,cfg=experiment_cfg)
sequence_loader = FeatureLoader(dataset, constructor=constructor, sampler=sampler)
constructor=partial(
    default_constructor,
    # token_loader=lambda: dataset.load_tokens(),
    token_loader=None,
    n_random=experiment_cfg.n_random, 
    ctx_len=experiment_cfg.example_ctx_len, 
    max_examples=feature_cfg.max_examples
)
for record in tqdm(sequence_loader):
    groups = []
    for quantile_index, quantile_data in enumerate(
        list(batched(record.train, len(record.train) // n_quantiles))[::-1]):
        group = []
        for example in quantile_data:
            default_list = [0.0] * len(example.tokens)
            logit_list = [[0.0]] * len(default_list)
            token_list = [[0]] * len(default_list)
            default_attrs = dict(
                loss_contribution=default_list,
                token_logits=default_list,
                top_token_ids=token_list,
                top_logits=logit_list,
                bottom_token_ids=token_list,
                bottom_logits=logit_list,
            )
            group.append(SequenceData(
                token_ids=example.tokens.tolist(),
                feat_acts=example.activations.tolist(),
                **default_attrs
            ))
        groups.append(SequenceGroupData(
            title=f"Quantile {quantile_index/n_quantiles:1%}-{(quantile_index+1)/n_quantiles:1%}",
            seq_data=group,
        ))
    feature_data_dict[record.feature.feature_index].sequence_data = SequenceMultiGroupData(
        seq_group_data=groups
    )
# %%
feature_list = list(feature_data_dict.keys())
tokenizer = dataset.tokenizer
model = Namespace(
    tokenizer=tokenizer,
)

sae_vis_data = SaeVisData(
    cfg=cfg,
    feature_data_dict=feature_data_dict,
    feature_stats=feature_stats,
    model=model,
)
from sae_dashboard.data_writing_fns import save_feature_centric_vis
save_feature_centric_vis(sae_vis_data=sae_vis_data, filename="results/feature_dashboard.html")
# %%
