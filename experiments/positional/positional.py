import argparse

from nnsight import LanguageModel
from positional_cache import FrequencyCache

from sae_auto_interp.autoencoders import load_oai_autoencoders
from sae_auto_interp.utils import load_tokenized_data


def main(args):
    model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
    submodule_dict = load_oai_autoencoders(
        model,
        [args.layer],
        "weights/gpt2_128k",
    )

    tokens = load_tokenized_data(
        args.seq_len,
        model.tokenizer,
        "kh4dien/fineweb-100m-sample",
        "train[:15%]",
    )

    cache = FrequencyCache(
        model,
        submodule_dict,
        args.minibatch_size,
        args.seq_len,
    )
    cache.run(args.n_tokens, tokens)
    _ = cache.save(threshold=0.06)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--n_tokens", type=int, default=100000)
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--minibatch_size", type=int, default=16)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)


# # %%

# results = cache.save()
# test_tokens = tokens[:100,:]

# test_tokens = torch.split(test_tokens, 10, dim=0)

# data = []

# for test in tqdm(test_tokens):

#     for layer, selected_features in results.items():
#         with model.trace(test):

#             ae_out = model.transformer.h[LAYER].ae.output
#             ae_out = torch.mean(ae_out, dim=0)
#             ae_out = ae_out[:,selected_features]
#             ae_out.save()

#         result = ae_out.value.detach().cpu()

#         data.append(result)

# data = torch.stack(data).numpy()
# data = np.mean(data, axis=0)

# # %%

# plt.figure(figsize=(10, 5))  # Increase figure size for better visibility

# # Transpose the array so each column becomes a line
# data_transposed = data.T

# # Plot each line with low opacity
# for i in range(data_transposed.shape[0]):
#     plt.plot(data_transposed[i], alpha=0.5, linewidth=2)

# plt.title('L0 Positional Features')
# plt.xlabel('Sequence')
# plt.ylabel('Activation')
# plt.xlim(0, 1023)  # Set x-axis limits explicitly
# plt.tight_layout()
# plt.show()
