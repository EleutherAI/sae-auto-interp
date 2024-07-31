python demo/generate.py \
    --width 131072 \
    --min_examples 100 \
    --max_examples 5000 \
    --ctx_len 64 \
    --n_splits 2 \
    --n_train 4 \
    --n_test 5 \
    --n_quantiles 10

python demo/generation_score.py
