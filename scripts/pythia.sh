python demo/pythia_eval.py \
    --width 32768 \
    --min_examples 100 \
    --max_examples 5000 \
    --ctx_len 64 \
    --n_splits 1 \
    --n_train 20 \
    --n_test 7 \
    --n_quantiles 10 \
    --batch_size 5
