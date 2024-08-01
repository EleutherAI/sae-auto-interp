for i in $(seq 0 11); do
    python experiments/positional.py \
        --layer $i \
        --n_tokens 100000 \
        --seq_len 1024 \
        --minibatch_size 15
done
