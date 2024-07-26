for i in $(seq 1 9); do
    python experiments/positional.py \
        --layer $i \
        --n_tokens 100000 \
        --seq_len 1024 \
        --minibatch_size 15
done

python experiments/positional.py \
    --layer 11 \
    --n_tokens 100000 \
    --seq_len 1024 \
    --minibatch_size 15