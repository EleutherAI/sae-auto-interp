layers=(0 1 2 3 4 5 6 7 8 9 10 11)
for layer in ${layers[@]}; do
    python extras/scripts/score_different_samples.py --layer $layer &
    sleep 20
    python extras/scripts/score_cotvssimpple.py --layer $layer &
    sleep 20
    python extras/scripts/score_random.py --layer $layer &
    sleep 20
    python extras/scripts/score_gpt2_llama8b.py --layer $layer &
    sleep 20
    python extras/scripts/score_gpt2_sonnet.py --layer $layer &
    sleep 20
    python extras/scripts/score_gpt2_human.py --layer $layer &
    sleep 20
    python extras/scripts/score_gpt2_small.py --layer $layer &
done 