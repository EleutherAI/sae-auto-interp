layers=(0 1 2 3 4 5 6 7 8 9 10 11)
for layer in ${layers[@]}; do
    python extras/scripts/score_neighbor_gpt2.py --layer $layer &
    sleep 60
    # python extras/scripts/score_different_samples.py --layer $layer &
    # sleep 30
    # python extras/scripts/score_cotvssimpple.py --layer $layer &
    # sleep 30
done 