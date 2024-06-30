from typing import List, Tuple
import random
from ..features.features import FeatureRecord, Example

def sample_from_quantiles(
    record: FeatureRecord, 
    n_quantiles=4, 
    n_train=10, 
    n_test=10, 
    seed=22
) -> List[Tuple[List[Example], List[Example]]]:
    random.seed(seed)
    num_examples = len(record.max_activations)
    bin_size = num_examples // n_quantiles
    bins = [record.examples[i * bin_size:(i + 1) * bin_size] for i in range(n_quantiles)]

    train_samples = []
    test_samples = []
    
    # Calculate samples per quantile
    train_per_quantile = n_train // n_quantiles
    test_per_quantile = n_test // n_quantiles
    
    for bin in bins:
        if len(bin) >= (train_per_quantile + test_per_quantile):
            selected_samples = random.sample(bin, train_per_quantile + test_per_quantile)
            train_samples += selected_samples[:train_per_quantile]
            test_samples += selected_samples[train_per_quantile:]
        else:
            # If bin is too small, add all to test set first, then fill train set if possible
            test_samples += bin[:test_per_quantile]
            if len(bin) > test_per_quantile:
                train_samples += bin[test_per_quantile:test_per_quantile+train_per_quantile]

    # Handle any remaining samples needed due to rounding
    remaining_train = n_train - len(train_samples)
    remaining_test = n_test - len(test_samples)
    
    all_remaining = [ex for bin in bins for ex in bin if ex not in train_samples and ex not in test_samples]
    random.shuffle(all_remaining)
    
    train_samples += all_remaining[:remaining_train]
    test_samples += all_remaining[remaining_train:remaining_train+remaining_test]

    return [(train_samples, test_samples)]