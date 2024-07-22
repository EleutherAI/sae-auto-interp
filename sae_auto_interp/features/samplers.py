import random
import torch
from ..logger import logger
from .features import FeatureRecord

class Sampler:

    def __call__(self, record, **kwargs):
        self.sample(record, **kwargs)

    def split_activation_quantiles(self, examples, n_quantiles):
        max_activation = examples[0].max_activation
        
        thresholds = [max_activation * i / n_quantiles for i in range(1, n_quantiles)]
        quantiles = [[] for _ in range(n_quantiles)]

        for example in examples:
            for i, threshold in enumerate(thresholds):
                if example.max_activation <= threshold:
                    quantiles[i].append(example)
                    break
            else:
                quantiles[-1].append(example)
        
        return quantiles

    def split_quantiles(self, examples, n_quantiles):
        n = len(examples)
        quantile_size = n // n_quantiles
        
        return [
            examples[i:i + quantile_size] if i < (n_quantiles - 1) * quantile_size
            else examples[i:]
            for i in range(0, n, quantile_size)
        ]

    def check_quantile(self, quantile, n_test):
        if len(quantile) < n_test:
            logger.error(f"Quantile has too few examples")
            raise ValueError(f"Quantile has too few examples")


class TopAndQuantilesSampler(Sampler):

    def sample(
        self,
        record: FeatureRecord,
        n_train=10,
        n_test=10,
        n_quantiles=3,
        seed=22,
    ):
        random.seed(seed)
        torch.manual_seed(seed)

        activation_quantiles = self.split_activation_quantiles(record.examples, n_quantiles)
        train_examples = random.sample(activation_quantiles[0], n_train)

        test_quantiles = activation_quantiles[1:]
        test_examples = []

        for quantile in test_quantiles:
            self.check_quantile(quantile, n_test)
            test_examples.append(random.sample(quantile, n_test))

        record.train = train_examples
        record.test = test_examples


class TopAndActivationQuantilesSampler(Sampler):

    def sample_top_and_activation_quantiles(
        self,
        record: FeatureRecord,
        n_train=10,
        n_test=5,
        n_quantiles=4,
        seed=22,
    ):
        random.seed(seed)
        torch.manual_seed(seed)
        
        # print(record, n_train, n_test, n_quantiles, seed, decode)
        train_examples = record.examples[:n_train]

        activation_quantiles = self.split_activation_quantiles(
            record.examples[n_train:], n_quantiles
        )

        test_examples = []

        for quantile in activation_quantiles:
            self.check_quantile(quantile, n_test)
            test_examples.append(random.sample(quantile, n_test))

        record.train = train_examples
        record.test = test_examples


class TopAndQuantilesSampler(Sampler):
    def sample(
        self,
        record: FeatureRecord,
        n_train=10,
        n_test=10,
        n_quantiles=4,
        seed=22,
    ):
        random.seed(seed)
        torch.manual_seed(seed)

        examples = record.examples

        # Sample n_train examples for training
        train_examples = examples[:n_train]
        remaining_examples = examples[n_train:]

        quantiles = self.split_quantiles(remaining_examples, n_quantiles)

        test_examples = []

        for quantile in quantiles:
            self.check_quantile(quantile, n_test)
            test_examples.append(random.sample(quantile, n_test))

        record.train = train_examples
        record.test = test_examples