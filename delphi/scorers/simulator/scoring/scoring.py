import asyncio
import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from simple_parsing import Serializable

from delphi.latents import ActivatingExample, NonActivatingExample

from ..activations.activation_records import ActivationRecord
from ..scoring.simulator import NeuronSimulator


@dataclass
class SequenceSimulation(Serializable):
    """The result of a simulation of neuron activations on one text sequence."""

    tokens: list[str]
    """The sequence of tokens that was simulated."""
    expected_activations: list[float]
    """Expected value of the possibly-normalized activation for
    each token in the sequence."""
    distribution_values: list[list[float]]
    """
    For each token in the sequence, a list of values from the discrete
    distribution of activations produced from simulation.
    Tokens will be included here if and only if they are in the top K=15
    tokens predicted by the simulator, and excluded otherwise.

    May be transformed to another unit by calibration. When we simulate a neuron,
    we produce a discrete distribution with values in the arbitrary discretized space
    of the neuron, e.g. 10% chance of 0, 70% chance of 1, 20% chance of 2.
    Which we store as distribution_values = [0, 1, 2],
    distribution_probabilities = [0.1, 0.7, 0.2]. When we transform the distribution to
    the real activation units, we can correspondingly transform the values of this
    distribution to get a distribution in the units of the neuron. e.g. if the mapping
    from the discretized space to the real activation unit of the neuron is f(x) = x/2,
    then the distribution becomes 10% chance of 0, 70% chance of 0.5, 20% chance of 1.
    Which we store as distribution_values = [0, 0.5, 1], distribution_probabilities =
    [0.1, 0.7, 0.2].
    """
    distribution_probabilities: list[list[float]]
    """
    For each token in the sequence, the probability of the corresponding value in
    distribution_values.
    """

    uncalibrated_simulation: Optional["SequenceSimulation"] = None
    """The result of the simulation before calibration."""


@dataclass
class ScoredSequenceSimulation(Serializable):
    """
    SequenceSimulation result with a score (for that sequence only) and ground truth
    activations.
    """

    distance: float | int
    """Quantile or neighbor distance"""
    simulation: SequenceSimulation
    """The result of a simulation of neuron activations."""
    true_activations: list[float]
    """Ground truth activations on the sequence (not normalized)"""
    ev_correlation_score: float
    """
    Correlation coefficient between the expected values of the normalized activations
    from the simulation and the unnormalized true activations of the neuron on the text
    sequence.
    """
    rsquared_score: Optional[float] = None
    """R^2 of the simulated activations."""
    absolute_dev_explained_score: Optional[float] = None
    """
    Score based on absolute difference between real and simulated activations.
    absolute_dev_explained_score = 1 - mean(abs(real-predicted))/ mean(abs(real))
    """


@dataclass
class ScoredSimulation(Serializable):
    """Result of scoring a neuron simulation on multiple sequences."""

    distance: int
    """Distance of the sequence from the original sequence."""

    scored_sequence_simulations: list[ScoredSequenceSimulation]
    """ScoredSequenceSimulation for each sequence"""
    ev_correlation_score: Optional[float] = None
    """
    Correlation coefficient between the expected values of the normalized activations
    from the simulation and the unnormalized true activations on a dataset created from
    all score_results. (Note that this is not equivalent to averaging across sequences.)
    """
    rsquared_score: Optional[float] = None
    """R^2 of the simulated activations."""
    absolute_dev_explained_score: Optional[float] = None
    """
    Score based on absolute difference between real and simulated activations.
    absolute_dev_explained_score = 1 - mean(abs(real-predicted))/ mean(abs(real)).
    """


def correlation_score(
    real_activations: list[float] | np.ndarray,
    predicted_activations: list[float] | np.ndarray,
) -> float:
    return float(np.corrcoef(real_activations, predicted_activations)[0, 1])


def score_from_simulation(
    real_activations: ActivationRecord,
    simulation: SequenceSimulation,
    score_function: Callable[
        [list[float] | np.ndarray, list[float] | np.ndarray], float
    ],
) -> float:
    if len(simulation.expected_activations) > 0:
        return score_function(
            real_activations.activations, simulation.expected_activations
        )
    else:
        return 0


def rsquared_score_from_lists(
    real_activations: list[float] | np.ndarray,
    predicted_activations: list[float] | np.ndarray,
) -> float:
    return float(
        1
        - np.mean(
            np.square(np.array(real_activations) - np.array(predicted_activations))
        )
        / np.mean(np.square(np.array(real_activations)))
    )


def absolute_dev_explained_score_from_lists(
    real_activations: list[float] | np.ndarray,
    predicted_activations: list[float] | np.ndarray,
) -> float:
    return float(
        1
        - np.mean(np.abs(np.array(real_activations) - np.array(predicted_activations)))
        / np.mean(np.abs(np.array(real_activations)))
    )


async def _simulate_and_score_list(
    simulator: NeuronSimulator, example: ActivatingExample | NonActivatingExample
) -> ScoredSequenceSimulation:
    """Score an explanation of a neuron by how well it predicts activations
    on a sentence."""

    simulation = await simulator.simulate(example.str_tokens)
    logging.debug(simulation)
    rsquared_score = 0
    absolute_dev_explained_score = 0

    match example:
        case ActivatingExample():
            distance = example.quantile
            activating = True
        case NonActivatingExample():
            distance = example.distance
            activating = False
    activation_record = ActivationRecord(
        example.str_tokens, example.activations.tolist()
    )
    scored_sequence_simulation = ScoredSequenceSimulation(
        distance=distance,
        simulation=simulation,
        true_activations=activation_record.activations,
        ev_correlation_score=(
            score_from_simulation(activation_record, simulation, correlation_score)
            if activating
            else 0
        ),  # can't do EV when truth is 0
        rsquared_score=rsquared_score,
        absolute_dev_explained_score=absolute_dev_explained_score,
    )
    return scored_sequence_simulation


def fix_nan(val):
    if np.isnan(val):
        return "nan"
    else:
        return float(val)


def aggregate_scored_sequence_simulations(
    scored_sequence_simulations: list[ScoredSequenceSimulation],
    distance: int,
) -> ScoredSimulation:
    """
    Aggregate a list of scored sequence simulations. The logic for doing this is
    non-trivial for EV scores, since we want to calculate the correlation over all
    activations from all lists at once rather than simply averaging
    per-list correlations.
    """
    all_true_activations: list[float] = []
    all_expected_values: list[float] = []
    for scored_sequence_simulation in scored_sequence_simulations:
        all_true_activations.extend(scored_sequence_simulation.true_activations or [])
        all_expected_values.extend(
            scored_sequence_simulation.simulation.expected_activations
        )
    ev_correlation_score = (
        correlation_score(all_true_activations, all_expected_values)
        if (len(all_true_activations) > 0 and len(all_expected_values) > 0)
        else 0
    )
    rsquared_score = (
        rsquared_score_from_lists(all_true_activations, all_expected_values)
        if (len(all_true_activations) > 0 and len(all_expected_values) > 0)
        else 0
    )
    absolute_dev_explained_score = (
        absolute_dev_explained_score_from_lists(
            all_true_activations, all_expected_values
        )
        if (len(all_true_activations) > 0 and len(all_expected_values) > 0)
        else 0
    )

    ev_correlation_score = fix_nan(ev_correlation_score)
    absolute_dev_explained_score = fix_nan(absolute_dev_explained_score)
    rsquared_score = fix_nan(rsquared_score)

    return ScoredSimulation(
        distance=distance,
        scored_sequence_simulations=scored_sequence_simulations,
        ev_correlation_score=ev_correlation_score,
        rsquared_score=rsquared_score,
        absolute_dev_explained_score=absolute_dev_explained_score,
    )


async def simulate_and_score(
    simulator: NeuronSimulator,
    activation_records: list[ActivatingExample],
    non_activation_records: list[NonActivatingExample],
) -> ScoredSimulation:
    """
    Score an explanation of a neuron by how well it predicts activations
    on the given text lists.
    """
    scored_sequence_simulations = await asyncio.gather(
        *[
            _simulate_and_score_list(simulator, activation_record)
            for activation_record in activation_records
        ]
    )
    if len(non_activation_records) > 0:
        non_activating_scored_seq_simulations = await asyncio.gather(
            *[
                _simulate_and_score_list(simulator, non_activation_record)
                for non_activation_record in non_activation_records
            ]
        )
    values = []
    scores_per_distance = {}
    for sequence in scored_sequence_simulations:
        distance = sequence.distance
        if distance not in scores_per_distance:
            scores_per_distance[distance] = []
        scores_per_distance[distance].append(sequence)

    for distance, sequences in scores_per_distance.items():
        values.append(aggregate_scored_sequence_simulations(sequences, distance + 1))

    # plus an aggregate score for all distances
    values.append(aggregate_scored_sequence_simulations(scored_sequence_simulations, 0))

    # plus an aggregate score for non-activated + all distances
    all_data = scored_sequence_simulations + non_activating_scored_seq_simulations
    values.append(aggregate_scored_sequence_simulations(all_data, -1))

    return values
