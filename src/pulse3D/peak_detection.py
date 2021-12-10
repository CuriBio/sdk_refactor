# -*- coding: utf-8 -*-
"""Detecting peak and valleys of incoming Mantarray data."""
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from uuid import UUID

from nptyping import NDArray
import numpy as np
from scipy import signal

from .constants import *
from .exceptions import TooFewPeaksDetectedError
from .exceptions import TwoPeaksInARowError
from .exceptions import TwoValleysInARowError
from .metrics import *

TWITCH_WIDTH_PERCENTS = np.arange(10, 95, 5)
TWITCH_WIDTH_INDEX_OF_CONTRACTION_VELOCITY_START = np.where(TWITCH_WIDTH_PERCENTS == 10)[0]
TWITCH_WIDTH_INDEX_OF_CONTRACTION_VELOCITY_END = np.where(TWITCH_WIDTH_PERCENTS == 90)[0]


def peak_detector(
    filtered_magnetic_signal: NDArray[(2, Any), int],
    twitches_point_up: bool = True,
) -> Tuple[List[int], List[int]]:
    """Locates peaks and valleys and returns the indices.

    Args:
        filtered_magnetic_signal: a 2D array of the magnetic signal vs time data after it has gone through noise cancellation. It is assumed that the time values are in microseconds
        twitches_point_up: whether in the incoming data stream the biological twitches are pointing up (in the positive direction) or down
        sampling_period: Optional value indicating the period that magnetic data was sampled at. If not given, the sampling period will be calculated using the difference of the first two time indices

    Returns:
        A tuple containing a list of the indices of the peaks and a list of the indices of valleys
    """
    magnetic_signal: NDArray[int] = filtered_magnetic_signal[1, :]

    (peak_invertor_factor, valley_invertor_factor) = (1, -1) if twitches_point_up else (-1, 1)
    sampling_period_us = filtered_magnetic_signal[0, 1] - filtered_magnetic_signal[0, 0]

    max_possible_twitch_freq = 7
    min_required_samples_between_twitches = int(  # pylint:disable=invalid-name # (Eli 9/1/20): I can't think of a shorter name to describe this concept fully
        round(
            (1 / max_possible_twitch_freq) * MICRO_TO_BASE_CONVERSION / sampling_period_us,
            0,
        ),
    )

    # find required height of peaks
    max_height = np.max(magnetic_signal)
    min_height = np.min(magnetic_signal)
    max_prominence = abs(max_height - min_height)

    # find peaks and valleys
    peak_indices, _ = signal.find_peaks(
        magnetic_signal * peak_invertor_factor,
        width=min_required_samples_between_twitches / 2,
        distance=min_required_samples_between_twitches,
        prominence=max_prominence / 4,
    )

    valley_indices, properties = signal.find_peaks(
        magnetic_signal * valley_invertor_factor,
        width=min_required_samples_between_twitches / 2,
        distance=min_required_samples_between_twitches,
        prominence=max_prominence / 4,
    )

    left_ips = properties["left_ips"]
    right_ips = properties["right_ips"]

    # Patches error in B6 file for when two valleys are found in a single valley. If this is true left_bases, right_bases, prominences, and raw magnetic sensor data will also be equivalent to their previous value. This if statement indicates that the valley should be disregarded if the interpolated values on left and right intersection points of a horizontal line at the an evaluation height are equivalent. This would mean that the left and right sides of the peak and its neighbor peak align, indicating that it just one peak rather than two.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_widths.html#scipy.signal.peak_widths

    # Tanner (10/28/21): be careful modifying any of this while loop, it is currently not unit tested
    i = 1
    while i < len(valley_indices):
        if left_ips[i] == left_ips[i - 1] and right_ips[i] == right_ips[i - 1]:  # pragma: no cover
            valley_idx = valley_indices[i]
            valley_idx_last = valley_indices[i - 1]

            if magnetic_signal[valley_idx_last] >= magnetic_signal[valley_idx]:
                valley_indices = np.delete(valley_indices, i)
                left_ips = np.delete(left_ips, i)
                right_ips = np.delete(right_ips, i)
            else:  # pragma: no cover # (Anna 3/31/21): we don't have a case as of yet in which the first peak is higher than the second however know that it is possible and therefore aren't worried about code coverage in this case.
                valley_indices = np.delete(valley_indices, i - 1)
                left_ips = np.delete(left_ips, i - 1)
                right_ips = np.delete(right_ips, i - 1)
        else:
            i += 1

    return peak_indices, valley_indices


def too_few_peaks_or_valleys(
    peak_indices: NDArray[int],
    valley_indices: NDArray[int],
    min_number_peaks: int = MIN_NUMBER_PEAKS,
    min_number_valleys: int = MIN_NUMBER_VALLEYS,
) -> None:
    """Raise an error if there are too few peaks or valleys detected.

    Args:
        peak_indices: NDArray
            a 1D array of integers representing the indices of the peaks
        valley_indices: NDArray
            a 1D array of integeres representing the indices of the valleys
        min_number_peaks: int
            minimum number of required peaks
        min_number_valleys: int
            minumum number of required valleys
    Raises:
        TooFewPeaksDetectedError
    """
    if len(peak_indices) < min_number_peaks:
        raise TooFewPeaksDetectedError(
            f"A minimum of {min_number_peaks} peaks is required to extract twitch metrics, however only {len(peak_indices)} peak(s) were detected."
        )
    if len(valley_indices) < min_number_valleys:
        raise TooFewPeaksDetectedError(
            f"A minimum of {min_number_valleys} valleys is required to extract twitch metrics, however only {len(valley_indices)} valley(s) were detected."
        )


def find_twitch_indices(
    peak_and_valley_indices: Tuple[NDArray[int], NDArray[int]],
) -> Dict[int, Dict[UUID, Optional[int]]]:
    """Find twitches that can be analyzed.

    Sometimes the first and last peak in a trace can't be analyzed as a full twitch because not
    enough information is present.
    In order to be analyzable, a twitch needs to have a valley prior to it and another peak after it.

    Args:
        peak_and_valley_indices: a Tuple of 1D array of integers representing the indices of the
        peaks and valleys

    Returns:
        a dictionary in which the key is an integer representing the time points of all the peaks
        of interest and the value is an inner dictionary with various UUIDs of prior/subsequent
        peaks and valleys and their index values.
    """
    too_few_peaks_or_valleys(*peak_and_valley_indices)
    peak_indices, valley_indices = peak_and_valley_indices

    twitches: Dict[int, Dict[UUID, Optional[int]]] = {}

    starts_with_peak = peak_indices[0] < valley_indices[0]
    prev_feature_is_peak = starts_with_peak
    peak_idx, valley_idx = _find_start_indices(starts_with_peak)

    # check for two back-to-back features
    while peak_idx < len(peak_indices) and valley_idx < len(valley_indices):
        if prev_feature_is_peak:
            if valley_indices[valley_idx] > peak_indices[peak_idx]:
                raise TwoPeaksInARowError((peak_indices[peak_idx - 1], peak_indices[peak_idx]))
            valley_idx += 1
        else:
            if valley_indices[valley_idx] < peak_indices[peak_idx]:
                raise TwoValleysInARowError((valley_indices[valley_idx - 1], valley_indices[valley_idx]))
            peak_idx += 1
        prev_feature_is_peak = not prev_feature_is_peak

    if peak_idx < len(peak_indices) - 1:
        raise TwoPeaksInARowError((peak_indices[peak_idx], peak_indices[peak_idx + 1]))
    if valley_idx < len(valley_indices) - 1:
        raise TwoValleysInARowError((valley_indices[valley_idx], valley_indices[valley_idx + 1]))

    # compile dict of twitch information
    for itr_idx, itr_peak_index in enumerate(peak_indices):
        if itr_idx < peak_indices.shape[0] - 1:  # last peak
            if itr_idx == 0 and starts_with_peak:
                continue

            twitches[itr_peak_index] = {
                PRIOR_PEAK_INDEX_UUID: None if itr_idx == 0 else peak_indices[itr_idx - 1],
                PRIOR_VALLEY_INDEX_UUID: valley_indices[itr_idx - 1 if starts_with_peak else itr_idx],
                SUBSEQUENT_PEAK_INDEX_UUID: peak_indices[itr_idx + 1],
                SUBSEQUENT_VALLEY_INDEX_UUID: valley_indices[itr_idx if starts_with_peak else itr_idx + 1],
            }

    return twitches


def data_metrics(
    peak_and_valley_indices: Tuple[NDArray[int], NDArray[int]],
    filtered_data: NDArray[(2, Any), int],
    rounded: bool = False,
    metrics_to_create: Iterable[UUID] = ALL_METRICS,
) -> Tuple[Dict[int, Dict[UUID, Any]], Dict[UUID, Any]]:
    # pylint:disable=too-many-locals # Eli (9/8/20): there are a lot of metrics to calculate that need local variables
    """Find all data metrics for individual twitches and averages.

    Args:
        peak_and_valley_indices: a tuple of integer value arrays representing the time indices of peaks and valleys within the data
        filtered_data: a 2D array of the time and voltage data after it has gone through noise cancellation
        rounded: whether to round estimates to the nearest int
        metrics_to_create: list of desired metrics
    Returns:
        main_twitch_dict: a dictionary of individual peak metrics in which the twitch timepoint is accompanied by a dictionary in which the UUIDs for each twitch metric are the key and with its accompanying value as the value. For the Twitch Width metric UUID, another dictionary is stored in which the key is the percentage of the way down and the value is another dictionary in which the UUIDs for the rising coord, falling coord or width value are stored with the value as an int for the width value or a tuple of ints for the x/y coordinates
        aggregate_dict: a dictionary of entire metric statistics. Most metrics have the stats underneath the UUID, but for twitch widths, there is an additional dictionary where the percent of repolarization is the key
    """
    # create main dictionaries
    main_twitch_dict: Dict[int, Dict[UUID, Any]] = dict()
    aggregate_dict: Dict[UUID, Any] = dict()

    # get values needed for metrics creation
    twitch_indices = find_twitch_indices(peak_and_valley_indices)
    num_twitches = len(twitch_indices)
    time_series = filtered_data[0, :]

    metric_parameters = {
        "peak_and_valley_indices": peak_and_valley_indices,
        "filtered_data": filtered_data,
        "twitch_indices": twitch_indices,
    }

    # create top level dict
    twitch_peak_indices = tuple(twitch_indices.keys())
    main_twitch_dict = {time_series[twitch_peak_indices[i]]: dict() for i in range(num_twitches)}

    # Krisian 10/26/21
    # dictionary of metric functions
    # this could probably be made cleaner at some point
    metric_mapper: Dict[UUID, BaseMetric]
    metric_mapper = {
        AMPLITUDE_UUID: TwitchAmplitude(rounded=rounded),
        AUC_UUID: TwitchAUC(rounded=rounded),
        BASELINE_TO_PEAK_UUID: TwitchPeakToBaseline(rounded=rounded, is_contraction=True),
        CONTRACTION_TIME_UUID: TwitchPeakTime(rounded=rounded, is_contraction=True),
        CONTRACTION_VELOCITY_UUID: TwitchVelocity(rounded=rounded, is_contraction=True),
        FRACTION_MAX_UUID: TwitchFractionAmplitude(rounded=rounded),
        IRREGULARITY_INTERVAL_UUID: TwitchIrregularity(rounded=rounded),
        PEAK_TO_BASELINE_UUID: TwitchPeakToBaseline(rounded=rounded, is_contraction=False),
        RELAXATION_TIME_UUID: TwitchPeakTime(rounded=rounded, is_contraction=False),
        RELAXATION_VELOCITY_UUID: TwitchVelocity(rounded=rounded, is_contraction=False),
        TWITCH_FREQUENCY_UUID: TwitchFrequency(rounded=rounded),
        TWITCH_PERIOD_UUID: TwitchPeriod(rounded=rounded),
        WIDTH_UUID: TwitchWidth(rounded=rounded),
    }

    for metric_id in metrics_to_create:
        metric = metric_mapper[metric_id]
        estimate = metric.fit(**metric_parameters)
        metric.add_per_twitch_metrics(
            main_twitch_dict=main_twitch_dict, metric_id=metric_id, metrics=estimate
        )
        metric.add_aggregate_metrics(aggregate_dict=aggregate_dict, metric_id=metric_id, metrics=estimate)

    return main_twitch_dict, aggregate_dict


def _find_start_indices(starts_with_peak: bool) -> Tuple[int, int]:
    """Find start indices for peaks and valleys.

    Args:
        starts_with_peak: bool indicating whether or not a peak rather than a valley comes first

    Returns:
        peak_idx: peak start index
        valley_idx: valley start index
    """
    peak_idx = 0
    valley_idx = 0
    if starts_with_peak:
        peak_idx += 1
    else:
        valley_idx += 1

    return peak_idx, valley_idx
