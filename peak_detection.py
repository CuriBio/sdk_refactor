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

from constants import *
from exceptions import TooFewPeaksDetectedError
from exceptions import TwoPeaksInARowError
from exceptions import TwoValleysInARowError

TWITCH_WIDTH_PERCENTS = np.arange(10, 95, 5)
TWITCH_WIDTH_INDEX_OF_CONTRACTION_VELOCITY_START = np.where(TWITCH_WIDTH_PERCENTS == 10)[0]
TWITCH_WIDTH_INDEX_OF_CONTRACTION_VELOCITY_END = np.where(TWITCH_WIDTH_PERCENTS == 90)[0]


def peak_detector(
    filtered_magnetic_signal: NDArray[(2, Any), int],
    twitches_point_up: bool = True,
    is_magnetic_data: bool = True,
) -> Tuple[List[int], List[int]]:
    """Locates peaks and valleys and returns the indices.

    Args:
        filtered_magnetic_signal: a 2D array of the magnetic signal vs time data after it has gone through noise cancellation. It is assumed that the time values are in microseconds
        twitches_point_up: whether in the incoming data stream the biological twitches are pointing up (in the positive direction) or down
        is_magnetic_data: whether the incoming data stream is magnetic data or not
        sampling_period: Optional value indicating the period that magnetic data was sampled at. If not given, the sampling period will be calculated using the difference of the first two time indices

    Returns:
        A tuple containing a list of the indices of the peaks and a list of the indices of valleys
    """
    magnetic_signal: NDArray[int] = filtered_magnetic_signal[1, :] * -1 if is_magnetic_data else filtered_magnetic_signal[1, :]

    peak_invertor_factor, valley_invertor_factor = 1, -1
    if not twitches_point_up:
        peak_invertor_factor *= -1
        valley_invertor_factor *= -1

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
            f"A minimum of {min_number_valleys} peaks is required to extract twitch metrics, however only {len(peak_indices)} peak(s) were detected."
        )
    if len(valley_indices) < min_number_valleys:
        raise TooFewPeaksDetectedError(
            f"A minimum of {min_number_valleys} valleys is required to extract twitch metrics, however only {len(valley_indices)} valley(s) were detected."
        )


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
