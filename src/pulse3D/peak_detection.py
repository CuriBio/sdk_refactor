# -*- coding: utf-8 -*-
"""Detecting peak and valleys of incoming Mantarray data."""
import logging
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

log = logging.getLogger(__name__)


def peak_detector(
    filtered_magnetic_signal: NDArray[(2, Any), int],
    twitches_point_up: bool = True,
    start_time: float = 0,
    end_time: float = np.inf,
    prominence_factors: Tuple[Union[int, float], Union[int, float]] = DEFAULT_PROMINENCE_FACTORS,
    width_factors: Tuple[Union[int, float], Union[int, float]] = DEFAULT_WIDTH_FACTORS,
) -> Tuple[List[int], List[int]]:
    """Locates peaks and valleys and returns the indices.

    Args:
        filtered_magnetic_signal: a 2D array of the magnetic signal vs time data after it has gone through noise cancellation. It is assumed that the time values are in microseconds
        twitches_point_up: whether in the incoming data stream the biological twitches are pointing up (in the positive direction) or down
        start_time (float): start time of windowed analysis, in seconds. Default value = 0 seconds.
        end_time (float): end time of windowed analysis, in seconds.  Default value = Inf seconds.
        prominence_factors: (int/float, int/float) scaling factors for peak/valley prominences.  Larger values make peak-finding more flexible by reducing minimum-required prominence
        width_factors: (int/float, int/float) scaling factors for peak/valley widths.  Larger values make peak-finding more flexible by reducing minimum-required width

    Returns:
        A tuple containing a list of the indices of the peaks and a list of the indices of valleys
    """
    # make sure width and prominece factors are a tuple of two ints
    width_factors = _format_factors(width_factors)
    prominence_factors = _format_factors(prominence_factors)

    time_signal: NDArray[float] = filtered_magnetic_signal[0, :]
    magnetic_signal: NDArray[float] = filtered_magnetic_signal[1, :]

    max_time = time_signal[-1]
    start_time = np.max([0, start_time])
    end_time = np.min([end_time, max_time / MICRO_TO_BASE_CONVERSION])
    # if provided end time is less than or equal to start time, reset  # TODO figure out how to handle this
    if end_time <= start_time:
        end_time = np.inf

    peak_invertor_factor, valley_invertor_factor = (1, -1) if twitches_point_up else (-1, 1)
    sampling_period_us = filtered_magnetic_signal[0, 1] - filtered_magnetic_signal[0, 0]

    max_possible_twitch_freq = 7
    min_required_samples_between_twitches = int(
        round((1 / max_possible_twitch_freq) * MICRO_TO_BASE_CONVERSION / sampling_period_us, 0),
    )
    # find required height of peaks
    max_height = np.max(magnetic_signal)
    min_height = np.min(magnetic_signal)
    max_prominence = abs(max_height - min_height)

    # find peaks and valleys
    peak_indices, _ = signal.find_peaks(
        magnetic_signal * peak_invertor_factor,
        width=min_required_samples_between_twitches / width_factors[0],
        distance=min_required_samples_between_twitches,
        prominence=max_prominence / prominence_factors[0],
    )

    valley_indices, properties = signal.find_peaks(
        magnetic_signal * valley_invertor_factor,
        width=min_required_samples_between_twitches / width_factors[1],
        distance=min_required_samples_between_twitches,
        prominence=max_prominence / prominence_factors[1],
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

    # don't perform windowing of twitches unless requested
    if start_time > 0 or end_time < max_time:
        peak_times = time_signal[peak_indices] / MICRO_TO_BASE_CONVERSION
        # identify peaks within time window
        filtered_peaks = np.where((peak_times >= start_time) & (peak_times <= end_time))[0]
        peak_indices = peak_indices[filtered_peaks]

        valley_times = time_signal[valley_indices] / MICRO_TO_BASE_CONVERSION
        # identify valleys within time window
        filtered_valleys = np.where((valley_times >= start_time) & (valley_times <= end_time))[0]
        valley_indices = valley_indices[filtered_valleys]

    return peak_indices, valley_indices


def _format_factors(factors):
    if isinstance(factors, int):
        expected_prominences = (factors, factors)
    else:
        expected_prominences = factors if len(factors) == 2 else (factors[0], factors[0])
    return expected_prominences


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
    peak_indices, valley_indices = peak_and_valley_indices

    if len(peak_indices) < MIN_NUMBER_PEAKS:
        raise TooFewPeaksDetectedError(
            f"A minimum of {MIN_NUMBER_PEAKS} peaks is required to extract twitch metrics, however only {len(peak_indices)} peak(s) were detected."
        )
    if len(valley_indices) < MIN_NUMBER_VALLEYS:
        raise TooFewPeaksDetectedError(
            f"A minimum of {MIN_NUMBER_VALLEYS} valleys is required to extract twitch metrics, however only {len(valley_indices)} valley(s) were detected."
        )

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
    twitch_width_percents: NDArray = DEFAULT_TWITCH_WIDTH_PERCENTS,
    baseline_widths_to_use: Tuple[int, ...] = DEFAULT_BASELINE_WIDTHS,
) -> Tuple[DataFrame, DataFrame]:
    # pylint:disable=too-many-locals # Eli (9/8/20): there are a lot of metrics to calculate that need local variables
    """Find all data metrics for individual twitches and averages.

    Args:
        peak_and_valley_indices: a tuple of integer value arrays representing the time indices of peaks and valleys within the data
        filtered_data: a 2D array of the time and voltage data after it has gone through noise cancellation
        rounded: whether to round estimates to the nearest int
        metrics_to_create: list of desired metrics
        twitch_width_percents: twitch width percents including those input by user
        baseline_widths_to_use: twitch widths to use as baseline metrics
    Returns:
        per_twitch_df: a dictionary of individual peak metrics in which the twitch timepoint is accompanied by a dictionary in which the UUIDs for each twitch metric are the key and with its accompanying value as the value. For the Twitch Width metric UUID, another dictionary is stored in which the key is the percentage of the way down and the value is another dictionary in which the UUIDs for the rising coord, falling coord or width value are stored with the value as an int for the width value or a tuple of ints for the x/y coordinates
        aggregate_df: a dictionary of entire metric statistics. Most metrics have the stats underneath the UUID, but for twitch widths, there is an additional dictionary where the percent of repolarization is the key
    """
    # get values needed for metrics creation
    twitch_indices = find_twitch_indices(peak_and_valley_indices)

    metric_parameters = {
        "peak_and_valley_indices": peak_and_valley_indices,
        "filtered_data": filtered_data,
        "twitch_indices": twitch_indices,
    }

    dfs = init_dfs(twitch_indices.keys(), twitch_widths_range=twitch_width_percents)

    # Kristian (10/26/21): dictionary of metric functions. this could probably be made cleaner at some point
    metric_mapper: Dict[UUID, BaseMetric] = {
        AMPLITUDE_UUID: TwitchAmplitude(rounded=rounded),
        AUC_UUID: TwitchAUC(rounded=rounded, twitch_width_percents=twitch_width_percents),
        BASELINE_TO_PEAK_UUID: TwitchPeakTime(
            rounded=rounded,
            is_contraction=True,
            twitch_width_percents=(baseline_widths_to_use[0], 100 - baseline_widths_to_use[0]),
        ),
        CONTRACTION_TIME_UUID: TwitchPeakTime(
            rounded=rounded, is_contraction=True, twitch_width_percents=twitch_width_percents
        ),
        CONTRACTION_VELOCITY_UUID: TwitchVelocity(
            rounded=rounded, is_contraction=True, twitch_width_percents=twitch_width_percents
        ),
        FRACTION_MAX_UUID: TwitchFractionAmplitude(),
        IRREGULARITY_INTERVAL_UUID: TwitchIrregularity(rounded=rounded),
        PEAK_TO_BASELINE_UUID: TwitchPeakTime(
            rounded=rounded,
            is_contraction=False,
            twitch_width_percents=(baseline_widths_to_use[1], 100 - baseline_widths_to_use[1]),
        ),
        RELAXATION_TIME_UUID: TwitchPeakTime(
            rounded=rounded, is_contraction=False, twitch_width_percents=twitch_width_percents
        ),
        RELAXATION_VELOCITY_UUID: TwitchVelocity(rounded=rounded, is_contraction=False),
        TWITCH_FREQUENCY_UUID: TwitchFrequency(rounded=rounded),
        TWITCH_PERIOD_UUID: TwitchPeriod(rounded=rounded),
        WIDTH_UUID: TwitchWidth(rounded=rounded, twitch_width_percents=twitch_width_percents),
    }

    # add scalar metrics to corresponding DataFrames
    for metric_type, metrics in CALCULATED_METRICS.items():
        per_twitch_df = dfs["per_twitch"][metric_type]
        aggregate_df = dfs["aggregate"][metric_type]
        # sort first to improve performance
        per_twitch_df.sort_index(inplace=True)
        aggregate_df.sort_index(inplace=True)

        for metric_id in metrics:
            if metric_id in metrics_to_create:
                metric = metric_mapper[metric_id]
                try:
                    estimate = metric.fit(**metric_parameters)
                except Exception:  # nosec B110
                    continue
                metric.add_per_twitch_metrics(per_twitch_df, metric_id, estimate)
                metric.add_aggregate_metrics(aggregate_df, metric_id, estimate)

    per_twitch_df = concat([dfs["per_twitch"][j] for j in dfs["per_twitch"].keys()], axis=1)
    aggregate_df = concat([dfs["aggregate"][j] for j in dfs["aggregate"].keys()], axis=1)

    return per_twitch_df, aggregate_df


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


def init_dfs(indices: Iterable[int] = [], twitch_widths_range: Tuple[int, ...] = DEFAULT_TWITCH_WIDTHS):
    """Initialize empty dataframes for metrics computations.

    Note: scalar metrics are those representing a single value per twitch (e.g. AUC, AMPLITUDE, etc.)
          by-width metrics are those such as twitch-width, time-to-percent contraction / relaxation

    Args:
        indices (List[int]): list of twitch indices

    Returns:
        data_frames (Dict): keys correspond to initialized per-twitch or aggregate dataframes, on a scalar, or by-width basis
    """
    # per-twitch metrics data-frames
    per_twitch_scalar = pd.DataFrame(index=indices, columns=CALCULATED_METRICS["scalar"])
    per_twitch_scalar.columns = per_twitch_scalar.sort_index(axis=1, level=[0], ascending=[True]).columns

    columns = pd.MultiIndex.from_product(
        [CALCULATED_METRICS["by_width"], twitch_widths_range], names=["metric", "width"]
    )
    per_twitch_by_width = pd.DataFrame(index=indices, columns=columns)
    per_twitch_by_width.columns = per_twitch_by_width.sort_index(
        axis=1, level=[0, 1], ascending=[True, True]
    ).columns

    # aggregate metrics data-frames
    columns = pd.MultiIndex.from_product(
        [CALCULATED_METRICS["scalar"], ["n", "Mean", "StDev", "CoV", "SEM", "Min", "Max"]],
        names=["metric", "statistic"],
    )
    aggregate_scalar = pd.DataFrame(index=[0], columns=columns)
    aggregate_scalar.columns = aggregate_scalar.sort_index(
        axis=1, level=[0, 1], ascending=[True, True]
    ).columns

    columns = pd.MultiIndex.from_product(
        [
            CALCULATED_METRICS["by_width"],
            twitch_widths_range,
            ["n", "Mean", "StDev", "CoV", "SEM", "Min", "Max"],
        ],
        names=["metric", "width", "statistic"],
    )
    aggregate_by_width = pd.DataFrame(index=[0], columns=columns)
    aggregate_by_width.columns = aggregate_by_width.sort_index(
        axis=1, level=[0, 1, 2], ascending=[True, True, True]
    ).columns

    data_frames = {
        "per_twitch": {"scalar": per_twitch_scalar, "by_width": per_twitch_by_width},
        "aggregate": {"scalar": aggregate_scalar, "by_width": aggregate_by_width},
    }

    return data_frames


def concat(dfs, axis=0, *args, **kwargs):
    """Wrap `pandas.concat` to concatenate pandas objects even if they have
    unequal number of levels on concatenation axis.

    Levels containing empty strings are added from below (when concatenating along
    columns) or right (when concateniting along rows) to match the maximum number
    found in the dataframes.

    Parameters
    ----------
    dfs : Iterable
        Dataframes that must be concatenated.
    axis : int, optional
        Axis along which concatenation must take place. The default is 0.

    Returns
    -------
    pd.DataFrame
        Concatenated Dataframe.

    Notes
    -----
    Any arguments and kwarguments are passed onto the `pandas.concat` function.

    See Also
    --------
    pandas.concat
    """

    def index(df):
        return df.columns if axis == 1 else df.index

    want = np.max([index(df).nlevels for df in dfs])

    def add_levels(df):
        need = want - index(df).nlevels
        if need > 0:
            df = pd.concat([df], keys=[("",) * need], axis=axis)  # prepend empty levels
            for i in range(want - need):  # move empty levels to bottom
                df = df.swaplevel(i, i + need, axis=axis)
        return df

    dfs = [add_levels(df) for df in dfs]
    return pd.concat(dfs, axis=axis, *args, **kwargs)


def get_windowed_peaks_valleys(
    start_idx: int, end_idx: int, peaks: NDArray, valleys: NDArray
) -> Tuple[NDArray, NDArray]:
    """Convert peaks and valleys to fit window of analysis.

    Args:
        start_idx: int start index to window of analysis to substract from peak/valley indices
        end_idx: int end index to window of analysis to contain only relevant peak/valley indices
        peaks: NDArray contains new peak indices to fit window of analysis
        valleys: NDArray contains new valley indices to fit window of analysis

    Returns:
        peaks: NDArray of peak indices
        valleys: NDArray of valley indices
    """
    windowed_indices = end_idx - start_idx
    # remove up to starting index to make start time index 0
    sub_peaks = np.subtract(peaks, start_idx)
    sub_valleys = np.subtract(valleys, start_idx)
    # remove indices greater than max windowed index
    filtered_peaks = np.where((sub_peaks < windowed_indices) & (sub_peaks >= 0))[0]
    filtered_valleys = np.where((sub_valleys < windowed_indices) & (sub_valleys >= 0))[0]
    # grab indices from original sub peaks and valleys
    return sub_peaks[filtered_peaks], sub_valleys[filtered_valleys]
