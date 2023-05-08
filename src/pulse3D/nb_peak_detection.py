# -*- coding: utf-8 -*-
"""Detecting peak and valleys of incoming Mantarray data."""

from typing import Any
from typing import Optional
from typing import Tuple

from nptyping import NDArray
import numpy as np
from pulse3D.exceptions import InvalidValleySearchDurationError
from pulse3D.exceptions import TooFewPeaksDetectedError
from pulse3D.transforms import get_time_window_indices
from scipy import signal
from scipy.optimize import curve_fit

from .constants import DEFAULT_NB_HEIGHT_FACTOR
from .constants import DEFAULT_NB_NOISE_PROMINENCE_FACTOR
from .constants import DEFAULT_NB_RELATIVE_PROMINENCE_FACTOR
from .constants import DEFAULT_NB_UPSLOPE_DUR
from .constants import DEFAULT_NB_UPSLOPE_NOISE_ALLOWANCE_DUR
from .constants import DEFAULT_NB_VALLEY_SEARCH_DUR
from .constants import DEFAULT_NB_WIDTH_FACTORS
from .constants import MIN_NUMBER_PEAKS


def quadratic(x, a, b, c):
    return a * (x**2) + b * x + c


# TODO ? prefix args with peak_ or valley_ so it's more clear which they affect
def noise_based_peak_finding(
    tissue_data: NDArray[(2, Any), float],
    start_time: float = 0,
    end_time: float = np.inf,
    noise_prominence_factor: float = DEFAULT_NB_NOISE_PROMINENCE_FACTOR,
    relative_prominence_factor: Optional[float] = DEFAULT_NB_RELATIVE_PROMINENCE_FACTOR,
    width_factors: Tuple[float, float] = DEFAULT_NB_WIDTH_FACTORS,
    height_factor: float = DEFAULT_NB_HEIGHT_FACTOR,
    max_frequency: Optional[float] = None,
    valley_search_duration: float = DEFAULT_NB_VALLEY_SEARCH_DUR,
    upslope_duration: float = DEFAULT_NB_UPSLOPE_DUR,
    upslope_noise_allowance_duration: float = DEFAULT_NB_UPSLOPE_NOISE_ALLOWANCE_DUR,
):
    """
    Args:
        tissue_data: Waveform Amplitude (force/displacement/etc.) v. Time array. Time values should be in seconds. Data should be interpolated and normalized
        start_time: The earliest timepoint to consider
        end_time: The greatest timepoint to consider
        noise_prominence_factor: The minimum required SNR of a peak
        relative_prominence_factor: If specified, the prominence of each peak relative to the tallest peak will be taken into consideration.
            If this falls below the noise-based prominence threshold determined by `noise_prominence_factor`, that will be used instead.
        width_factor: The minimum and maximum width in seconds of a peak required to be considered
        height_factor: The minimum height of a peak required to be considered. This should be given in the unit of measure as the Waveform Amplitude values in `tissue_data`
        max_frequency: The maximum frequency (Hz) at which a peak can occur. Specifically, this is used to calculate the minimum required distance between adjacent peaks.
            For example, if the value given is 1, then at most peaks will occur at 1Hz. In other words, the peaks will be no closer than 1 second.
            If not specified, the sampling frequency is used instead, which means that every point can be considered a peak.
            If a value is given that exceeds the sampling frequency, the sampling frequency is used instead.
        valley_search_duration: The duration of time in seconds prior to a peak in which to search for a valley.
            If this window includes a previous peak then for that peak the window will automatically be shortened
        upslope_duration: The min duration of time in seconds through which the waveform values must continuously rise in order to be considered an upslope.
        upslope_noise_allowance_duration: The max duration of time in seconds in the upslope in which there is an amplitude decrease which can be tolerated within a single upslope.

    Returns:
        A tuple containing an of the indices of the peaks and an array of the indices of valleys
    """
    window_indices = get_time_window_indices(tissue_data[0], start_time, end_time)
    time_axis, waveform = tissue_data[:, window_indices]

    # TODO split into subfunctions, one for finding peaks and the other for finding valleys

    # extract sample frequency from time_axis (assumes sampling freq is constant)
    sample_freq = 1 / (time_axis[1] - time_axis[0])

    if max_frequency:
        # if max freq is greater than the sampling freq, use sampling freq instead
        max_frequency = min(max_frequency, sample_freq)
    else:
        # if no max freq given, use sampling freq
        max_frequency = sample_freq

    # set estimate of peak to peak noise amplitude is 10uN for average recording
    default_noise = 10
    default_prom = 5

    # find peaks with this estimated amplitude
    peaks, _ = signal.find_peaks(waveform, prominence=default_prom * default_noise)

    # if first attempt finds no peaks as they are too small, retry with smaller prominence
    # this approach should return a list of peak indices even if no true peaks exist as it will terminate at a prominence of 1.
    correction_factor = 1
    while len(peaks) == 0 and correction_factor <= default_prom:
        peaks, _ = signal.find_peaks(waveform, prominence=(default_prom - correction_factor) * default_noise)

        correction_factor += 1

    if (num_peaks := len(peaks)) < MIN_NUMBER_PEAKS:
        raise TooFewPeaksDetectedError(
            f"A minimum of {MIN_NUMBER_PEAKS} peaks is required to extract twitch metrics, however only {num_peaks} peak(s) were detected."
        )

    # use peaks to extract waveform segments from which noise data can be extracted - control over this could be given to the user if required
    segment_size = 10

    while peaks[-1] + segment_size > len(waveform):
        # possible bug deletes only peak and you get a len(list) == 0
        peaks = np.delete(peaks, -1)

    noise_segements = np.array([[waveform[i] for i in range(peak, peak + segment_size)] for peak in peaks])
    time_segments = np.array([[time_axis[i] for i in range(peak, peak + segment_size)] for peak in peaks])

    # fit quadratic to bring noise to baseline and remove peak information
    # fit_models = [curve_fit(quadratic, time, signal) for time, signal in zip(time_segments,noise_segements)]
    # quad_fit = [quadratic(i,*popt[0]) for i,popt in zip(time_segments, fit_models)]
    quad_fit = [
        quadratic(time, *curve_fit(quadratic, time, signal)[0])
        for time, signal in zip(time_segments, noise_segements)
    ]

    # baseline correct with quadratic fits
    noise_segements_corrected = np.array([noise - fit for noise, fit in zip(noise_segements, quad_fit)])

    # extract peak to peak noise for each segment and average
    noise_amplitude_from_data = np.average(
        np.max(noise_segements_corrected, axis=1) - np.min(noise_segements_corrected, axis=1)
    )

    # use either set prominence or calculate the relative prominence factor
    if relative_prominence_factor:
        max_peak_prom = (waveform.max() - waveform.min()) / noise_amplitude_from_data
        relative_prom = max_peak_prom * relative_prominence_factor
        # compare relative prom to static prom factor and use the larger value
        noise_prominence_factor = max(relative_prom, noise_prominence_factor)

    # refind peaks with the identified peak to peak values and user defined limits
    peaks, _ = signal.find_peaks(
        waveform,
        prominence=noise_prominence_factor * noise_amplitude_from_data,
        width=(width_factors[0] * sample_freq, width_factors[1] * sample_freq),
        height=height_factor,
        distance=sample_freq // max_frequency,
    )

    if (num_peaks := len(peaks)) < MIN_NUMBER_PEAKS:
        raise TooFewPeaksDetectedError(
            f"A minimum of {MIN_NUMBER_PEAKS} peaks is required to extract twitch metrics, however only {num_peaks} peak(s) were detected."
        )

    # the valley search size of the initial peak must not extend back beyond the initial timepoint, so remove any peaks that are too close to the start
    segment_size = int(valley_search_duration * sample_freq)
    while len(peaks) != 0 and peaks[0] - segment_size < 0:
        peaks = np.delete(peaks, 0)

    if len(peaks) == 0:
        raise InvalidValleySearchDurationError()

    # generate localised search windows based on peak positions
    padded_peaks = np.pad(peaks, (1, 0), constant_values=0)
    search_windows = np.array(
        [
            peak - padded_peak
            for peak, padded_peak in zip(np.pad(peaks, (0, 1), constant_values=len(waveform)), padded_peaks)
        ]
    )

    # if a window is smaller than the segment size then use this else use the defined segment size
    search_windows[search_windows > segment_size] = segment_size

    # generate waveform segments
    valley_segments = [
        np.array([waveform[i] for i in range(peak - segment, peak)])
        for peak, segment in zip(peaks, search_windows)
    ]

    upslope_num_samples = upslope_duration * sample_freq
    upslope_noise_allowance_num_samples = upslope_noise_allowance_duration * sample_freq

    # identify areas where waveform increases sample after sample for a minimum stretch, default to min in search area if no areas found
    upslope_indices = [np.where(np.diff(segment) > 0)[0] for segment in valley_segments]
    upslope_indices = [
        [
            i
            for i in np.split(
                upslope, np.where(np.diff(upslope) > (1 + upslope_noise_allowance_num_samples))[0] + 1
            )
            if len(i) >= upslope_num_samples
        ]
        for upslope in upslope_indices
    ]

    valley_indices = []
    for upslope, valley in zip(upslope_indices, valley_segments):
        # if no qualifying upslope is identified then use the min value in the segment
        if len(upslope) == 0:
            valley_index = np.argmin(valley)
            # print("min")

        # if only one upslope is identified the use the first value in the upslope
        elif len(upslope) == 1:
            valley_index = upslope[0][0]
            # print('upslope-single')

        # if multiple qualifying upslopes are found use the longest identified upslope.
        # if multiple equal length slopes are identified the latest upslope is used
        else:
            longest_upslope = max([len(length) for length in upslope])
            valley_index = [slope[0] for slope in upslope if len(slope) == longest_upslope][0]
            # print('upslope')

        valley_indices.append(valley_index)

    valleys = np.array(
        [peak - (window - index) for peak, index, window in zip(peaks, valley_indices, search_windows)]
    )

    # indices are only valid with the given window, so adjust to match original signal
    peaks += window_indices[0]
    valleys += window_indices[0]

    return peaks, valleys
