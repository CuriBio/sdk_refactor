# -*- coding: utf-8 -*-
"""Detecting peak and valleys of incoming Mantarray data."""

from typing import Optional

import numpy as np
from scipy import signal
from scipy.optimize import curve_fit

from .constants import DEFAULT_NB_HEIGHT_FACTOR
from .constants import DEFAULT_NB_NOISE_PROMINENCE_FACTOR
from .constants import DEFAULT_NB_RELATIVE_PROMINENCE_FACTOR
from .constants import DEFAULT_NB_WIDTH_FACTORS


def quadratic(x, a, b, c):
    return a * (x**2) + b * x + c


# TODO ? prefix args with peak_ or valley_ so it's more clear which they affect
def noise_based_peak_finding(
    tissue_data,
    noise_prominence_factor=DEFAULT_NB_NOISE_PROMINENCE_FACTOR,
    relative_prominence_factor: Optional[float] = DEFAULT_NB_RELATIVE_PROMINENCE_FACTOR,
    width_factors=DEFAULT_NB_WIDTH_FACTORS,
    height_factor=DEFAULT_NB_HEIGHT_FACTOR,
    max_frequency: Optional[float] = None,
    valley_search_size=1,
    upslope_length=7,
    upslope_noise_allowance=1,
):
    """

    Parameters
    ----------
    TODO clean this up
    tissue_data : pd.Series/np.array
        Time axis of the given waveform, units are relevant to valley_search_size and so should be known by the user
        The waveform to be analysed, this should be a tuple, list, pd.Series, or np.array containing Y values for the waveform
    prominence_factor : float
        The default is 2.5. Adjusting this changes the prominence of a given peak required to be considered. This value is used as a factor to multiply the peak to peak noise estimate and so can be
        considered as the minimum SNR of a peak required to be detected
    width_factor : tuple
        The default is (0,5). The minimum and maximum width of a peak required to be considered. This should be given in the units of the time axis
    height_factor : int
        The default is 0. The minimum height of a peak required to be considered. This should be given in the units of the waveform Y axis
    max_frequency: int
        The default is 100. This value defines how often a peak in the waveform is found in the units of the time axis. eg. A max frequency of 1 finds only 1 peak every unit of the time axis
        Note if this is set higher than the sample frequency then it is rest to the sample frequency to find a max of one peak per sample ie every point can be a peak

    relative_prominence_factor : float
        If specified, also take the prominence of a peak relative to the tallest peak into consideration. If this falls below the noise-based prominence threshold, that will be used instead.

    valley_search_size : float
        The default is 1. TThe search distance before a peak used to find a valley (given in the units of the time axis).
        If this window includes a previous peak then for that peak the window will automatically be shortened
    upslope_length : int
        The default is 7. The number of samples through which the waveform values must continuously rise in order to be considered an upslope
    upslope_noise_allowance : float
        The default is 1. This is the number of points in the upslope which are not increases which can be tolerated within a single upslope

    Returns
    -------
    peaks : np.array
        1-D array of peak indicies.
    valleys: np.array
        1-D array of valley indicies.

    """
    time_axis, waveform = tissue_data

    # TODO split into subfunctions, one for finding peaks and the other for finding valleys

    # set estimate of peak to peak noise amplitude is 10uN for average recording
    default_noise = 10
    default_prom = 5

    # extract sample frequency from time_axis (assumes sampling freq is constant)
    sample_freq = 1 / (time_axis[1] - time_axis[0])

    if max_frequency:
        # if max freq is greater than the sampling freq, use sampling freq instead
        max_frequency = min(max_frequency, sample_freq)
    else:
        # no max freq given, use sampling freq
        max_frequency = sample_freq

    # find peaks with this estimated amplitude
    peaks, _ = signal.find_peaks(waveform, prominence=default_prom * default_noise)

    # if first attempt finds no peaks as they are too small, retry with smaller prominence
    # this approach should return a list of peak indices even if no true peaks exist as it will terminate at a prominence of 1.
    correction_factor = 1
    while len(peaks) == 0 and correction_factor <= default_prom:
        peaks, _ = signal.find_peaks(waveform, prominence=(default_prom - correction_factor) * default_noise)

        correction_factor += 1

    # use peaks to extract waveform segments from which noise data can be extracted - control over this could be given to the user if required
    segment_size = 10

    while peaks[-1] + segment_size > len(
        waveform
    ):  # possible bug deletes only peak and you get a len(list) == 0
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

    if len(peaks) == 0:
        return np.array([]), np.array([])

    # TODO what is this doing?
    segment_size = int(valley_search_size * sample_freq)
    while len(peaks) != 0 and peaks[0] - segment_size < 0:
        peaks = np.delete(peaks, 0)

    if len(peaks) == 0:
        return np.array([]), np.array([])

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

    # identify areas where waveform increases sample after sample for a minimum stretch, default to min in search area if no areas found
    upslope_indices = [np.where(np.diff(segment) > 0)[0] for segment in valley_segments]
    upslope_indices = [
        [
            i
            for i in np.split(upslope, np.where(np.diff(upslope) > (1 + upslope_noise_allowance))[0] + 1)
            if len(i) >= upslope_length
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

    return peaks, valleys
