# -*- coding: utf-8 -*-
"""Transforming arrays of Mantarray data throughout the analysis pipeline."""
from typing import Any
from typing import Dict
from typing import List
from typing import Union
import uuid

from nptyping import NDArray
import numpy as np
from scipy import signal

from .constants import ADC_GAIN
from .constants import BESSEL_BANDPASS_UUID
from .constants import BESSEL_LOWPASS_10_UUID
from .constants import BESSEL_LOWPASS_30_UUID
from .constants import BUTTERWORTH_LOWPASS_30_UUID
from .constants import GAUSS_PER_MILLITESLA
from .constants import MEMSIC_CENTER_OFFSET
from .constants import MEMSIC_FULL_SCALE
from .constants import MEMSIC_MSB
from .constants import MICRO_TO_BASE_CONVERSION
from .constants import MILLI_TO_BASE_CONVERSION
from .constants import MILLIMETERS_PER_MILLITESLA
from .constants import MILLIVOLTS_PER_MILLITESLA
from .constants import NEWTONS_PER_MILLIMETER
from .constants import RAW_TO_SIGNED_CONVERSION_VALUE
from .constants import REFERENCE_VOLTAGE
from .exceptions import FilterCreationNotImplementedError
from .exceptions import UnrecognizedFilterUuidError


FILTER_CHARACTERISTICS: Dict[uuid.UUID, Dict[str, Union[str, float, int]]] = {
    BESSEL_BANDPASS_UUID: {
        "filter_type": "bessel",
        "order": 4,
        "high_pass_hz": 0.1,
        "low_pass_hz": 10,
    },
    BESSEL_LOWPASS_10_UUID: {"filter_type": "bessel", "order": 4, "low_pass_hz": 10},
    BESSEL_LOWPASS_30_UUID: {"filter_type": "bessel", "order": 4, "low_pass_hz": 30},
    BUTTERWORTH_LOWPASS_30_UUID: {
        "filter_type": "butterworth",
        "order": 4,
        "low_pass_hz": 30,
    },
}


def create_filter(
    filter_uuid: uuid.UUID,
    sample_period_microseconds: int,
) -> NDArray[(Any, Any), float]:
    """Create a filter to apply to data streams.

    Args:
        filter_uuid: a UUID of an already accepted and approved filter
        sample_period_microseconds: the sampling period for the data stream you want to apply the filter to

    Returns:
        The scipy 'b' and 'a' vectors to use in scipy.signal.filtfilt
    """
    sampling_frequency_hz = 1 / (sample_period_microseconds / MICRO_TO_BASE_CONVERSION)
    nyquist_frequency_limit = sampling_frequency_hz / 2

    if filter_uuid not in FILTER_CHARACTERISTICS:
        raise UnrecognizedFilterUuidError(filter_uuid)

    the_filter_characteristics = FILTER_CHARACTERISTICS[filter_uuid]
    normalized_high_pass_frequency: Union[int, float]
    normalized_low_pass_frequency: Union[int, float]
    filter_order: int
    if "order" in the_filter_characteristics:
        if not isinstance(the_filter_characteristics["order"], int):
            raise NotImplementedError("The filter order must always be an int.")

        filter_order = the_filter_characteristics["order"]
    pass_boundaries: List[Union[float, int]] = list()
    bandpass_type = "lowpass"  # by default
    if "high_pass_hz" in the_filter_characteristics:
        if isinstance(the_filter_characteristics["high_pass_hz"], str):
            raise NotImplementedError("The high pass frequency should never be a string.")
        normalized_high_pass_frequency = the_filter_characteristics["high_pass_hz"] / nyquist_frequency_limit
        pass_boundaries.append(normalized_high_pass_frequency)
    if "low_pass_hz" in the_filter_characteristics:
        if isinstance(the_filter_characteristics["low_pass_hz"], str):
            raise NotImplementedError("The low pass frequency should never be a string.")
        normalized_low_pass_frequency = the_filter_characteristics["low_pass_hz"] / nyquist_frequency_limit
        pass_boundaries.append(normalized_low_pass_frequency)

    if len(pass_boundaries) == 2:
        bandpass_type = "bandpass"
    filter_type = the_filter_characteristics["filter_type"]
    if filter_type == "bessel":
        sos_polys = signal.bessel(filter_order, pass_boundaries, btype=bandpass_type, output="sos")
    elif filter_type == "butterworth":
        sos_polys = signal.butter(filter_order, pass_boundaries, btype=bandpass_type, output="sos")
    else:
        raise FilterCreationNotImplementedError(filter_uuid)

    if not isinstance(sos_polys, NDArray[float]):
        raise NotImplementedError("Returned polynomials most be float arrays")
    return sos_polys


def apply_sensitivity_calibration(raw_gmr_reading: NDArray[(2, Any), int]) -> NDArray[(2, Any), int]:
    """Apply the result of a sensor sensitivity calibration.

    Actual sensitivity calibration will be performed once information obtained from Jason.

    Args:
        raw_gmr_reading: an original 2d array of time vs GMR readings. Could be from tissue construct or reference sensor.

    Returns:
        A 2d array of the time vs GMR readings after sensitivity calibration. Data will be rounded to integers if calibration results in slight decimal behavior.
    """
    return raw_gmr_reading


def noise_cancellation(
    tissue_gmr_reading: NDArray[(2, Any), int],
    reference_gmr_reading: NDArray[
        (2, Any), int
    ],  # pylint: disable=unused-argument # this will eventually be used
) -> NDArray[(2, Any), int]:
    """Perform cancellation of ambient magnetic noise using reference sensor.

    This should be performed after sensitivity calibration has been applied to the raw data from each sensor.

    Args:
        tissue_gmr_reading: from the tissue construct sensor
        reference_gmr_reading: from the reference sensor

    Returns:
        A single 2D array of time vs GMR reading.
    """
    return tissue_gmr_reading


def apply_empty_plate_calibration(
    noise_cancelled_gmr: NDArray[(2, Any), int],
) -> NDArray[(2, Any), int]:
    """Apply the result of an empty plate calibration.

    Actual empty plate calibration will be performed once information obtained from Jason.

    Args:
        noise_cancelled_gmr: an 2D array of Time and GMR readings after combining reference and tissue sensor readings.

    Returns:
        A 2D array of the Time and GMR readings after empty plate calibration. Data will be rounded to integers if calibration results in slight decimal behavior.
    """
    return noise_cancelled_gmr


def apply_noise_filtering(
    fully_calibrated_gmr: NDArray[(2, Any), int],
    scipy_filter_sos_coefficients: NDArray[(Any, Any), float],
) -> NDArray[(2, Any), int]:
    """Apply the result of an empty plate calibration.

    Actual empty plate calibration will be performed once information obtained from Jason.

    Args:
        fully_calibrated_gmr: an 2D array of Time and GMR readings after applying the Empty Plate calibration.
        scipy_filter_sos_coefficients: The 'second order system' coefficient array that scipy filters generate when created

    Returns:
        A 2D array of the Time and GMR readings after empty plate calibration. Data will be rounded to integers if calibration results in slight decimal behavior.
    """
    time_readings = fully_calibrated_gmr[0, :]
    gmr_readings = fully_calibrated_gmr[1, :]

    float_array = signal.sosfiltfilt(scipy_filter_sos_coefficients, gmr_readings)
    int_array = np.rint(float_array).astype(np.int32)

    filtered_data: NDArray[(2, Any), int] = np.vstack((time_readings, int_array))
    return filtered_data


def calculate_voltage_from_gmr(
    gmr_data: NDArray[(2, Any), int],
    reference_voltage: Union[float, int] = REFERENCE_VOLTAGE,
    adc_gain: int = ADC_GAIN,
) -> NDArray[(2, Any), np.float64]:
    """Convert 'signed' 24-bit values from an ADC to measured voltage.

    Conversion values were obtained 03/09/2021 by Kevin Grey

    Args:
        gmr_data: time and GMR numpy array. Typically coming from filtered_gmr_data
        reference_voltage: Almost always leave as default of 2.5V
        adc_gain: Current implementation of Mantarray is constant value of 2, but may change in the future

    Returns:
        A 2D array of time vs Voltage
    """
    millivolts_per_lsb = 1000 * reference_voltage / RAW_TO_SIGNED_CONVERSION_VALUE
    sample_in_millivolts = gmr_data[1, :].astype(np.float64) * millivolts_per_lsb * (1 / adc_gain)
    sample_in_volts = sample_in_millivolts / MILLI_TO_BASE_CONVERSION
    return np.vstack((gmr_data[0, :].astype(np.float64), sample_in_volts))


def calculate_displacement_from_voltage(
    voltage_data: NDArray[(2, Any), np.float64],
) -> NDArray[(2, Any), np.float64]:
    """Convert voltage to displacement.

    Conversion values were obtained 03/09/2021 by Kevin Grey

    Args:
        voltage_data: time and Voltage numpy array. Typically coming from calculate_voltage_from_gmr

    Returns:
        A 2D array of time vs Displacement (meters)
    """
    sample_in_millivolts = voltage_data[1, :] * MILLI_TO_BASE_CONVERSION
    time = voltage_data[0, :]

    # calculate magnetic flux density
    sample_in_milliteslas = sample_in_millivolts / MILLIVOLTS_PER_MILLITESLA

    # calculate displacement
    sample_in_millimeters = sample_in_milliteslas * MILLIMETERS_PER_MILLITESLA
    sample_in_meters = sample_in_millimeters / MILLI_TO_BASE_CONVERSION

    return np.vstack((time, sample_in_meters)).astype(np.float64)


def calculate_force_from_displacement(
    displacement_data: NDArray[(2, Any), np.float64],
) -> NDArray[(2, Any), np.float64]:
    """Convert displacement to force.

    Conversion values were obtained 03/09/2021 by Kevin Grey

    Args:
        displacement_data: time and Displacement numpy array. Typically coming from calculate_displacement_from_voltage

    Returns:
        A 2D array of time vs Force (Newtons)
    """
    sample_in_millimeters = displacement_data[1, :] * MILLI_TO_BASE_CONVERSION
    time = displacement_data[0, :]

    # calculate force
    sample_in_newtons = sample_in_millimeters * NEWTONS_PER_MILLIMETER

    return np.vstack((time, sample_in_newtons)).astype(np.float64)


def calculate_magnetic_flux_density_from_memsic(  # pylint: disable=invalid-name  # Tanner (11/22/21): Can't think of a shorter name that is better
    memsic_data: NDArray[(1, Any), int],
) -> NDArray[(1, Any), np.float64]:
    """Convert raw data from memsic sensor into magnetic flux density.

    Conversion values are valid as of 11/19/2021

    Args:
        memsic_data: A 1D array of raw memsic signal

    Returns:
        A 1D array of magnetic flux density
    """
    samples_in_milliteslas = (
        (memsic_data.astype(np.int64) - MEMSIC_CENTER_OFFSET)
        * MEMSIC_FULL_SCALE
        / MEMSIC_MSB
        / GAUSS_PER_MILLITESLA
    )
    return samples_in_milliteslas.astype(np.float64)
