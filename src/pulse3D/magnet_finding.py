# -*- coding: utf-8 -*-
"""More accurate estimation of magnet positions."""
from typing import Any
from typing import Dict
from typing import List
from typing import TYPE_CHECKING
from typing import Union

from mantarray_magnet_finding.magnet_finding import get_positions
from nptyping import NDArray
import numpy as np
import scipy.signal as signal

from .constants import NUM_CHANNELS_24_WELL_PLATE
from .constants import NUM_CHANNELS_PER_WELL
from .constants import TISSUE_SENSOR_READINGS


if TYPE_CHECKING:
    from .plate_recording import WellFile


def find_magnet_positions(
    fields: NDArray[(NUM_CHANNELS_24_WELL_PLATE, Any), float],
    baseline: NDArray[(NUM_CHANNELS_24_WELL_PLATE, Any), float],
    initial_magnet_finding_params: Dict[str, Union[int, float]],
    filter_outputs: bool = True,
) -> Dict[str, NDArray[(1, Any), float]]:
    output_dict = get_positions((fields.T - baseline).T, **initial_magnet_finding_params)  # type: ignore  # mypy complaining about **
    if filter_outputs:
        for param, output_arr in output_dict.items():
            output_dict[param] = filter_magnet_positions(output_arr)
    return output_dict


def filter_magnet_positions(magnet_positions: NDArray[(Any, 24), float]) -> NDArray[(Any, 24), float]:
    high_cut_hz = 30
    b, a = signal.butter(4, high_cut_hz, "low", fs=100)
    filtered_magnet_positions = np.empty(magnet_positions.shape)
    # Tanner (1/7/22): need to filter each well individually, can't filter over the entire axis of the array at once
    for well_arr_idx in range(magnet_positions.shape[1]):
        filtered_magnet_positions[:, well_arr_idx] = signal.filtfilt(b, a, magnet_positions[:, well_arr_idx])
    return filtered_magnet_positions


def format_well_file_data(well_files: List["WellFile"]) -> NDArray[(NUM_CHANNELS_24_WELL_PLATE, Any), float]:
    """Convert well data to input array format of magnet finding alg."""
    plate_data_array = None
    for well_idx, well_file in enumerate(well_files):
        tissue_data = well_file[TISSUE_SENSOR_READINGS][:]
        if plate_data_array is None:
            num_samples = tissue_data.shape[-1]
            plate_data_array = np.empty((NUM_CHANNELS_24_WELL_PLATE, num_samples))
        reshaped_data = tissue_data.reshape((NUM_CHANNELS_PER_WELL, num_samples))
        plate_data_array[
            well_idx * NUM_CHANNELS_PER_WELL : (well_idx + 1) * NUM_CHANNELS_PER_WELL, :
        ] = reshaped_data
    return plate_data_array


def fix_dropped_samples(raw_signal: NDArray[Any, np.uint16]) -> NDArray[Any, np.uint16]:
    # Tanner (2/7/22): may want to add additional conditions if this has issues
    fixed_signal = raw_signal.copy()
    dropped_sample_indices = [tuple(indices) for indices in np.argwhere(raw_signal == 0)]
    for index_tuple in dropped_sample_indices:
        innermost_arr = fixed_signal[index_tuple[:-1]]
        sample_idx = index_tuple[-1]
        if sample_idx == 0:
            innermost_arr[sample_idx] = innermost_arr[sample_idx + 1]
        elif sample_idx == len(innermost_arr) - 1:
            innermost_arr[sample_idx] = innermost_arr[sample_idx - 1]
        else:
            innermost_arr[sample_idx] = np.mean(
                [innermost_arr[sample_idx - 1], innermost_arr[sample_idx + 1]]
            )
    return fixed_signal
