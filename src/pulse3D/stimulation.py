# -*- coding: utf-8 -*-
import itertools
from typing import Any
from typing import Dict

from nptyping import NDArray
import numpy as np


def truncate_interpolated_subprotocol_waveform(
    full_waveform: NDArray[(2, Any), int], cutoff_timepoint: int
) -> NDArray[(2, Any), int]:
    if cutoff_timepoint >= full_waveform[0, -1]:
        return full_waveform

    truncated_waveform = full_waveform[:, full_waveform[0] < cutoff_timepoint]

    cutoff_idx = truncated_waveform.shape[1]
    earliest_two_discarded_pairs = full_waveform[:, cutoff_idx : cutoff_idx + 2]

    # if truncated in the middle of a phase, then need to add the completion of the phase back
    if earliest_two_discarded_pairs[1, 0]:
        earliest_two_discarded_pairs[0, :] = [cutoff_timepoint] * 2
        truncated_waveform = np.concatenate([truncated_waveform, earliest_two_discarded_pairs], axis=1)

    return truncated_waveform


def create_interpolated_subprotocol_waveform(
    subprotocol: Dict[str, int], start_timepoint: int, stop_timepoint: int
) -> NDArray[(2, Any), int]:
    try:
        subprotocol_type = subprotocol["type"]
    except KeyError:
        raise ValueError("This format of subprotocol is not supported, must have a 'type'")

    if subprotocol_type == "loop":
        raise ValueError("Cannot interpolate loops")

    if subprotocol_type == "delay":
        # don't need to take stop_timepoint into account here
        return np.array([[start_timepoint], [0]], dtype=int)

    time_components = ["phase_one_duration"]
    amplitude_components = ["phase_one_charge"]
    if subprotocol_type == "biphasic":
        time_components.extend(["interphase_interval", "phase_two_duration"])
        # interphase_amplitude will never be present in the subprotocol dict, so 0 will be returned for it below
        amplitude_components.extend(["interphase_amplitude", "phase_two_charge"])

    # create first cycle
    first_cycle_timepoints = np.repeat(
        list(itertools.accumulate([subprotocol[comp] for comp in time_components], initial=start_timepoint)),
        2,
    )
    first_cycle_amplitudes = [
        0,
        *np.repeat([subprotocol.get(comp, 0) for comp in amplitude_components], 2),
        0,
    ]
    cycle_dur = first_cycle_timepoints[-1] + subprotocol["postphase_interval"] - start_timepoint
    # repeat cycle with incremented timepoints
    all_cycles_timepoints = [
        t + (cycle_num * cycle_dur)
        for cycle_num in range(subprotocol["num_cycles"])
        for t in first_cycle_timepoints
    ]
    all_cycles_amplitudes = first_cycle_amplitudes * subprotocol["num_cycles"]
    interpolated_waveform_arr = np.array([all_cycles_timepoints, all_cycles_amplitudes], dtype=int)

    # truncate waveform at the stop_timepoint
    interpolated_waveform_arr = truncate_interpolated_subprotocol_waveform(
        interpolated_waveform_arr, stop_timepoint
    )

    return interpolated_waveform_arr
