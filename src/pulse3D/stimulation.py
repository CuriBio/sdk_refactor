# -*- coding: utf-8 -*-
import itertools
import math
from operator import ge
from operator import gt
from operator import le
from operator import lt
from typing import Any
from typing import Dict
from typing import List

from nptyping import NDArray
import numpy as np

from .constants import STIM_COMPLETE_SUBPROTOCOL_IDX
from .exceptions import SubprotocolFormatIncompatibleWithInterpolationError


def truncate_interpolated_subprotocol_waveform(
    waveform: NDArray[(2, Any), int], cutoff_timepoint: int, from_start: bool
) -> NDArray[(2, Any), int]:
    if not waveform.shape[-1]:
        return waveform

    # flip before truncating if needed so always truncating from the end
    if from_start:
        waveform = waveform[:, ::-1]

    inclusive_ineq = ge if from_start else le
    strict_ineq = gt if from_start else lt

    if inclusive_ineq(cutoff_timepoint, waveform[0, 0]):
        return np.empty((2, 0))

    if strict_ineq(cutoff_timepoint, waveform[0, -1]):
        # truncate data points after the cutoff
        truncated_waveform = waveform[:, strict_ineq(waveform[0], cutoff_timepoint)]
        # extend the waveform at the final amplitude to the cutoff point
        cutoff_idx = truncated_waveform.shape[1]
        final_pair = [[cutoff_timepoint], [waveform[1, cutoff_idx]]]
        waveform = np.concatenate([truncated_waveform, final_pair], axis=1)

    # flip back if needed
    if from_start:
        waveform = waveform[:, ::-1]

    return waveform


def remove_intermediate_interpolation_data(
    stim_waveform: NDArray[(2, Any), int], timepoint: int
) -> NDArray[(2, Any), int]:
    if not stim_waveform.shape[-1]:
        return stim_waveform

    idxs = np.nonzero(stim_waveform[0] == timepoint)[0]
    return np.delete(stim_waveform, idxs[1:-1], axis=1)


def create_interpolated_subprotocol_waveform(
    subprotocol: Dict[str, int], start_timepoint: int, stop_timepoint: int, include_start_timepoint: bool
) -> NDArray[(2, Any), int]:
    try:
        subprotocol_type = subprotocol["type"]
    except KeyError:
        raise SubprotocolFormatIncompatibleWithInterpolationError("Must have a 'type'")

    if subprotocol_type == "loop":
        raise ValueError("Cannot interpolate loops")

    if subprotocol_type == "delay":
        # duration ignored here since stop timepoint is when the next subprotocol starts
        interpolated_waveform_arr = np.array([[start_timepoint, stop_timepoint], [0, 0]], dtype=int)
    else:
        # postphase_amplitude and interphase_amplitude will never be present in the subprotocol dict, so 0 will be returned for them below
        time_components = ["phase_one_duration", "postphase_interval"]
        amplitude_components = ["phase_one_charge", "postphase_amplitude"]
        if subprotocol_type == "biphasic":
            time_components[1:1] = ["phase_two_duration"]
            amplitude_components[1:1] = ["phase_two_charge"]
            # only add interphase interval if it is non-zero
            if subprotocol["interphase_interval"] > 0:
                time_components[1:1] = ["interphase_interval"]
                amplitude_components[1:1] = ["interphase_amplitude"]

        # create first cycle except for initial pair which may be added later,
        # and don't duplicate the pair at the end of the postphase interval
        first_cycle_timepoints = np.repeat(
            list(
                itertools.accumulate([subprotocol[comp] for comp in time_components], initial=start_timepoint)
            ),
            2,
        )[1:-1]
        first_cycle_amplitudes = np.repeat([subprotocol.get(comp, 0) for comp in amplitude_components], 2)
        cycle_dur = first_cycle_timepoints[-1] - start_timepoint
        # determine how many cycles this subprotocols actually ran for
        actual_num_cycles = math.ceil((stop_timepoint - start_timepoint) / cycle_dur)
        # add repeated cycle with incremented timepoints
        all_cycles_timepoints = [
            t + (cycle_num * cycle_dur)
            for cycle_num in range(actual_num_cycles)
            for t in first_cycle_timepoints
        ]
        all_cycles_amplitudes = list(first_cycle_amplitudes) * actual_num_cycles
        # add initial pair if needed
        if include_start_timepoint:
            all_cycles_timepoints = [start_timepoint] + all_cycles_timepoints
            all_cycles_amplitudes = [0] + all_cycles_amplitudes
        # convert to array
        interpolated_waveform_arr = np.array([all_cycles_timepoints, all_cycles_amplitudes], dtype=int)

    # truncate end of waveform at the stop timepoint
    interpolated_waveform_arr = truncate_interpolated_subprotocol_waveform(
        interpolated_waveform_arr, stop_timepoint, from_start=False
    )

    return interpolated_waveform_arr


def interpolate_stim_session(
    subprotocols: List[Dict[str, int]],
    stim_status_updates: NDArray[(2, Any), int],
    session_start_timepoint: int,
    session_stop_timepoint: int,
) -> NDArray[(2, Any), int]:
    # if protocol starts after the session completes, return empty array
    if session_stop_timepoint <= stim_status_updates[0, 0]:
        return np.empty((2, 0))

    if stim_status_updates[1, -1] == STIM_COMPLETE_SUBPROTOCOL_IDX:
        # if protocol completes before the session starts, return empty array
        if session_start_timepoint >= stim_status_updates[0, -1]:
            return np.empty((2, 0))
        # 'protocol complete' is the final status update, which doesn't need a waveform created for it
        stim_status_updates = stim_status_updates[:, :-1]

    session_waveform = np.empty((2, 0))
    for next_status_idx, (start_timepoint, subprotocol_idx) in enumerate(stim_status_updates.T, 1):
        is_final_status_update = next_status_idx == stim_status_updates.shape[-1]

        stop_timepoint = (
            session_stop_timepoint if is_final_status_update else stim_status_updates[0, next_status_idx]
        )

        # start point only needs to be included for the very first subprotocol
        include_start_timepoint = next_status_idx == 1

        subprotocol_waveform = create_interpolated_subprotocol_waveform(
            subprotocols[subprotocol_idx], start_timepoint, stop_timepoint, include_start_timepoint
        )

        if not subprotocol_waveform.shape[-1]:
            continue

        session_waveform = np.concatenate([session_waveform, subprotocol_waveform], axis=1)

        # remove unnecessary data points, if any
        session_waveform = remove_intermediate_interpolation_data(session_waveform, start_timepoint)

    # truncate beginning of waveform at the initial timepoint
    session_waveform = truncate_interpolated_subprotocol_waveform(
        session_waveform, session_start_timepoint, from_start=True
    )

    return session_waveform


def create_stim_session_waveforms(
    subprotocols: List[Dict[str, int]],
    stim_status_updates: NDArray[(2, Any), int],
    initial_timepoint: int,
    final_timepoint: int,
):
    stim_sessions = [
        session
        for session in np.split(
            stim_status_updates,
            np.where(stim_status_updates[1] == STIM_COMPLETE_SUBPROTOCOL_IDX)[0] + 1,
            axis=1,
        )
        if session.shape[-1]
    ]

    stop_timepoints_of_each_session = [session[0, -1] for session in stim_sessions[:-1]] + [
        stim_sessions[-1][0, -1]
        if stim_sessions[-1][1, -1] == STIM_COMPLETE_SUBPROTOCOL_IDX
        else final_timepoint
    ]

    interpolated_stim_sessions = [
        interpolate_stim_session(subprotocols, session_updates, initial_timepoint, session_stop_timepoint)
        for session_updates, session_stop_timepoint in zip(stim_sessions, stop_timepoints_of_each_session)
    ]
    return interpolated_stim_sessions


def aggregate_timepoints(timepoints_from_wells: List[NDArray[(1, Any), int]]) -> NDArray[(1, Any), int]:
    unique_timepoints = set(t for timepoints in timepoints_from_wells for t in timepoints)
    return np.array(sorted(unique_timepoints), dtype=int)


def realign_interpolated_stim_data(
    new_timepoints: NDArray[(1, Any), int], orignal_stim_status_data: NDArray[(2, Any), int]
) -> NDArray[(1, Any), int]:
    adjusted_interpolated_stim_data = np.full((len(new_timepoints)), np.NaN)
    orignal_timepoints_list = orignal_stim_status_data[0].tolist()
    for new_idx, new_t in enumerate(new_timepoints):
        try:
            old_idx = orignal_timepoints_list.index(new_t)
        except ValueError:
            continue
        else:
            orignal_timepoints_list[old_idx] = None  # remove since there will be duplicate timepoints
            adjusted_interpolated_stim_data[new_idx] = orignal_stim_status_data[1, old_idx]
    return adjusted_interpolated_stim_data
