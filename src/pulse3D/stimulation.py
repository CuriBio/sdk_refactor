# -*- coding: utf-8 -*-
import itertools
from operator import ge
from operator import gt
from operator import le
from operator import lt
from typing import Any
from typing import Dict
from typing import List

from nptyping import NDArray
import numpy as np


def truncate_interpolated_subprotocol_waveform(
    waveform: NDArray[(2, Any), int], cutoff_timepoint: int, from_start: bool
) -> NDArray[(2, Any), int]:
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
        interpolated_waveform_arr = np.array(
            [[start_timepoint, start_timepoint + subprotocol["duration"]], [0, 0]], dtype=int
        )
    else:
        # postphase_amplitude and interphase_amplitude will never be present in the subprotocol dict, so 0 will be returned for them below
        time_components = ["phase_one_duration", "postphase_interval"]
        amplitude_components = ["phase_one_charge", "postphase_amplitude"]
        if subprotocol_type == "biphasic":
            time_components[1:1] = ["interphase_interval", "phase_two_duration"]
            amplitude_components[1:1] = ["interphase_amplitude", "phase_two_charge"]

        # create first cycle except for initial pair which will be added later,
        # and don't duplicate the pair at the end of the postphase interval
        first_cycle_timepoints = np.repeat(
            list(
                itertools.accumulate([subprotocol[comp] for comp in time_components], initial=start_timepoint)
            ),
            2,
        )[1:-1]
        first_cycle_amplitudes = np.repeat([subprotocol.get(comp, 0) for comp in amplitude_components], 2)
        cycle_dur = first_cycle_timepoints[-1] - start_timepoint
        # add repeated cycle with incremented timepoints to initial pair
        all_cycles_timepoints = [start_timepoint] + [
            t + (cycle_num * cycle_dur)
            for cycle_num in range(subprotocol["num_cycles"])
            for t in first_cycle_timepoints
        ]
        all_cycles_amplitudes = [0] + list(first_cycle_amplitudes) * subprotocol["num_cycles"]
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

    protocol_complete_idx = 255  # TODO make a constant for this

    if stim_status_updates[1, -1] == protocol_complete_idx:
        # if protocol completes before the session starts, return empty array
        if session_start_timepoint >= stim_status_updates[0, -1]:
            return np.empty((2, 0))
        # 'protocol complete' is the final status update, which doesn't need a waveform created for it
        stim_status_updates = stim_status_updates[:, :-1]

    subprotocol_waveforms = []
    for next_status_idx, (start_timepoint, subprotocol_idx) in enumerate(stim_status_updates.T, 1):
        is_final_status_update = next_status_idx == stim_status_updates.shape[-1]
        stop_timepoint = (
            session_stop_timepoint if is_final_status_update else stim_status_updates[0, next_status_idx]
        )

        subprotocol_waveform = create_interpolated_subprotocol_waveform(
            subprotocols[subprotocol_idx], start_timepoint, stop_timepoint
        )
        subprotocol_waveforms.append(subprotocol_waveform)

    session_waveform = np.concatenate(subprotocol_waveforms, axis=1)

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
        for session in np.split(stim_status_updates, np.where(stim_status_updates[1] == 255)[0] + 1, axis=1)
        if session.shape[-1]
    ]

    stop_timepoints_of_each_session = [session[0, -1] for session in stim_sessions[:-1]] + [
        stim_sessions[-1][0, -1] if stim_sessions[-1][1, -1] == 255 else final_timepoint
    ]

    interpolated_stim_sessions = [
        interpolate_stim_session(subprotocols, session_updates, initial_timepoint, session_stop_timepoint)
        for session_updates, session_stop_timepoint in zip(stim_sessions, stop_timepoints_of_each_session)
    ]
    return interpolated_stim_sessions
