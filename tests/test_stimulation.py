# -*- coding: utf-8 -*-
import itertools
from random import randint

import numpy as np
from pulse3D import stimulation
from pulse3D.stimulation import create_interpolated_subprotocol_waveform
from pulse3D.stimulation import truncate_interpolated_subprotocol_waveform
import pytest


@pytest.mark.parametrize(
    "test_cutoff_timepoint,expected_truncated_arr",
    [
        (-1, np.array([[], []])),
        (0, np.array([[], []])),
        (1, np.array([[0, 0, 1, 1], [0, 2, 2, 0]])),
        (5, np.array([[0, 0, 5, 5], [0, 2, 2, 0]])),
        (6, np.array([[0, 0, 5, 5], [0, 2, 2, 0]])),
        (10, np.array([[0, 0, 5, 5], [0, 2, 2, 0]])),
        (11, np.array([[0, 0, 5, 5, 10, 10, 11, 11], [0, 2, 2, 0, 0, -2, -2, 0]])),
        (15, np.array([[0, 0, 5, 5, 10, 10, 15, 15], [0, 2, 2, 0, 0, -2, -2, 0]])),
        (16, np.array([[0, 0, 5, 5, 10, 10, 15, 15], [0, 2, 2, 0, 0, -2, -2, 0]])),
    ],
)
def test_truncate_interpolated_subprotocol_waveform__returns_correctly_truncated_array(
    test_cutoff_timepoint, expected_truncated_arr
):
    # using a biphasic array since it contains all array components present in monophasic and delay
    test_biphasic_arr = np.array(
        [
            [0, 0, 5, 5, 10, 10, 15, 15],
            [0, 2, 2, 0, 0, -2, -2, 0],
        ]
    )

    np.testing.assert_array_equal(
        truncate_interpolated_subprotocol_waveform(test_biphasic_arr, test_cutoff_timepoint),
        expected_truncated_arr,
    )


def test_create_interpolated_subprotocol_waveform__raises_error_if_subprotocol_is_not_of_a_supported_format():
    with pytest.raises(ValueError, match="This format of subprotocol is not supported, must have a 'type'"):
        create_interpolated_subprotocol_waveform(
            {
                "phase_one_duration": 1,
                "phase_one_charge": 2,
                "interphase_interval": 3,
                "phase_two_duration": 4,
                "phase_two_charge": 5,
                "repeat_delay_interval": 6,
                "total_active_duration": 1000,
            },
            # start and stop timepoint don't matter here
            randint(0, 100),
            randint(100, 200),
        )


def test_create_interpolated_subprotocol_waveform__raises_error_if_loop_given():
    with pytest.raises(ValueError, match="Cannot interpolate loops"):
        create_interpolated_subprotocol_waveform(
            {"type": "loop"},
            # start and stop timepoint don't matter here
            randint(0, 100),
            randint(100, 200),
        )


def test_create_interpolated_subprotocol_waveform__creates_delay_correctly():
    # the delay duration and stop timepoint are ignored
    test_delay = {"type": "delay", "duration": randint(0, 100)}

    test_start_timepoint = randint(0, 100)

    actual_delay_arr = create_interpolated_subprotocol_waveform(
        test_delay, test_start_timepoint, randint(0, 100)
    )
    np.testing.assert_array_equal(actual_delay_arr, [[test_start_timepoint], [0]])


def test_create_interpolated_subprotocol_waveform___creates_monophasic_waveform_correctly(mocker):
    mocked_truncate = mocker.patch.object(
        stimulation, "truncate_interpolated_subprotocol_waveform", autospec=True
    )

    # make sure keys are inserted out of order so the insertion order can't be relied on
    test_biphasic_pulse = {"type": "monophasic"}
    test_biphasic_pulse["phase_one_charge"] = test_phase_one_charge = randint(10, 50)
    test_biphasic_pulse["postphase_interval"] = test_postphase_interval = randint(1, 100)
    test_biphasic_pulse["num_cycles"] = test_num_cycles = randint(1, 5)
    test_biphasic_pulse["phase_one_duration"] = test_phase_one_duration = randint(1, 100)

    test_start_timepoint = randint(0, 100)
    test_first_cycle_timepoints = (
        np.array([0, 0, test_phase_one_duration, test_phase_one_duration]) + test_start_timepoint
    )
    test_cycle_dur = test_phase_one_duration + test_postphase_interval

    # this value won't have any impact on the output arr due to mocking
    test_stop_timepoint = randint(10000, 20000)
    actual_pulse_arr = create_interpolated_subprotocol_waveform(
        test_biphasic_pulse, test_start_timepoint, test_stop_timepoint
    )
    assert actual_pulse_arr == mocked_truncate.return_value

    expected_timepoints = [
        t + (cycle_num * test_cycle_dur)
        for cycle_num in range(test_num_cycles)
        for t in test_first_cycle_timepoints
    ]
    expected_amplitudes = [0, test_phase_one_charge, test_phase_one_charge, 0] * test_num_cycles

    mocked_truncate.assert_called_once()
    np.testing.assert_array_equal(
        mocked_truncate.call_args[0][0], np.array([expected_timepoints, expected_amplitudes])
    )
    assert mocked_truncate.call_args[0][1] == test_stop_timepoint


def test_create_interpolated_subprotocol_waveform___creates_biphasic_waveform_correctly(mocker):
    mocked_truncate = mocker.patch.object(
        stimulation, "truncate_interpolated_subprotocol_waveform", autospec=True
    )

    # make sure keys are inserted out of order so the insertion order can't be relied on
    test_biphasic_pulse = {"type": "biphasic"}
    test_biphasic_pulse["phase_two_duration"] = test_phase_two_duration = randint(1, 100)
    test_biphasic_pulse["phase_one_duration"] = test_phase_one_duration = randint(1, 100)
    test_biphasic_pulse["postphase_interval"] = test_postphase_interval = randint(1, 100)
    test_biphasic_pulse["interphase_interval"] = test_interphase_interval = randint(1, 100)
    test_biphasic_pulse["num_cycles"] = test_num_cycles = randint(1, 5)
    test_biphasic_pulse["phase_two_charge"] = test_phase_two_charge = -randint(10, 50)
    test_biphasic_pulse["phase_one_charge"] = test_phase_one_charge = randint(10, 50)

    test_start_timepoint = randint(0, 100)
    test_first_cycle_timepoints = np.repeat(
        list(
            itertools.accumulate(
                [test_phase_one_duration, test_interphase_interval, test_phase_two_duration],
                initial=test_start_timepoint,
            )
        ),
        2,
    )
    test_cycle_dur = test_first_cycle_timepoints[-1] + test_postphase_interval - test_start_timepoint

    # this value won't have any impact on the output arr due to mocking
    test_stop_timepoint = randint(10000, 20000)
    actual_pulse_arr = create_interpolated_subprotocol_waveform(
        test_biphasic_pulse, test_start_timepoint, test_stop_timepoint
    )
    assert actual_pulse_arr == mocked_truncate.return_value

    expected_timepoints = [
        t + (cycle_num * test_cycle_dur)
        for cycle_num in range(test_num_cycles)
        for t in test_first_cycle_timepoints
    ]
    expected_amplitudes = [
        0,
        test_phase_one_charge,
        test_phase_one_charge,
        0,
        0,
        test_phase_two_charge,
        test_phase_two_charge,
        0,
    ] * test_num_cycles

    mocked_truncate.assert_called_once()
    np.testing.assert_array_equal(
        mocked_truncate.call_args[0][0], np.array([expected_timepoints, expected_amplitudes])
    )
    assert mocked_truncate.call_args[0][1] == test_stop_timepoint


def test_create_interpolated_protocol_waveform__creates_full_waveform_correctly():
    assert not "TODO"
