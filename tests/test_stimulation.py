# -*- coding: utf-8 -*-
import itertools
from random import randint

import numpy as np
from pulse3D import stimulation
from pulse3D.stimulation import create_interpolated_subprotocol_waveform
from pulse3D.stimulation import create_stim_session_waveforms
from pulse3D.stimulation import interpolate_stim_session
from pulse3D.stimulation import truncate_interpolated_subprotocol_waveform
import pytest


def get_test_subprotocols():
    return [
        {
            "type": "biphasic",
            "phase_one_duration": 1,
            "phase_one_charge": 15,
            "interphase_interval": 2,
            "phase_two_charge": -20,
            "phase_two_duration": 3,
            "postphase_interval": 4,
            "num_cycles": 3,
        },
        {"type": "delay", "duration": 20},
        {
            "type": "monophasic",
            "phase_one_duration": 9,
            "phase_one_charge": 7,
            "postphase_interval": 6,
            "num_cycles": 2,
        },
    ]


@pytest.mark.parametrize(
    "test_cutoff_timepoint,expected_truncated_arr",
    [
        (-1, np.array([[], []])),
        (0, np.array([[], []])),
        (1, np.array([[0, 0, 1], [0, 2, 2]])),
        (5, np.array([[0, 0, 5], [0, 2, 2]])),
        (6, np.array([[0, 0, 5, 5, 6], [0, 2, 2, 0, 0]])),
        (10, np.array([[0, 0, 5, 5, 10], [0, 2, 2, 0, 0]])),
        (11, np.array([[0, 0, 5, 5, 10, 10, 11], [0, 2, 2, 0, 0, -2, -2]])),
        (15, np.array([[0, 0, 5, 5, 10, 10, 15, 15], [0, 2, 2, 0, 0, -2, -2, 0]])),
        (16, np.array([[0, 0, 5, 5, 10, 10, 15, 15], [0, 2, 2, 0, 0, -2, -2, 0]])),
    ],
)
def test_truncate_interpolated_subprotocol_waveform__returns_correctly_truncated_array__when_truncating_from_the_end(
    test_cutoff_timepoint, expected_truncated_arr
):
    # using a biphasic array since it contains all array components present in monophasic and delay
    test_biphasic_arr = np.array(
        [
            [0, 0, 5, 5, 10, 10, 15, 15],
            [0, 2, 2, 0, 0, -2, -2, 0],
        ]
    )

    actual_truncated_arr = truncate_interpolated_subprotocol_waveform(
        test_biphasic_arr, test_cutoff_timepoint, from_start=False
    )
    np.testing.assert_array_equal(actual_truncated_arr, expected_truncated_arr)


@pytest.mark.parametrize(
    "test_cutoff_timepoint,expected_truncated_arr",
    [
        (16, np.array([[], []])),
        (15, np.array([[], []])),
        (14, np.array([[14, 15, 15], [-2, -2, 0]])),
        (10, np.array([[10, 15, 15], [-2, -2, 0]])),
        (9, np.array([[9, 10, 10, 15, 15], [0, 0, -2, -2, 0]])),
        (5, np.array([[5, 10, 10, 15, 15], [0, 0, -2, -2, 0]])),
        (4, np.array([[4, 5, 5, 10, 10, 15, 15], [2, 2, 0, 0, -2, -2, 0]])),
        (0, np.array([[0, 0, 5, 5, 10, 10, 15, 15], [0, 2, 2, 0, 0, -2, -2, 0]])),
        (-1, np.array([[0, 0, 5, 5, 10, 10, 15, 15], [0, 2, 2, 0, 0, -2, -2, 0]])),
    ],
)
def test_truncate_interpolated_subprotocol_waveform__returns_correctly_truncated_array__when_truncating_from_the_start(
    test_cutoff_timepoint, expected_truncated_arr
):
    # using a biphasic array since it contains all array components present in monophasic and delay
    test_biphasic_arr = np.array(
        [
            [0, 0, 5, 5, 10, 10, 15, 15],
            [0, 2, 2, 0, 0, -2, -2, 0],
        ]
    )

    actual_truncated_arr = truncate_interpolated_subprotocol_waveform(
        test_biphasic_arr, test_cutoff_timepoint, from_start=True
    )
    np.testing.assert_array_equal(actual_truncated_arr, expected_truncated_arr)


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


def test_create_interpolated_subprotocol_waveform__creates_delay_correctly(mocker):
    mocked_truncate = mocker.patch.object(
        stimulation, "truncate_interpolated_subprotocol_waveform", autospec=True
    )

    test_duration = randint(0, 100)
    test_delay = {"type": "delay", "duration": test_duration}

    test_start_timepoint = randint(0, 100)
    test_stop_timepoint = randint(100, 200)

    actual_delay_arr = create_interpolated_subprotocol_waveform(
        test_delay, test_start_timepoint, test_stop_timepoint
    )
    assert actual_delay_arr == mocked_truncate.return_value

    expected_original_delay_arr = np.array(
        [[test_start_timepoint, test_start_timepoint + test_duration], [0, 0]]
    )

    mocked_truncate.assert_called_once()
    np.testing.assert_array_equal(mocked_truncate.call_args[0][0], expected_original_delay_arr)
    assert mocked_truncate.call_args[0][1] == test_stop_timepoint


def test_create_interpolate_subprotocol_waveform__creates_monophasic_waveform_correctly(mocker):
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
    test_cycle_dur = test_phase_one_duration + test_postphase_interval
    test_first_cycle_timepoints = (
        np.array([0, test_phase_one_duration, test_phase_one_duration, test_cycle_dur]) + test_start_timepoint
    )

    # this value won't have any impact on the output arr due to mocking
    test_stop_timepoint = randint(10000, 20000)
    actual_pulse_arr = create_interpolated_subprotocol_waveform(
        test_biphasic_pulse, test_start_timepoint, test_stop_timepoint
    )
    assert actual_pulse_arr == mocked_truncate.return_value

    expected_timepoints = [test_start_timepoint] + [
        t + (cycle_num * test_cycle_dur)
        for cycle_num in range(test_num_cycles)
        for t in test_first_cycle_timepoints
    ]
    expected_amplitudes = [0] + [test_phase_one_charge, test_phase_one_charge, 0, 0] * test_num_cycles

    # need to make assertion on the array separately
    mocked_truncate.assert_called_once_with(mocker.ANY, test_stop_timepoint, from_start=False)
    np.testing.assert_array_equal(
        mocked_truncate.call_args[0][0], np.array([expected_timepoints, expected_amplitudes])
    )


def test_create_interpolate_subprotocol_waveform__creates_biphasic_waveform_correctly(mocker):
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
                [
                    test_phase_one_duration,
                    test_interphase_interval,
                    test_phase_two_duration,
                    test_postphase_interval,
                ],
                initial=test_start_timepoint,
            )
        ),
        2,
    )[1:-1]
    test_cycle_dur = test_first_cycle_timepoints[-1] - test_start_timepoint

    # this value won't have any impact on the output arr due to mocking
    test_stop_timepoint = randint(10000, 20000)
    actual_pulse_arr = create_interpolated_subprotocol_waveform(
        test_biphasic_pulse, test_start_timepoint, test_stop_timepoint
    )
    assert actual_pulse_arr == mocked_truncate.return_value

    expected_timepoints = [test_start_timepoint] + [
        t + (cycle_num * test_cycle_dur)
        for cycle_num in range(test_num_cycles)
        for t in test_first_cycle_timepoints
    ]
    expected_amplitudes = [0] + [
        test_phase_one_charge,
        test_phase_one_charge,
        0,
        0,
        test_phase_two_charge,
        test_phase_two_charge,
        0,
        0,
    ] * test_num_cycles

    # need to make assertion on the array separately
    mocked_truncate.assert_called_once_with(mocker.ANY, test_stop_timepoint, from_start=False)
    np.testing.assert_array_equal(
        mocked_truncate.call_args[0][0], np.array([expected_timepoints, expected_amplitudes])
    )


def test_interpolate_stim_session__returns_empty_array_if_start_timepoint_is_greater_than_protocol_complete_status_of_stim_status_updates():
    test_start_timepoint = 0
    test_stop_timepoint = 100

    actual_waveform_arr = interpolate_stim_session(
        [],  # not needed for this test
        np.array([[test_start_timepoint, test_start_timepoint - 1], [randint(0, 254), 255]]),
        test_start_timepoint,
        test_stop_timepoint,
    )
    np.testing.assert_array_equal(actual_waveform_arr, np.empty((2, 0)))


def test_interpolate_stim_session__returns_empty_array_if_stop_timepoint_is_less_than_first_timepoint_of_stim_status_updates():
    test_start_timepoint = 0
    test_stop_timepoint = 100

    actual_waveform_arr = interpolate_stim_session(
        [],  # not needed for this test
        np.array([[test_stop_timepoint + 1, test_stop_timepoint], [randint(0, 254), randint(0, 254)]]),
        test_start_timepoint,
        test_stop_timepoint,
    )
    np.testing.assert_array_equal(actual_waveform_arr, np.empty((2, 0)))


@pytest.mark.parametrize("final_subprotocol_completes", [True, False])
def test_interpolate_stim_session__creates_full_waveform_correctly(final_subprotocol_completes, mocker):
    test_waveforms = [np.arange(0, 10).reshape((2, 5)), np.arange(10, 20).reshape((2, 5))]

    mocked_create = mocker.patch.object(
        stimulation,
        "create_interpolated_subprotocol_waveform",
        autospec=True,
        side_effect=test_waveforms,
    )
    mocked_truncate = mocker.patch.object(
        stimulation, "truncate_interpolated_subprotocol_waveform", autospec=True
    )

    test_start_timepoint = randint(0, 100)
    test_stop_timepoint = randint(200, 300)

    test_stim_status_updates = np.array([[100, 130], [0, 1]])
    if final_subprotocol_completes:
        test_stim_status_updates = np.concatenate(
            [test_stim_status_updates, np.array([[150], [255]])], axis=1
        )

    test_subprotocols = get_test_subprotocols()

    actual_interpolated_session = interpolate_stim_session(
        test_subprotocols, test_stim_status_updates, test_start_timepoint, test_stop_timepoint
    )
    assert actual_interpolated_session == mocked_truncate.return_value

    assert mocked_create.call_args_list == [
        mocker.call(test_subprotocols[0], test_stim_status_updates[0, 0], test_stim_status_updates[0, 1]),
        mocker.call(test_subprotocols[1], test_stim_status_updates[0, 1], test_stop_timepoint),
    ]

    mocked_truncate.assert_called_once_with(mocker.ANY, test_start_timepoint, from_start=True)
    np.testing.assert_array_equal(mocked_truncate.call_args[0][0], np.concatenate(test_waveforms, axis=1))


@pytest.mark.parametrize("final_protocol_completes", [True, False])
def test_create_stim_session_waveforms__creates_list_of_session_waveforms_correctly__with_a_single_session_given(
    final_protocol_completes, mocker
):
    expected_mock_waveform_return = mocker.Mock()

    def se(*args, **kwargs):
        # still calling actual method to make sure no errors are raised
        interpolate_stim_session(*args, **kwargs)

        return expected_mock_waveform_return

    mocked_interpolate_stim_session = mocker.patch.object(
        stimulation, "interpolate_stim_session", autospec=True, side_effect=se
    )

    test_final_status_update = 255 if final_protocol_completes else 0
    test_stim_status_updates = np.array([np.arange(0, 4) * 100, [0, 1, 2, test_final_status_update]])
    test_initial_timepoint = test_stim_status_updates[0, 0]
    test_final_timepoint = test_stim_status_updates[0, -1]

    # set to a timepoint earlier than it should've completed
    if not final_protocol_completes:
        test_final_timepoint -= 1

    test_subprotocols = get_test_subprotocols()

    actual_waveforms = create_stim_session_waveforms(
        test_subprotocols, test_stim_status_updates, test_initial_timepoint, test_final_timepoint
    )
    assert actual_waveforms == [expected_mock_waveform_return]

    mocked_interpolate_stim_session.assert_called_once_with(
        test_subprotocols, mocker.ANY, test_initial_timepoint, test_final_timepoint
    )
    # have to make assertions on the stim_status_updates array separately
    np.testing.assert_array_equal(mocked_interpolate_stim_session.call_args[0][1], test_stim_status_updates)


@pytest.mark.parametrize("final_protocol_completes", [True, False])
def test_create_stim_session_waveforms__creates_list_of_session_waveforms_correctly__with_multiple_sessions_given(
    final_protocol_completes, mocker
):
    test_initial_timepoint = randint(0, 100)
    test_final_timepoint = randint(400, 500)

    expected_mock_waveform_returns = []

    def se(*args, **kwargs):
        # still calling actual method to make sure no errors are raised
        interpolate_stim_session(*args, **kwargs)

        mock_return = mocker.Mock()
        expected_mock_waveform_returns.append(mock_return)
        return mock_return

    mocked_interpolate_stim_session = mocker.patch.object(
        stimulation, "interpolate_stim_session", autospec=True, side_effect=se
    )

    test_final_status_update = 255 if final_protocol_completes else 0
    test_session_arrs = [
        np.array([[100, 110], [1, 255]]),
        np.array([[200, 201, 202, 203], [0, 1, 0, 255]]),
        np.array([[300, 301, 302, 303], [0, 1, 2, test_final_status_update]]),
    ]
    test_stim_status_updates = np.concatenate(test_session_arrs, axis=1)

    test_subprotocols = get_test_subprotocols()

    actual_waveforms = create_stim_session_waveforms(
        test_subprotocols, test_stim_status_updates, test_initial_timepoint, test_final_timepoint
    )
    assert actual_waveforms == expected_mock_waveform_returns

    expected_final_timepoints = [
        test_session_arrs[0][0, -1],
        test_session_arrs[1][0, -1],
        test_session_arrs[2][0, -1] if final_protocol_completes else test_final_timepoint,
    ]

    assert mocked_interpolate_stim_session.call_args_list == [
        mocker.call(test_subprotocols, mocker.ANY, test_initial_timepoint, final_timepoint)
        for final_timepoint in expected_final_timepoints
    ]
    # have to make assertions on the stim_session_arrays_separately
    for i, stim_session_arr in enumerate(test_session_arrs):
        np.testing.assert_array_equal(
            mocked_interpolate_stim_session.call_args_list[i][0][1], stim_session_arr
        )
