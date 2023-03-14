# -*- coding: utf-8 -*-
import itertools
from random import randint

import numpy as np
from pulse3D import stimulation
from pulse3D.constants import STIM_COMPLETE_SUBPROTOCOL_IDX
from pulse3D.exceptions import SubprotocolFormatIncompatibleWithInterpolationError
from pulse3D.stimulation import aggregate_timepoints
from pulse3D.stimulation import create_interpolated_subprotocol_waveform
from pulse3D.stimulation import create_stim_session_waveforms
from pulse3D.stimulation import interpolate_stim_session
from pulse3D.stimulation import realign_interpolated_stim_data
from pulse3D.stimulation import remove_intermediate_interpolation_data
from pulse3D.stimulation import truncate_interpolated_subprotocol_waveform
import pytest

from .fixtures_utils import rand_bool


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


def test_truncate_interpolated_subprotocol_waveform__returns_empty_array_if_given_empty_array():
    actual_truncated_arr = truncate_interpolated_subprotocol_waveform(
        np.empty((2, 0)), randint(0, 1000), from_start=rand_bool()
    )
    np.testing.assert_array_equal(actual_truncated_arr, np.empty((2, 0)))


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


def test_remove_intermediate_interpolation_data__handles_empty_array_correctly():
    empty_array = np.array([])
    actual = remove_intermediate_interpolation_data(empty_array, randint(1, 10))
    np.testing.assert_array_equal(actual, empty_array)


@pytest.mark.parametrize("test_num_occurences", list(range(3)))
def test_remove_intermediate_interpolation_data__does_not_modify_array_if_no_data_needs_to_be_removed(
    test_num_occurences,
):
    test_timepoints = np.array([1] + ([2] * test_num_occurences) + [3])
    test_array = np.array([test_timepoints, np.zeros(test_timepoints.shape)])

    actual = remove_intermediate_interpolation_data(test_array, 2)
    np.testing.assert_array_equal(actual, test_array)


@pytest.mark.parametrize("test_num_occurences", list(range(3, 6)))
def test_remove_intermediate_interpolation_data__removes_correct_amount_of_data_when_necessary(
    test_num_occurences,
):
    test_timepoints = np.array([1] + ([2] * test_num_occurences) + [3])
    test_array = np.array([test_timepoints, np.zeros(test_timepoints.shape)])

    actual = remove_intermediate_interpolation_data(test_array, 2)

    expected_array = np.array([[1, 2, 2, 3], [0] * 4])
    np.testing.assert_array_equal(actual, expected_array)


def test_create_interpolated_subprotocol_waveform__raises_error_if_subprotocol_is_not_of_a_supported_format():
    with pytest.raises(SubprotocolFormatIncompatibleWithInterpolationError, match="Must have a 'type'"):
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
            # other args don't matter here
            randint(0, 100),
            randint(100, 200),
            rand_bool(),
        )


def test_create_interpolated_subprotocol_waveform__raises_error_if_loop_given():
    with pytest.raises(ValueError, match="Cannot interpolate loops"):
        create_interpolated_subprotocol_waveform(
            {"type": "loop"},
            # other args don't matter here
            randint(0, 100),
            randint(100, 200),
            rand_bool(),
        )


@pytest.mark.parametrize("include_start_timepoint", [True, False])
def test_create_interpolated_subprotocol_waveform__creates_delay_correctly(include_start_timepoint, mocker):
    mocked_truncate = mocker.patch.object(
        stimulation, "truncate_interpolated_subprotocol_waveform", autospec=True
    )

    test_delay = {"type": "delay", "duration": randint(0, 100)}

    test_start_timepoint = randint(0, 100)
    test_stop_timepoint = randint(100, 200)

    actual_delay_arr = create_interpolated_subprotocol_waveform(
        test_delay, test_start_timepoint, test_stop_timepoint, include_start_timepoint
    )
    assert actual_delay_arr == mocked_truncate.return_value

    expected_original_delay_arr = np.array([[test_start_timepoint, test_stop_timepoint], [0, 0]])

    mocked_truncate.assert_called_once()
    np.testing.assert_array_equal(mocked_truncate.call_args[0][0], expected_original_delay_arr)
    assert mocked_truncate.call_args[0][1] == test_stop_timepoint


@pytest.mark.parametrize("include_start_timepoint", [True, False])
def test_create_interpolated_subprotocol_waveform__creates_monophasic_waveform_correctly(
    include_start_timepoint, mocker
):
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

    test_stop_timepoint = randint(1000, 1300)
    actual_pulse_arr = create_interpolated_subprotocol_waveform(
        test_biphasic_pulse, test_start_timepoint, test_stop_timepoint, include_start_timepoint
    )
    assert actual_pulse_arr == mocked_truncate.return_value

    expected_timepoints = [
        t + (cycle_num * test_cycle_dur)
        for cycle_num in range(test_num_cycles)
        for t in test_first_cycle_timepoints
    ]
    expected_amplitudes = [test_phase_one_charge, test_phase_one_charge, 0, 0] * test_num_cycles

    if include_start_timepoint:
        expected_timepoints = [test_start_timepoint] + expected_timepoints
        expected_amplitudes = [0] + expected_amplitudes

    expected_arr = np.array([expected_timepoints, expected_amplitudes])

    # need to make assertion on the array separately
    mocked_truncate.assert_called_once_with(mocker.ANY, test_stop_timepoint, from_start=False)
    np.testing.assert_array_equal(mocked_truncate.call_args[0][0], expected_arr)


@pytest.mark.parametrize("include_start_timepoint", [True, False])
def test_create_interpolated_subprotocol_waveform__creates_biphasic_waveform_correctly__with_non_zero_interphase_interval(
    include_start_timepoint, mocker
):
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

    test_stop_timepoint = randint(1000, 2000)
    actual_pulse_arr = create_interpolated_subprotocol_waveform(
        test_biphasic_pulse, test_start_timepoint, test_stop_timepoint, include_start_timepoint
    )
    assert actual_pulse_arr == mocked_truncate.return_value

    expected_timepoints = [
        t + (cycle_num * test_cycle_dur)
        for cycle_num in range(test_num_cycles)
        for t in test_first_cycle_timepoints
    ]
    expected_amplitudes = [
        test_phase_one_charge,
        test_phase_one_charge,
        0,
        0,
        test_phase_two_charge,
        test_phase_two_charge,
        0,
        0,
    ] * test_num_cycles
    if include_start_timepoint:
        expected_timepoints = [test_start_timepoint] + expected_timepoints
        expected_amplitudes = [0] + expected_amplitudes

    expected_arr = np.array([expected_timepoints, expected_amplitudes])

    # need to make assertion on the array separately
    mocked_truncate.assert_called_once_with(mocker.ANY, test_stop_timepoint, from_start=False)
    np.testing.assert_array_equal(mocked_truncate.call_args[0][0], expected_arr)


@pytest.mark.parametrize("include_start_timepoint", [True, False])
def test_create_interpolated_subprotocol_waveform__creates_biphasic_waveform_correctly__with_zero_interphase_interval(
    include_start_timepoint, mocker
):
    mocked_truncate = mocker.patch.object(
        stimulation, "truncate_interpolated_subprotocol_waveform", autospec=True
    )

    # make sure keys are inserted out of order so the insertion order can't be relied on
    test_biphasic_pulse = {"type": "biphasic"}
    test_biphasic_pulse["phase_two_duration"] = test_phase_two_duration = randint(1, 100)
    test_biphasic_pulse["phase_one_duration"] = test_phase_one_duration = randint(1, 100)
    test_biphasic_pulse["postphase_interval"] = test_postphase_interval = randint(1, 100)
    test_biphasic_pulse["interphase_interval"] = 0
    test_biphasic_pulse["num_cycles"] = test_num_cycles = randint(1, 5)
    test_biphasic_pulse["phase_two_charge"] = test_phase_two_charge = -randint(10, 50)
    test_biphasic_pulse["phase_one_charge"] = test_phase_one_charge = randint(10, 50)

    test_start_timepoint = randint(0, 100)
    test_first_cycle_timepoints = np.repeat(
        list(
            itertools.accumulate(
                [
                    test_phase_one_duration,
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
    test_stop_timepoint = randint(1000, 2000)
    actual_pulse_arr = create_interpolated_subprotocol_waveform(
        test_biphasic_pulse, test_start_timepoint, test_stop_timepoint, include_start_timepoint
    )
    assert actual_pulse_arr == mocked_truncate.return_value

    expected_timepoints = [
        t + (cycle_num * test_cycle_dur)
        for cycle_num in range(test_num_cycles)
        for t in test_first_cycle_timepoints
    ]
    expected_amplitudes = [
        test_phase_one_charge,
        test_phase_one_charge,
        test_phase_two_charge,
        test_phase_two_charge,
        0,
        0,
    ] * test_num_cycles

    if include_start_timepoint:
        expected_timepoints = [test_start_timepoint] + expected_timepoints
        expected_amplitudes = [0] + expected_amplitudes

    expected_arr = np.array([expected_timepoints, expected_amplitudes])

    # need to make assertion on the array separately
    mocked_truncate.assert_called_once_with(mocker.ANY, test_stop_timepoint, from_start=False)
    np.testing.assert_array_equal(mocked_truncate.call_args[0][0], expected_arr)


def test_interpolate_stim_session__returns_empty_array_if_start_timepoint_is_greater_than_protocol_complete_status_of_stim_status_updates():
    test_start_timepoint = 0
    test_stop_timepoint = 100

    actual_waveform_arr = interpolate_stim_session(
        [],  # not needed for this test
        np.array(
            [
                [test_start_timepoint, test_start_timepoint - 1],
                [randint(0, STIM_COMPLETE_SUBPROTOCOL_IDX - 1), STIM_COMPLETE_SUBPROTOCOL_IDX],
            ]
        ),
        test_start_timepoint,
        test_stop_timepoint,
    )
    np.testing.assert_array_equal(actual_waveform_arr, np.empty((2, 0)))


def test_interpolate_stim_session__returns_empty_array_if_stop_timepoint_is_less_than_first_timepoint_of_stim_status_updates():
    test_start_timepoint = 0
    test_stop_timepoint = 100

    actual_waveform_arr = interpolate_stim_session(
        [],  # not needed for this test
        np.array(
            [
                [test_stop_timepoint + 1, test_stop_timepoint],
                [
                    randint(0, STIM_COMPLETE_SUBPROTOCOL_IDX - 1),
                    randint(0, STIM_COMPLETE_SUBPROTOCOL_IDX - 1),
                ],
            ]
        ),
        test_start_timepoint,
        test_stop_timepoint,
    )
    np.testing.assert_array_equal(actual_waveform_arr, np.empty((2, 0)))


@pytest.mark.parametrize("final_subprotocol_completes", [True, False])
def test_interpolate_stim_session__creates_full_waveform_correctly(final_subprotocol_completes, mocker):
    # arbitrary values in these arrays
    test_waveforms = [np.arange(0, 10).reshape((2, 5)), np.arange(10, 20).reshape((2, 5))]
    test_cleaned_waveforms = [np.arange(20, 26).reshape((2, 3)), np.arange(26, 32).reshape((2, 3))]

    mocked_create = mocker.patch.object(
        stimulation,
        "create_interpolated_subprotocol_waveform",
        autospec=True,
        side_effect=test_waveforms,
    )
    mocked_remove = mocker.patch.object(
        stimulation,
        "remove_intermediate_interpolation_data",
        autospec=True,
        side_effect=test_cleaned_waveforms,
    )
    mocked_truncate = mocker.patch.object(
        stimulation, "truncate_interpolated_subprotocol_waveform", autospec=True
    )

    test_start_timepoint = randint(0, 100)
    test_stop_timepoint = randint(200, 300)

    test_stim_status_updates = np.array([[100, 130], [0, 1]])
    if final_subprotocol_completes:
        test_stim_status_updates = np.concatenate(
            [test_stim_status_updates, np.array([[150], [STIM_COMPLETE_SUBPROTOCOL_IDX]])], axis=1
        )

    test_subprotocols = get_test_subprotocols()

    actual_interpolated_session = interpolate_stim_session(
        test_subprotocols, test_stim_status_updates, test_start_timepoint, test_stop_timepoint
    )
    assert actual_interpolated_session == mocked_truncate.return_value

    assert mocked_create.call_args_list == [
        mocker.call(
            test_subprotocols[0], test_stim_status_updates[0, 0], test_stim_status_updates[0, 1], True
        ),
        mocker.call(test_subprotocols[1], test_stim_status_updates[0, 1], test_stop_timepoint, False),
    ]

    expected_waveforms_in_remove_calls = [
        test_waveforms[0],
        np.concatenate([test_cleaned_waveforms[0], test_waveforms[1]], axis=1),
    ]
    for i, call_args in enumerate(mocked_remove.call_args_list):
        np.testing.assert_array_equal(
            call_args[0][0], expected_waveforms_in_remove_calls[i], err_msg=f"Call {i}"
        )

    mocked_truncate.assert_called_once_with(mocker.ANY, test_start_timepoint, from_start=True)
    np.testing.assert_array_equal(mocked_truncate.call_args[0][0], test_cleaned_waveforms[1])


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

    test_final_status_update = STIM_COMPLETE_SUBPROTOCOL_IDX if final_protocol_completes else 0
    test_stim_status_updates = np.array([np.arange(0, 4) * 100, [0, 1, 2, test_final_status_update]])
    test_initial_timepoint = test_stim_status_updates[0, 0]
    test_final_timepoint = test_stim_status_updates[0, -1]

    # set to a timepoint earlier than it should've completed
    if not final_protocol_completes:
        test_final_timepoint -= 1

    test_subprotocols = get_test_subprotocols()

    actual_waveforms = create_stim_session_waveforms(
        test_subprotocols,
        test_stim_status_updates,
        test_initial_timepoint,
        test_final_timepoint,
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

    test_final_status_update = STIM_COMPLETE_SUBPROTOCOL_IDX if final_protocol_completes else 0
    test_session_arrs = [
        np.array([[100, 110], [1, STIM_COMPLETE_SUBPROTOCOL_IDX]]),
        np.array([[200, 201, 202, 203], [0, 1, 0, STIM_COMPLETE_SUBPROTOCOL_IDX]]),
        np.array([[300, 301, 302, 303], [0, 1, 2, test_final_status_update]]),
    ]
    test_stim_status_updates = np.concatenate(test_session_arrs, axis=1)

    test_subprotocols = get_test_subprotocols()

    actual_waveforms = create_stim_session_waveforms(
        test_subprotocols,
        test_stim_status_updates,
        test_initial_timepoint,
        test_final_timepoint,
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


def test_aggregate_timepoints__returns_correct_values_sorted():
    test_timepoint_arrs = [np.arange(0, 35, 5), np.arange(0, 33, 3), np.array([22, 13, 7]), np.array([])]
    expected_timepoint_aggregate = [0, 3, 5, 6, 7, 9, 10, 12, 13, 15, 18, 20, 21, 22, 24, 25, 27, 30]

    actual_timepoints = aggregate_timepoints(test_timepoint_arrs)
    np.testing.assert_array_equal(actual_timepoints, expected_timepoint_aggregate)


def test_realign_interpolated_stim_data__returns_correct_array():
    new_timepoints = np.repeat(np.arange(0, 15, 1), 2)
    orignal_stim_status_data = np.array([[0, 0, 3, 3, 14], [10, 20, 20, 30, 30]])

    actual_adjusted_stim_status_data = realign_interpolated_stim_data(
        new_timepoints, orignal_stim_status_data
    )
    expected_data = np.array([10, 20] + ([np.NaN] * 4) + [20, 30] + ([np.NaN] * 20) + [30, np.NaN])
    np.testing.assert_array_equal(actual_adjusted_stim_status_data, expected_data)
