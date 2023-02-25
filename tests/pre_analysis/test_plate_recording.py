# -*- coding: utf-8 -*-
from collections import defaultdict
import os
from secrets import choice

import numpy as np
from pulse3D import plate_recording
from pulse3D.constants import MICRO_TO_BASE_CONVERSION
from pulse3D.constants import NOT_APPLICABLE_H5_METADATA
from pulse3D.constants import NOT_APPLICABLE_LABEL
from pulse3D.constants import PLATEMAP_LABEL_UUID
from pulse3D.constants import PLATEMAP_NAME_UUID
from pulse3D.constants import TISSUE_SAMPLING_PERIOD_UUID
from pulse3D.constants import TWENTY_FOUR_WELL_PLATE
from pulse3D.constants import WELL_INDEX_UUID
from pulse3D.constants import WELL_NAME_UUID
from pulse3D.magnet_finding import format_well_file_data
from pulse3D.plate_recording import PlateRecording
from pulse3D.plate_recording import WellFile
import pytest

from ..fixtures_utils import PATH_TO_H5_FILES
from ..fixtures_utils import PATH_TO_MAGNET_FINDING_FILES
from ..fixtures_utils import TEST_SMALL_BETA_1_FILE_PATH
from ..fixtures_utils import TEST_SMALL_BETA_2_FILE_PATH

TEST_TWO_STIM_SESSIONS_FILE_PATH = os.path.join(
    PATH_TO_H5_FILES, "stim", "StimInterpolationTest-TwoSessions.zip"
)

TEST_VAR_STIM_SESSIONS_FILE_PATH = os.path.join(
    PATH_TO_H5_FILES, "stim", "StimInterpolationTest-VariableSessions.zip"
)


@pytest.mark.parametrize(
    "test_platemap_name,test_label_meta_options",
    [
        (None, [None]),
        (NOT_APPLICABLE_H5_METADATA, [str(NOT_APPLICABLE_H5_METADATA)]),
        ("test_name", ["testlabel1", "testlabel2"]),
    ],
)
def test_PlateRecording__loads_platemap_info_correctly(test_platemap_name, test_label_meta_options, mocker):
    # mock so magnet finding alg doesn't run
    mocker.patch.object(
        plate_recording,
        "find_magnet_positions",
        autospec=True,
        side_effect=lambda x, *args, **kwargs: {"X": np.empty((x.shape[-1], 24))},
    )

    if test_platemap_name and test_platemap_name != NOT_APPLICABLE_H5_METADATA:
        if len(test_label_meta_options) == 1:
            test_label_metadata = test_label_meta_options * 24
        else:
            test_label_metadata = test_label_meta_options + [
                choice(test_label_meta_options) for _ in range(22)
            ]
        expected_labels = {
            label: [
                TWENTY_FOUR_WELL_PLATE.get_well_name_from_well_index(well_idx)
                for well_idx, label_ in enumerate(test_label_metadata)
                if label_ == label
            ]
            for label in test_label_meta_options
        }
    else:
        expected_labels = defaultdict(list)

    unmocked_load = WellFile._load_data_from_h5_file

    def load_se(wf, file_path):
        unmocked_load(wf, file_path)
        if test_platemap_name and test_platemap_name != NOT_APPLICABLE_H5_METADATA:
            wf.attrs[str(PLATEMAP_NAME_UUID)] = test_platemap_name
            wf.attrs[str(PLATEMAP_LABEL_UUID)] = test_label_metadata[wf[WELL_INDEX_UUID]]
        # these metadata tags are not in the file currently chosen for this test, so don't need to remove them for the test where they shouldn't be present

    mocker.patch.object(
        plate_recording.WellFile, "_load_data_from_h5_file", autospec=True, side_effect=load_se
    )

    pr = PlateRecording(TEST_TWO_STIM_SESSIONS_FILE_PATH)  # arbitrary file chosen

    # test full dict
    assert pr.platemap_labels == expected_labels


def test_PlateRecording__creates_WellFiles_with_correct_value_for_has_inverted_post_magnet(mocker):
    # make sure A1 is always present
    wells_to_flip = ["A1"] + [
        TWENTY_FOUR_WELL_PLATE.get_well_name_from_well_index(well_idx)
        for well_idx in range(1, 24)
        if choice([True, False])
    ]

    pr = PlateRecording(TEST_SMALL_BETA_1_FILE_PATH, inverted_post_magnet_wells=wells_to_flip)

    for wf in pr:
        well_name = wf[WELL_NAME_UUID]
        assert wf.has_inverted_post_magnet is (well_name in wells_to_flip), well_name


@pytest.mark.parametrize("test_file_path", [TEST_SMALL_BETA_1_FILE_PATH, TEST_SMALL_BETA_2_FILE_PATH])
def test_PlateRecording__force_timepoints_start_at_zero(test_file_path, mocker):
    # mock so magnet finding alg doesn't run
    mocker.patch.object(
        plate_recording,
        "find_magnet_positions",
        autospec=True,
        side_effect=lambda x, *args, **kwargs: {"X": np.empty((x.shape[-1], 24))},
    )

    pr = PlateRecording(test_file_path)

    for well_idx, wf in enumerate(pr):
        assert wf.force[0, 0] == 0, well_idx


def test_PlateRecording__stim_timepoints_start_at_zero_or_earlier(mocker):
    # mock so magnet finding alg doesn't run
    mocker.patch.object(
        plate_recording,
        "find_magnet_positions",
        autospec=True,
        side_effect=lambda x, *args, **kwargs: {"X": np.empty((x.shape[-1], 24))},
    )

    pr = PlateRecording(TEST_TWO_STIM_SESSIONS_FILE_PATH)

    for well_idx, wf in enumerate(pr):
        if not wf.stim_sessions:
            continue

        first_session = wf.stim_sessions[0]
        assert first_session[0, 0] <= 0, well_idx


def test_PlateRecording__removes_dropped_samples_from_raw_tissue_signal_before_converting_to_mfd(mocker):
    spied_fix = mocker.spy(plate_recording, "fix_dropped_samples")
    spied_mfd = mocker.spy(plate_recording, "calculate_magnetic_flux_density_from_memsic")
    # mock instead of spy so magnet finding alg doesn't run
    mocker.patch.object(
        plate_recording,
        "find_magnet_positions",
        autospec=True,
        side_effect=lambda data, *args: {"X": np.zeros((data[0].shape[-1], 24))},
    )

    pr = PlateRecording(
        os.path.join(
            PATH_TO_MAGNET_FINDING_FILES,
            "MA200440001__2020_02_09_190359__with_calibration_recordings__zipped_as_folder.zip",
        )
    )
    actual_plate_data = spied_fix.call_args[0][0]
    expected_plate_data = format_well_file_data(pr.wells)
    np.testing.assert_array_equal(actual_plate_data, expected_plate_data)
    np.testing.assert_array_equal(spied_mfd.call_args_list[0][0][0], spied_fix.spy_return)


def test_PlateRecording__slices_data_before_analysis(mocker):
    # mock instead of spy so magnet finding alg doesn't run
    mocked_fmp = mocker.patch.object(
        plate_recording,
        "find_magnet_positions",
        autospec=True,
        side_effect=lambda data, *args: {"X": np.zeros((data[0].shape[-1], 24))},
    )

    test_start_time = 1
    test_end_time = 3.6

    pr = PlateRecording(
        os.path.join(
            PATH_TO_MAGNET_FINDING_FILES,
            "MA200440001__2020_02_09_190359__with_calibration_recordings__zipped_as_folder.zip",
        ),
        start_time=test_start_time,
        end_time=test_end_time,
    )

    recording_sampling_period_us = pr.wells[0][TISSUE_SAMPLING_PERIOD_UUID]
    recording_sampling_freq = MICRO_TO_BASE_CONVERSION / recording_sampling_period_us
    expected_start_idx = int(test_start_time * recording_sampling_freq)
    expected_end_idx = int(test_end_time * recording_sampling_freq)

    # make sure data is sliced before running through the magnet finding alg
    actual_plate_data = mocked_fmp.call_args[0][0]
    expected_plate_data = format_well_file_data(pr.wells)[:, expected_start_idx:expected_end_idx]
    assert actual_plate_data.shape == expected_plate_data.shape

    # also make sure for force that the first time index is correct and the last time index is within two sampling periods from the expected value
    expected_final_time_index = (test_end_time - test_start_time) * MICRO_TO_BASE_CONVERSION
    assert int(pr.wells[0].force[0][0]) == 0
    assert abs(expected_final_time_index - pr.wells[0].force[0][-1]) <= recording_sampling_period_us * 2


def test_PlateRecording__v1_data_loaded_from_dataframe_will_equal_original_well_data(mocker):
    def se(x, *args, **kwargs):
        x_len = x.shape[-1]
        test_data = np.empty((x_len, 24))

        for well_idx in range(24):
            test_data[:, well_idx] = np.arange(x_len) * well_idx
        return {"X": test_data}

    # mock so magnet finding alg doesn't run
    mocker.patch.object(plate_recording, "find_magnet_positions", autospec=True, side_effect=se)

    rec_path = TEST_VAR_STIM_SESSIONS_FILE_PATH

    pr_created_from_h5 = PlateRecording(rec_path)

    existing_df = pr_created_from_h5.to_dataframe()
    # make sure data was actually written to the dataframe
    for col in existing_df:
        assert existing_df[col].shape > (2, 0), existing_df

    pr_recreated_from_df = PlateRecording(rec_path, recording_df=existing_df)

    for well_idx, (original_wf, recreated_wf) in enumerate(zip(pr_created_from_h5, pr_recreated_from_df)):
        # to_dataframe normalizes time points so a PlateRecording sometimes has timepoints like
        # 0.0, 10000.0, 20003.0, 30000.0, 40001.0, 50000.0, 60001.0, 70001.0, 80001.0, 90001.0 and to_dataframe makes them
        # 0.0, 10000.0, 20000.0, 30000.0, 40000.0, 50000.0, 60000.0, 70000.0, 80000.0, 90000.0

        # convert to seconds and then only check two decimal places
        original_wf.force[0] /= MICRO_TO_BASE_CONVERSION
        recreated_wf.force[0] /= MICRO_TO_BASE_CONVERSION

        np.testing.assert_array_almost_equal(
            recreated_wf.force, original_wf.force, decimal=2, err_msg=f"Well {well_idx} force"
        )

        assert len(original_wf.stim_sessions) == len(recreated_wf.stim_sessions), f"Well {well_idx}"

        for session_idx, (original_session_data, recreated_session_data) in enumerate(
            zip(original_wf.stim_sessions, recreated_wf.stim_sessions)
        ):
            np.testing.assert_array_equal(
                recreated_session_data,
                original_session_data,
                err_msg=f"Well {well_idx}, Stim Session {session_idx}",
            )


def test_PlateRecording__overrides_h5_platemap_groups_if_well_groups_param_is_not_none(mocker):
    # mock so magnet finding alg doesn't run
    mocker.patch.object(
        plate_recording,
        "find_magnet_positions",
        autospec=True,
        side_effect=lambda x, *args, **kwargs: {"X": np.empty((x.shape[-1], 24))},
    )

    test_label_meta_options = ["label_one", "label_two"]
    test_platemap_meta_name = "original_platemap_name"
    test_label_metadata = (
        test_label_meta_options + test_label_meta_options + [NOT_APPLICABLE_LABEL for _ in range(20)]
    )
    unmocked_load = WellFile._load_data_from_h5_file

    def load_se(wf, file_path):
        unmocked_load(wf, file_path)
        wf.attrs[str(PLATEMAP_NAME_UUID)] = test_platemap_meta_name
        wf.attrs[str(PLATEMAP_LABEL_UUID)] = test_label_metadata[wf[WELL_INDEX_UUID]]

        # these metadata tags are not in the file currently chosen for this test, so don't need to remove them for the test where they shouldn't be present

    mocker.patch.object(
        plate_recording.WellFile, "_load_data_from_h5_file", autospec=True, side_effect=load_se
    )

    pr = PlateRecording(TEST_TWO_STIM_SESSIONS_FILE_PATH)
    # test full dict
    assert pr.platemap_labels == defaultdict(**{"label_one": ["A1", "C1"], "label_two": ["B1", "D1"]})

    new_well_groups = {
        "new_label_one": ["A2", "A3"],
        "new_label_two": ["B1"],
        "new_label_three": ["B4", "C5", "D6"],
    }
    pr = PlateRecording(
        TEST_TWO_STIM_SESSIONS_FILE_PATH, well_groups=new_well_groups
    )  # arbitrary file chosen

    # test full dict
    assert pr.platemap_labels == defaultdict(**new_well_groups)
