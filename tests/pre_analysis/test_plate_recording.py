# -*- coding: utf-8 -*-
import os
import tempfile

import numpy as np
from pulse3D import plate_recording
from pulse3D.constants import MICRO_TO_BASE_CONVERSION
from pulse3D.constants import TISSUE_SAMPLING_PERIOD_UUID
from pulse3D.magnet_finding import format_well_file_data
from pulse3D.plate_recording import PlateRecording

from ..fixtures_utils import PATH_TO_H5_FILES
from ..fixtures_utils import PATH_TO_MAGNET_FINDING_FILES


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


def test_PlateRecording__writes_time_force_csv_with_no_errors(mocker):
    # mock instead of spy so magnet finding alg doesn't run
    mocker.patch.object(
        plate_recording,
        "find_magnet_positions",
        autospec=True,
        side_effect=lambda data, *args: {"X": np.zeros((data.shape[-1], 24))},
    )

    zip_pr = PlateRecording(
        os.path.join(
            PATH_TO_MAGNET_FINDING_FILES,
            "MA200440001__2020_02_09_190359__with_calibration_recordings__zipped_as_folder.zip",
        )
    )
    h5_pr = PlateRecording.from_directory(os.path.join(PATH_TO_H5_FILES, "v0.3.2"))
    # raw_baseline_data = spied_mfd_from_memsic.spy_return
    with tempfile.TemporaryDirectory() as output_dir:
        zip_pr.write_time_force_csv(output_dir)
        for pr in h5_pr:
            df, _ = pr.write_time_force_csv(output_dir)
            assert len(df.index) == 7975
            assert len(df.columns) == 25

        assert (
            "MA200440001__2020_02_09_190359__with_calibration_recordings__zipped_as_folder.csv"
            in os.listdir(output_dir)
        )
        assert "MA20223322__2020_09_02_173919.csv" in os.listdir(output_dir)


def test_PlateRecording__beta_1_data_loaded_from_dataframe_will_equal_original_well_data(mocker):
    pass  # TODO


def test_PlateRecording__v1_data_loaded_from_dataframe_will_equal_original_well_data(mocker):
    def se(x, *args, **kwargs):
        x_len = x.shape[-1]
        test_data = np.empty((x_len, 24))

        for well_idx in range(24):
            test_data[:, well_idx] = np.arange(x_len) * well_idx
        return {"X": test_data}

    # mock so magnet finding alg doesn't run
    mocker.patch.object(plate_recording, "find_magnet_positions", autospec=True, side_effect=se)

    rec_path = os.path.join(PATH_TO_H5_FILES, "stim", "StimInterpolationTest-TwoSessions.zip")

    pr_created_from_h5 = PlateRecording(rec_path)
    existing_df = pr_created_from_h5.to_dataframe()
    pr_recreated_from_df = PlateRecording(rec_path, force_df=existing_df)

    for well_idx, (original_wf, recreated_wf) in enumerate(zip(pr_created_from_h5, pr_recreated_from_df)):
        # to_dataframe normalizes time points so platerecording sometimes has timepoints like
        # 0.0, 10000.0, 20003.0, 30000.0, 40001.0, 50000.0, 60001.0, 70001.0, 80001.0, 90001.0 and to_dataframe makes them
        # 0.0, 10000.0, 20000.0, 30000.0, 40000.0, 50000.0, 60000.0, 70000.0, 80000.0, 90000.0
        # convert to seconds and then only check two decimal places
        original_wf.force[0] /= MICRO_TO_BASE_CONVERSION
        recreated_wf.force[0] /= MICRO_TO_BASE_CONVERSION

        np.testing.assert_array_almost_equal(
            original_wf.force, recreated_wf.force, decimal=2, err_msg=f"Well {well_idx} force"
        )

        # np.testing.assert_array_almost_equal(
        #     original_wf.stim_readings, recreated_wf.force, decimal=2, err_msg=f"Well {well_idx} stim"
        # )
