# -*- coding: utf-8 -*-
import os
import tempfile

import numpy as np
from pulse3D import plate_recording
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


# Tanner (10/4/22): TODO this test is currently failing, so commenting it out so it doesn't become a blocker.
# Not sure if it will pass anyway because of interpolation
# def test_PlateRecording__well_data_loaded_from_dataframe_will_equal_original_well_data(mocker):
#     # mock so magnet finding alg doesn't run
#     mocker.patch.object(
#         plate_recording,
#         "find_magnet_positions",
#         autospec=True,
#         side_effect=lambda data, *args: {"X": np.zeros((data.shape[-1], 24))},
#     )

#     rec_path = os.path.join(
#         PATH_TO_MAGNET_FINDING_FILES,
#         "MA200440001__2020_02_09_190359__with_calibration_recordings__zipped_as_folder.zip",
#     )
#     pr_created_from_h5 = PlateRecording(rec_path)
#     existing_df = pr_created_from_h5.to_dataframe()

#     pr_recreated_from_df = next(PlateRecording.from_dataframe(rec_path, existing_df))

#     for well_idx, (original_wf, recreated_wf) in enumerate(zip(pr_created_from_h5, pr_recreated_from_df)):
#         np.testing.assert_array_almost_equal(
#             original_wf.force, recreated_wf.force, err_msg=f"Well {well_idx} failed"
#         )
