# -*- coding: utf-8 -*-
import os
import tempfile
import zipfile

from mantarray_magnet_finding.utils import calculate_magnetic_flux_density_from_memsic
from mantarray_magnet_finding.utils import load_h5_folder_as_array
import numpy as np
from pulse3D import magnet_finding
from pulse3D import plate_recording
from pulse3D.constants import BASELINE_MEAN_NUM_DATA_POINTS
from pulse3D.plate_recording import load_files
from pulse3D.plate_recording import PlateRecording
from stdlib_utils import get_current_file_abs_directory

PATH_OF_CURRENT_FILE = get_current_file_abs_directory()


def test_load_files__loads_zipped_folder_with_calibration_recordings_correctly():
    path = os.path.join(
        PATH_OF_CURRENT_FILE,
        "magnet_finding",
        "MA200440001__2020_02_09_190359__with_calibration_recordings__zipped_as_folder.zip",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        zf = zipfile.ZipFile(path)
        zf.extractall(path=tmpdir)
        tissue_recordings, baseline_recordings = load_files(tmpdir)

    assert len(tissue_recordings) == 24
    assert len(baseline_recordings) == 24


def test_load_files__loads_zipped_files_with_calibration_recordings_correctly():
    path = os.path.join(
        PATH_OF_CURRENT_FILE,
        "magnet_finding",
        "MA200440001__2020_02_09_190359__with_calibration_recordings__zipped_as_files.zip",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        zf = zipfile.ZipFile(path)
        zf.extractall(path=tmpdir)
        tissue_recordings, baseline_recordings = load_files(tmpdir)

    assert len(tissue_recordings) == 24
    assert len(baseline_recordings) == 24


def test_PlateRecording__uses_mean_of_baseline_by_default(mocker):
    # mock instead of spy so magnet finding alg doesn't run
    mocked_process_data = mocker.patch.object(PlateRecording, "_process_plate_data", autospec=True)

    pr = PlateRecording(
        os.path.join(
            PATH_OF_CURRENT_FILE,
            "magnet_finding",
            "MA200440001__2020_02_09_190359__with_calibration_recordings__zipped_as_folder.zip",
        )
    )

    mocked_process_data.assert_called_once_with(pr, mocker.ANY, use_mean_of_baseline=True)


def test_PlateRecording__creates_mean_of_baseline_data_correctly(mocker):
    # spy for easy access to baseline data array
    spied_mfd_from_memsic = mocker.spy(plate_recording, "calculate_magnetic_flux_density_from_memsic")
    # mock instead of spy so magnet finding alg doesn't run
    mocked_find_positions = mocker.patch.object(
        plate_recording,
        "find_magnet_positions",
        autospec=True,
        side_effect=lambda x, y: {"X": np.zeros((x.shape[-1], 24))},
    )

    pr = PlateRecording(
        os.path.join(
            PATH_OF_CURRENT_FILE,
            "magnet_finding",
            "MA200440001__2020_02_09_190359__with_calibration_recordings__zipped_as_folder.zip",
        )
    )
    raw_baseline_data = spied_mfd_from_memsic.spy_return

    actual_baseline_mean_arr = mocked_find_positions.call_args[0][1]
    assert actual_baseline_mean_arr.shape == (24, 3, 3, 1)
    for well_idx in range(actual_baseline_mean_arr.shape[0]):
        for sensor_idx in range(actual_baseline_mean_arr.shape[1]):
            for axis_idx in range(actual_baseline_mean_arr.shape[2]):
                expected_mean = np.mean(
                    raw_baseline_data[well_idx, sensor_idx, axis_idx, -BASELINE_MEAN_NUM_DATA_POINTS:]
                )
                assert actual_baseline_mean_arr[well_idx, sensor_idx, axis_idx] == expected_mean, (
                    well_idx,
                    sensor_idx,
                    axis_idx,
                )


def test_PlateRecording__passes_data_to_magnet_finding_alg_correctly__using_mean_of_baseline_data(
    mocker,
):
    # mock so slow function doesn't actually run
    mocked_get_positions = mocker.patch.object(
        magnet_finding,
        "get_positions",
        autospec=True,
        side_effect=lambda x: {"X": np.zeros((x.shape[-1], 24))},
    )

    test_zip_file_path = os.path.join(
        PATH_OF_CURRENT_FILE,
        "magnet_finding",
        "MA200440001__2020_02_09_190359__with_calibration_recordings__zipped_as_folder.zip",
    )

    # create expected input
    with tempfile.TemporaryDirectory() as tmpdir:
        zf = zipfile.ZipFile(test_zip_file_path)
        zf.extractall(path=tmpdir)
        tissue_data_memsic, baseline_data_memsic = load_h5_folder_as_array(
            os.path.join(tmpdir, "MA200440001__2020_02_09_190359")
        )
    tissue_data_mt = calculate_magnetic_flux_density_from_memsic(tissue_data_memsic)
    baseline_data_mt = calculate_magnetic_flux_density_from_memsic(baseline_data_memsic)
    baseline_data_mt_mean = np.mean(
        baseline_data_mt[:, :, :, -BASELINE_MEAN_NUM_DATA_POINTS:], axis=3
    ).reshape((24, 3, 3, 1))
    expected_input_data = tissue_data_mt - baseline_data_mt_mean

    # test alg input
    pr = PlateRecording(test_zip_file_path)
    mocked_get_positions.assert_called_once()
    np.testing.assert_array_almost_equal(mocked_get_positions.call_args[0][0], expected_input_data)
