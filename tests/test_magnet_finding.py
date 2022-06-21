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
from pulse3D.magnet_finding import fix_dropped_samples
from pulse3D.magnet_finding import format_well_file_data
from pulse3D.plate_recording import load_files
from pulse3D.plate_recording import PlateRecording
import pytest
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
        side_effect=lambda data, *args: {"X": np.zeros((data.shape[-1], 24))},
    )

    PlateRecording(
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
            PATH_OF_CURRENT_FILE,
            "magnet_finding",
            "MA200440001__2020_02_09_190359__with_calibration_recordings__zipped_as_folder.zip",
        )
    )
    h5_pr = PlateRecording.from_directory(
        os.path.join(
            PATH_OF_CURRENT_FILE,
            "h5",
            "v0.3.2",
        )
    )
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
            PATH_OF_CURRENT_FILE,
            "magnet_finding",
            "MA200440001__2020_02_09_190359__with_calibration_recordings__zipped_as_folder.zip",
        )
    )
    actual_plate_data = spied_fix.call_args[0][0]
    expected_plate_data = format_well_file_data(pr.wells)
    np.testing.assert_array_equal(actual_plate_data, expected_plate_data)
    np.testing.assert_array_equal(spied_mfd.call_args_list[0][0][0], spied_fix.spy_return)


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
    PlateRecording(test_zip_file_path)

    mocked_get_positions.assert_called_once()
    np.testing.assert_array_almost_equal(mocked_get_positions.call_args[0][0], expected_input_data)


@pytest.mark.parametrize(
    "path_to_recording,initial_params,flip_data",
    [
        (
            os.path.join(
                "magnet_finding",
                "MA200440001__2020_02_09_190359__with_calibration_recordings__zipped_as_folder.zip",
            ),
            {},
            False,
        ),
        (
            os.path.join("h5", "v1.1.0", "ML2022126006_Position 1 Baseline_2022_06_15_004655.zip"),
            {"X": 0, "Y": 2, "Z": -5, "REMN": 1200},
            True,
        ),
    ],
)
def test_PlateRecording__passes_initial_params_to_magnet_finding_alg_correctly__and_flips_displacement_result_correctly(
    path_to_recording, initial_params, flip_data, mocker
):
    # mock so no filtering occurs
    mocker.patch.object(magnet_finding, "filter_magnet_positions", side_effect=lambda data: data)

    def create_displacement(data):
        return np.arange(data.shape[-1] * 24).reshape((data.shape[-1], 24))

    # mock so slow function doesn't actually run
    mocked_get_positions = mocker.patch.object(
        magnet_finding,
        "get_positions",
        autospec=True,
        side_effect=lambda data, *args, **kwargs: {"X": create_displacement(data)},
    )

    test_zip_file_path = os.path.join(PATH_OF_CURRENT_FILE, path_to_recording)

    # test alg input
    pr = PlateRecording(test_zip_file_path)

    mocked_get_positions.assert_called_once()
    assert mocked_get_positions.call_args[1] == initial_params

    expected_displacement = create_displacement(mocked_get_positions.call_args[0][0])
    if flip_data:
        expected_displacement *= -1
    for well_idx, wf in enumerate(pr):
        np.testing.assert_array_almost_equal(
            wf.displacement[1], expected_displacement[:, well_idx], err_msg=f"Well {well_idx}"
        )


@pytest.mark.parametrize(
    "test_array,expected_array",
    [
        (np.array([0, 1, 2, 0, 4, 5, 0]), np.array([1, 1, 2, 3, 4, 5, 5])),
        (np.array([[[0, 1, 2, 0, 4, 5, 0]]]), np.array([[[1, 1, 2, 3, 4, 5, 5]]])),
        (np.array([[0, 1, 0, 3, 0], [5, 0, 3, 0, 1]]), np.array([[1, 1, 2, 3, 3], [5, 4, 3, 2, 1]])),
    ],
)
def test_fix_dropped_samples__makes_correct_modifications_to_input_array(test_array, expected_array):
    fixed_array = fix_dropped_samples(test_array)
    np.testing.assert_array_equal(fixed_array, expected_array)
