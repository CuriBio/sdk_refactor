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
from pulse3D.constants import CARDIAC_STIFFNESS_FACTOR
from pulse3D.constants import NUM_CHANNELS_24_WELL_PLATE
from pulse3D.magnet_finding import filter_raw_signal
from pulse3D.magnet_finding import fix_dropped_samples
from pulse3D.plate_recording import load_files
from pulse3D.plate_recording import PlateRecording
import pytest
from stdlib_utils import get_current_file_abs_directory

from ..fixtures_utils import PATH_TO_H5_FILES
from ..fixtures_utils import PATH_TO_MAGNET_FINDING_FILES

PATH_OF_CURRENT_FILE = get_current_file_abs_directory()


def test_load_files__loads_zipped_folder_with_calibration_recordings_correctly():
    path = os.path.join(
        PATH_TO_MAGNET_FINDING_FILES,
        "MA200440001__2020_02_09_190359__with_calibration_recordings__zipped_as_folder.zip",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        zf = zipfile.ZipFile(path)
        zf.extractall(path=tmpdir)
        tissue_recordings, baseline_recordings = load_files(tmpdir, CARDIAC_STIFFNESS_FACTOR)

    assert len(tissue_recordings) == 24
    assert len(baseline_recordings) == 24


def test_load_files__loads_zipped_files_with_calibration_recordings_correctly():
    path = os.path.join(
        PATH_TO_MAGNET_FINDING_FILES,
        "MA200440001__2020_02_09_190359__with_calibration_recordings__zipped_as_files.zip",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        zf = zipfile.ZipFile(path)
        zf.extractall(path=tmpdir)
        tissue_recordings, baseline_recordings = load_files(tmpdir, CARDIAC_STIFFNESS_FACTOR)

    assert len(tissue_recordings) == 24
    assert len(baseline_recordings) == 24


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
            PATH_TO_MAGNET_FINDING_FILES,
            "MA200440001__2020_02_09_190359__with_calibration_recordings__zipped_as_folder.zip",
        )
    )
    raw_baseline_data = spied_mfd_from_memsic.spy_return

    actual_baseline_mean_arr = mocked_find_positions.call_args[0][1]
    assert actual_baseline_mean_arr.shape == (NUM_CHANNELS_24_WELL_PLATE,)
    for channel_idx in range(actual_baseline_mean_arr.shape[0]):
        expected_mean = np.mean(raw_baseline_data[channel_idx, -BASELINE_MEAN_NUM_DATA_POINTS:])
        assert actual_baseline_mean_arr[channel_idx] == expected_mean, channel_idx


def test_PlateRecording__runs_mag_finding_alg_by_default(mocker):
    # mock instead of spy so magnet finding alg doesn't run
    mocked_process_data = mocker.patch.object(PlateRecording, "_process_plate_data", autospec=True)

    pr = PlateRecording(
        os.path.join(
            PATH_TO_MAGNET_FINDING_FILES,
            "MA200440001__2020_02_09_190359__with_calibration_recordings__zipped_as_folder.zip",
        )
    )

    mocked_process_data.assert_called_once_with(pr, mocker.ANY)


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
        PATH_TO_MAGNET_FINDING_FILES,
        "MA200440001__2020_02_09_190359__with_calibration_recordings__zipped_as_folder.zip",
    )

    # create expected input
    with tempfile.TemporaryDirectory() as tmpdir:
        zf = zipfile.ZipFile(test_zip_file_path)
        zf.extractall(path=tmpdir)
        tissue_data_memsic, baseline_data_memsic = load_h5_folder_as_array(
            os.path.join(
                tmpdir, "MA200440001__2020_02_09_190359__with_calibration_recordings__zipped_as_folder"
            )
        )
    tissue_data_mt = calculate_magnetic_flux_density_from_memsic(tissue_data_memsic)
    filtered_tissue_data_mt = filter_raw_signal(tissue_data_mt)
    baseline_data_mt = calculate_magnetic_flux_density_from_memsic(baseline_data_memsic)
    baseline_data_mt_mean = np.mean(baseline_data_mt[:, -BASELINE_MEAN_NUM_DATA_POINTS:], axis=1)
    expected_input_data = (filtered_tissue_data_mt.T - baseline_data_mt_mean).T

    # test alg input
    PlateRecording(test_zip_file_path)

    mocked_get_positions.assert_called_once()
    np.testing.assert_array_almost_equal(mocked_get_positions.call_args[0][0], expected_input_data)


@pytest.mark.parametrize(
    "path_to_recording,initial_params,flip_data",
    [
        (
            os.path.join(
                PATH_TO_MAGNET_FINDING_FILES,
                "MA200440001__2020_02_09_190359__with_calibration_recordings__zipped_as_folder.zip",
            ),
            {},
            False,
        ),
        (
            os.path.join(
                PATH_TO_H5_FILES, "v1.1.0", "ML2022126006_Position 1 Baseline_2022_06_15_004655.zip"
            ),
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
