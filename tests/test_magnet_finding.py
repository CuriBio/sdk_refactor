# -*- coding: utf-8 -*-
import os
import tempfile
import zipfile

from h5py import File
import numpy as np
from pulse3D import magnet_finding
from pulse3D import plate_recording
from pulse3D.constants import BASELINE_MEAN_NUM_DATA_POINTS
from pulse3D.constants import GAUSS_PER_MILLITESLA
from pulse3D.constants import MEMSIC_CENTER_OFFSET
from pulse3D.constants import MEMSIC_FULL_SCALE
from pulse3D.constants import MEMSIC_MSB
from pulse3D.constants import REFERENCE_SENSOR_READINGS
from pulse3D.constants import TIME_INDICES
from pulse3D.constants import TIME_OFFSETS
from pulse3D.constants import TISSUE_SENSOR_READINGS
from pulse3D.constants import WELL_IDX_TO_MODULE_ID
from pulse3D.plate_recording import load_files
from pulse3D.plate_recording import PlateRecording
from pulse3D.transforms import calculate_force_from_displacement
import pytest
from stdlib_utils import get_current_file_abs_directory

from .fixtures_utils import load_h5_folder_as_array

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


@pytest.mark.slow
def test_get_positions__returns_expected_values():
    loaded_data = load_h5_folder_as_array("Durability_Test_11162021_data_90min")
    loaded_data_mt = (
        (loaded_data - MEMSIC_CENTER_OFFSET) * MEMSIC_FULL_SCALE / MEMSIC_MSB / GAUSS_PER_MILLITESLA
    )
    outputs = magnet_finding.get_positions(loaded_data_mt[:, :, :, 2:102])

    output_file = File(
        os.path.join(
            PATH_OF_CURRENT_FILE,
            "magnet_finding",
            "magnet_finding_output_100pts.h5",
        ),
        "r",
        libver="latest",
    )

    acc = {output_name: -1 for output_name in outputs.keys()}
    for output_name, output in outputs.items():
        for decimal in range(0, 14):
            try:
                np.testing.assert_array_almost_equal(
                    output, output_file[output_name], decimal=decimal, err_msg=f"output_name"
                )
            except AssertionError:
                acc[output_name] = decimal - 1
                break
    print(acc)
    assert all(val >= 3 for val in acc.values())


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


@pytest.mark.slow
def test_PlateRecording__creates_correct_displacement_and_force_data_for_beta_2_files__using_elementwise_removal_of_baseline_data(
    mocker,
):
    num_points_to_test = 100

    def load_files_se(*args):
        tissue_well_files, baseline_well_files = load_files(*args)
        all_well_files = set(tissue_well_files) | set(baseline_well_files)
        for well_file in all_well_files:
            well_file[TIME_INDICES] = well_file[TIME_INDICES][:num_points_to_test]
            well_file[TIME_OFFSETS] = well_file[TIME_OFFSETS][:, :num_points_to_test]
            well_file[TISSUE_SENSOR_READINGS] = well_file[TISSUE_SENSOR_READINGS][:, :num_points_to_test]
            well_file[REFERENCE_SENSOR_READINGS] = well_file[REFERENCE_SENSOR_READINGS][
                :, :num_points_to_test
            ]
        return tissue_well_files, baseline_well_files

    mocker.patch.object(plate_recording, "load_files", autospec=True, side_effect=load_files_se)

    # mock this so data doesn't actually get filtered and is easier to test
    mocked_filter = mocker.patch.object(
        magnet_finding,
        "filter_magnet_positions",
        autospec=True,
        side_effect=lambda x: x,
    )

    pr = PlateRecording(
        os.path.join(
            PATH_OF_CURRENT_FILE,
            "magnet_finding",
            "MA200440001__2020_02_09_190359__with_calibration_recordings__zipped_as_folder.zip",
        ),
        use_mean_of_baseline=False,
    )
    assert mocked_filter.call_count == magnet_finding.NUM_PARAMS

    output_file = File(
        os.path.join(
            PATH_OF_CURRENT_FILE,
            "magnet_finding",
            "magnet_finding_output_100pts__baseline_removed.h5",
        ),
        "r",
        libver="latest",
    )

    for well_idx, well_file in enumerate(pr.wells):
        # test displacement
        module_id = WELL_IDX_TO_MODULE_ID[well_idx]
        expected_displacement = np.array([well_file[TIME_INDICES], output_file["X"][:, module_id - 1]])
        # Tanner (12/7/21): iterating through different decimal precision here since the precision is different for each well, but
        np.testing.assert_array_almost_equal(
            well_file.displacement,
            expected_displacement,
            decimal=5,
            err_msg=f"{well_idx}",
        )
        # test force
        expected_force = calculate_force_from_displacement(well_file.displacement)
        np.testing.assert_array_almost_equal(
            well_file.force,
            expected_force,
            err_msg=f"{well_idx}",
        )
