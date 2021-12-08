# -*- coding: utf-8 -*-
from pulse3D import magnet_finding
from pulse3D import plate_recording
from pulse3D import MEMSIC_CENTER_OFFSET
from pulse3D import MEMSIC_FULL_SCALE,MEMSIC_MSB
from pulse3D import GAUSS_PER_MILLITESLA
from pulse3D import REFERENCE_SENSOR_READINGS, TIME_INDICES,TIME_OFFSETS
from pulse3D import TISSUE_SENSOR_READINGS,WELL_IDX_TO_MODULE_ID
from pulse3D import PlateRecording
from pulse3D.plate_recording import _load_files
from h5py import File
import numpy as np
import pytest

from pulse3D.transforms import calculate_force_from_displacement

from .fixtures_utils import load_h5_folder_as_array


@pytest.mark.slow
def test_get_positions__returns_expected_values():
    loaded_data = load_h5_folder_as_array("Durability_Test_11162021_data_90min")
    loaded_data_mt = (
        (loaded_data - MEMSIC_CENTER_OFFSET)
        * MEMSIC_FULL_SCALE
        / MEMSIC_MSB
        / GAUSS_PER_MILLITESLA
    )
    outputs = magnet_finding.get_positions(loaded_data_mt[:, :, :, 2:102])

    output_file = File(
        "tests/magnet_finding/magnet_finding_output_100pts.h5",
        "r",
        libver="latest",
    )

    acc = {output_name: -1 for output_name in outputs.keys()}
    for output_name, output in outputs.items():
        for decimal in range(0, 14):
            try:
                np.testing.assert_array_almost_equal(
                    output,
                    output_file[output_name],
                    decimal=decimal,
                    err_msg=f"output_name"
                )
            except AssertionError:
                acc[output_name] = decimal - 1
                break
    print(acc)
    assert all(val >= 3 for val in acc.values())


@pytest.mark.slow
def test_PlateRecording__creates_correct_displacement_and_force_data_for_beta_2_files(mocker):
    num_points_to_test = 100

    def load_files_se(*args):
        well_files = _load_files(*args)
        for well_file in well_files:
            well_file[TIME_INDICES] = well_file[TIME_INDICES][:num_points_to_test]
            well_file[TIME_OFFSETS] = well_file[TIME_OFFSETS][:, :num_points_to_test]
            well_file[TISSUE_SENSOR_READINGS] = well_file[TISSUE_SENSOR_READINGS][:, :num_points_to_test]
            well_file[REFERENCE_SENSOR_READINGS] = well_file[REFERENCE_SENSOR_READINGS][:, :num_points_to_test]
        return well_files

    mocker.patch.object(
        plate_recording,
        "_load_files",
        autospec=True,
        side_effect=load_files_se
    )

    # mock this so data doesn't actually get filtered and is easier to test
    mocked_filter = mocker.patch.object(
        magnet_finding,
        "filter_magnet_positions",
        autospec=True,
        side_effect=lambda x: x,
    )

    pr = PlateRecording("tests/magnet_finding/MA200440001__2020_02_09_190359__with_calibration_recordings.zip")
    assert mocked_filter.call_count == magnet_finding.NUM_PARAMS

    output_file = File(
        "tests/magnet_finding/magnet_finding_output_100pts__baseline_removed.h5",
        "r",
        libver="latest",
    )

    for well_idx, well_file in enumerate(pr.wells):
        # test displacement
        module_id = WELL_IDX_TO_MODULE_ID[well_idx]
        expected_displacement = np.array(
            [well_file[TIME_INDICES], output_file["X"][:, module_id - 1]]
        )
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
