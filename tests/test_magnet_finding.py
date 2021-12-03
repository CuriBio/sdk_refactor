# -*- coding: utf-8 -*-
from pulse3D import magnet_finding
from pulse3D import MEMSIC_CENTER_OFFSET
from pulse3D import MEMSIC_FULL_SCALE,MEMSIC_MSB
from pulse3D import GAUSS_PER_MILLITESLA
from pulse3D import REFERENCE_SENSOR_READINGS
from pulse3D import TISSUE_SENSOR_READINGS
# from pulse3D import WellFile
# from pulse3D import MantarrayH5FileCreator
from h5py import File
import numpy as np

TIME_INDICES = "time_indices"
TIME_OFFSETS = "time_offsets"


# def test_data_validity():
#     processed_data = magnet_finding.processData("tests/datasets/magnet_finding/Durability_Test_11162021_data_90min")
#     loaded_data = load_h5_files_as_array("Durability_Test_11162021_data_90min")
#     loaded_data_mt = (
#         (loaded_data - MEMSIC_CENTER_OFFSET)
#         * MEMSIC_FULL_SCALE
#         / MEMSIC_MSB
#         / GAUSS_PER_MILLITESLA
#     )
#     np.testing.assert_array_almost_equal(
#         processed_data, loaded_data_mt[:, :, :, 2:]
#     )


def test_100_pts():
    # Cloud9 test times:
    # start:                                              32.98s
    # move "manta" calculation out of meas_field:         27.17s


    print("\n")

    loaded_data = load_h5_files_as_array("Durability_Test_11162021_data_90min")
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

    output_names = ["X", "Y", "Z", "THETA", "PHI", "REMN"]
    acc = [-1] * len(output_names)
    for i, output in enumerate(outputs):
        output_name = output_names[i]
        for decimal in range(0, 14):
            try:
                np.testing.assert_array_almost_equal(
                    output,
                    output_file[output_name],
                    decimal=decimal,
                    err_msg=f"{i}, {output_name}"
                )
            except AssertionError:
                acc[i] = decimal - 1
                break
    print(acc)
    assert all(val >= 3 for val in acc)


def load_h5_files_as_array(recording_name):
    # TODO Tanner (12/3/21): This should be in src
    plate_data_array = None
    for module_id in range(1, 25):
        file_path = f"tests/magnet_finding/{recording_name}/{recording_name}__module_{module_id}.h5"

        with File(file_path, "r") as well_file:
            tissue_data = well_file[TISSUE_SENSOR_READINGS][:]
        if plate_data_array is None:
            num_samples = tissue_data.shape[-1]
            plate_data_array = np.empty((24, 3, 3, num_samples))
        reshaped_data = tissue_data.reshape((3, 3, num_samples))
        plate_data_array[module_id - 1, :, :, :] = reshaped_data
    return plate_data_array


# def test_100_pts_og():
#     processed_data = magnet_finding.processData("tests/datasets/magnet_finding/Durability_Test_11162021_data_90min")
#     outputs = magnet_finding.getPositions(processed_data[:, :, :, :100])
#     save_outputs(outputs)


# def convert_to_h5(processed_data, processed_time_indices):
#     for module_id in range(1, 25):
#         print(module_id)
#         time_indices = processed_time_indices[module_id - 1, :, :]
#         data = processed_data[module_id - 1, :, :, :]

#         num_sensors_active = 3
#         num_axes_active = 3
#         num_channels_enabled = num_sensors_active  * num_axes_active
#         max_data_len = data.shape[-1]
#         data_shape = (num_channels_enabled, max_data_len)
#         maxshape = (num_channels_enabled, max_data_len)
#         data_dtype = "uint16"

#         this_file = MantarrayH5FileCreator(
#             f"tests/datasets/magnet_finding/Durability_Test_11162021_data_90min/Durability_Test_11162021_data_90min__module_{module_id}.h5",
#             file_format_version="1.0.0"
#         )
#         this_file.create_dataset(
#             TIME_INDICES,
#             (max_data_len,),
#             maxshape=(max_data_len,),
#             dtype="uint64",
#             chunks=True,
#         )
#         this_file.create_dataset(
#             TIME_OFFSETS,
#             (num_sensors_active, max_data_len),
#             maxshape=(num_sensors_active, max_data_len),
#             dtype="uint16",
#             chunks=True,
#         )
#         for idx in range(time_indices.shape[-1]):
#             paired_time_indices = time_indices[:, idx]
#             max_time_index = max(paired_time_indices)
#             time_offsets = max_time_index - paired_time_indices
#             this_file[TIME_INDICES][idx] = max_time_index
#             this_file[TIME_OFFSETS][:, idx] = time_offsets
#         this_file.create_dataset(
#             REFERENCE_SENSOR_READINGS,
#             data_shape,
#             maxshape=maxshape,
#             dtype=data_dtype,
#             chunks=True,
#         )
#         this_file[REFERENCE_SENSOR_READINGS][:] = np.zeros(data_shape)
#         this_file.create_dataset(
#             TISSUE_SENSOR_READINGS,
#             data_shape,
#             maxshape=maxshape,
#             dtype=data_dtype,
#             chunks=True,
#         )
#         for sensor_idx in range(num_sensors_active):
#             for axis_idx in range(num_axes_active):
#                 channel_idx = sensor_idx * 3 + axis_idx
#                 this_file[TISSUE_SENSOR_READINGS][channel_idx, :] = data[sensor_idx, axis_idx, :]
#         this_file.close()


def save_outputs(outputs):
    output_file = File(
        "tests/datasets/magnet_finding/magnet_finding_output_100pts.h5",
        "w",
        libver="latest",
        userblock_size=512,
    )

    output_names = ["X", "Y", "Z", "THETA", "PHI", "REMN"]
    for i, name in enumerate(output_names):
        output_file.create_dataset(
            name,
            outputs[i].shape,
            maxshape=outputs[i].shape,
            dtype="float64",
            chunks=True,
        )
        output_file[name][:] = outputs[i]