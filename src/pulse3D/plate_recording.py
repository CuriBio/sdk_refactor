import datetime
import os
import tempfile
import uuid
import zipfile

from typing import Any, Optional
from nptyping import NDArray
from semver import VersionInfo

import h5py
import numpy as np

from .constants import *
from .transforms import create_filter
from .transforms import apply_sensitivity_calibration
from .transforms import noise_cancellation
from .transforms import apply_empty_plate_calibration
from .transforms import apply_noise_filtering
from .transforms import calculate_voltage_from_gmr
from .transforms import calculate_displacement_from_voltage
from .transforms import calculate_force_from_displacement
from .transforms import calculate_voltage_from_gmr
from .transforms import calculate_displacement_from_voltage
from .transforms import calculate_force_from_displacement


class WellFile:
    def __init__(self, file_path: str, sampling_period=None):
        if file_path.endswith('.h5'):
            self.file = h5py.File(file_path, 'r')
            self.file_name = self.file.filename

            self.is_force_data = True
            self.is_magnetic_data = True

            self.attrs = {attr: self.file.attrs[attr] for attr in list(self.file.attrs)}
        elif file_path.endswith('.xlsx'):
            self.file_name = os.path.basename(file_path)
            self.attrs = {}
            self.is_force_data = False
            self.is_magnetic_data = False
            raise Exception('Not Implemented') #TODO

        self.version = self[FILE_FORMAT_VERSION_METADATA_KEY]

        # setup noise filter
        self.tissue_sampling_period = sampling_period if sampling_period else self[TISSUE_SAMPLING_PERIOD_UUID]
        self.noise_filter_uuid = TSP_TO_DEFAULT_FILTER_UUID[self.tissue_sampling_period] if self.is_magnetic_data else None
        self.filter_coefficients = create_filter(self.noise_filter_uuid, self.tissue_sampling_period)

        # extract datetimes
        self[UTC_BEGINNING_DATA_ACQUISTION_UUID] = self._extract_datetime(UTC_BEGINNING_DATA_ACQUISTION_UUID)
        self[UTC_BEGINNING_RECORDING_UUID] = self._extract_datetime(UTC_BEGINNING_RECORDING_UUID)
        self[UTC_FIRST_TISSUE_DATA_POINT_UUID] = self._extract_datetime(UTC_FIRST_TISSUE_DATA_POINT_UUID)
        self[UTC_FIRST_REF_DATA_POINT_UUID] = self._extract_datetime(UTC_FIRST_REF_DATA_POINT_UUID)

        is_untrimmed = self.get(IS_FILE_ORIGINAL_UNTRIMMED_UUID, True)
        time_trimmed = None if is_untrimmed else self[TRIMMED_TIME_FROM_ORIGINAL_START_UUID]

        # load sensor data
        self[TISSUE_SENSOR_READINGS] = self._load_reading(TISSUE_SENSOR_READINGS, time_trimmed)
        self[REFERENCE_SENSOR_READINGS] = self._load_reading(REFERENCE_SENSOR_READINGS, time_trimmed)

        self._load_magnetic_data()


    def _load_magnetic_data(self):
        adj_raw_tissue_reading = self[TISSUE_SENSOR_READINGS].copy()
        f = MICROSECONDS_PER_CENTIMILLISECOND if self.is_magnetic_data else MICRO_TO_BASE_CONVERSION
        adj_raw_tissue_reading[0] *= f

        # magnetic data is flipped
        if self.is_magnetic_data:
            adj_raw_tissue_reading[1] *= -1

        self.raw_tissue_magnetic_data: NDArray[(2, Any), int] = adj_raw_tissue_reading
        self.raw_reference_magnetic_data: NDArray[(2, Any), int] = self[REFERENCE_SENSOR_READINGS].copy()

        self.sensitivity_calibrated_tissue_gmr: NDArray[(2, Any), int] = \
            apply_sensitivity_calibration(self.raw_tissue_magnetic_data)

        self.sensitivity_calibrated_reference_gmr: NDArray[(2, Any), int] = \
            apply_sensitivity_calibration(self.raw_reference_magnetic_data)

        self.noise_cancelled_magnetic_data: NDArray[(2, Any), int] = noise_cancellation(
            self.sensitivity_calibrated_tissue_gmr,
            self.sensitivity_calibrated_reference_gmr,
        )

        self.fully_calibrated_magnetic_data: NDArray[(2, Any), int] = \
            apply_empty_plate_calibration(self.noise_cancelled_magnetic_data)

        if self.noise_filter_uuid is None: 
            self.noise_filtered_magnetic_data: NDArray[(2, Any), int] = self.fully_calibrated_magnetic_data
        else:
            self.noise_filtered_magnetic_data: NDArray[(2, Any), int] = apply_noise_filtering(
                self.fully_calibrated_magnetic_data,
                self.filter_coefficients,
        )

        # TODO
        # self.compressed_magnetic_data: NDArray[(2, Any), int] = compress_filtered_gmr(self.noise_filtered_magnetic_data)
        # self.compressed_voltage: NDArray[(2, Any), np.float32] = calculate_voltage_from_gmr(self.compressed_magnetic_data)
        # self.compressed_displacement: NDArray[(2, Any), np.float32] = calculate_displacement_from_voltage(self.compressed_voltage)
        # self.compressed_force: NDArray[(2, Any), np.float32] = calculate_force_from_displacement(self.compressed_displacement)

        self.voltage: NDArray[(2, Any), np.float32] = calculate_voltage_from_gmr(self.noise_filtered_magnetic_data)
        self.displacement: NDArray[(2, Any), np.float32] = calculate_displacement_from_voltage(self.voltage)
        self.force: NDArray[(2, Any), np.float32] = calculate_force_from_displacement(self.displacement)


    def get(self, key, default):
        try:
            return self[key]
        except:
            return default

    def __contains__(self, key):
        key = str(key) if isinstance(key, uuid.UUID) else key
        return key in self.attrs

    def __setitem__(self, key, newvalue):
        key = str(key) if isinstance(key, uuid.UUID) else key
        self.attrs[key] = newvalue


    def __getitem__(self, i):
        i = str(i) if isinstance(i, uuid.UUID) else i
        return self.attrs[i]


    def _extract_datetime(self, metadata_uuid: uuid.UUID) -> datetime.datetime:
        if self.version.split(".") < VersionInfo.parse("0.2.1"):
            if metadata_uuid == UTC_BEGINNING_RECORDING_UUID:
                """
                The use of this proxy value is justified by the fact that there is a 15 second delay
                between when data is recorded and when the GUI displays it, and because the GUI will
                send the timestamp of when the recording button is pressed.
                """
                acquisition_timestamp_str = self[UTC_BEGINNING_DATA_ACQUISTION_UUID]

                begin_recording = datetime.datetime.strptime(
                    acquisition_timestamp_str, DATETIME_STR_FORMAT
                ).replace(tzinfo=datetime.timezone.utc) + datetime.timedelta(seconds=15)

                return begin_recording
            if metadata_uuid == UTC_FIRST_TISSUE_DATA_POINT_UUID:
                """
                Early file versions did not include this metadata under a UUID, so we have to use this
                string identifier instead
                """
                metadata_name = "UTC Timestamp of Beginning of Recorded Tissue Sensor Data"
                timestamp_str = self[metadata_name]

                return datetime.datetime.strptime(timestamp_str, DATETIME_STR_FORMAT).replace(
                    tzinfo=datetime.timezone.utc
                )
            if metadata_uuid == UTC_FIRST_REF_DATA_POINT_UUID:
                """
                Early file versions did not include this metadata under a UUID, so we have to use this
                string identifier instead
                """
                timestamp_str = self["UTC Timestamp of Beginning of Recorded Reference Sensor Data"]

                return datetime.datetime.strptime(timestamp_str, DATETIME_STR_FORMAT).replace(
                    tzinfo=datetime.timezone.utc
                )

        timestamp_str = self[metadata_uuid]
        return datetime.datetime.strptime(timestamp_str, DATETIME_STR_FORMAT).replace(
            tzinfo=datetime.timezone.utc
        )


    def _load_reading(self, reading_type: str, time_trimmed) -> NDArray[(Any, Any), int]:
        recording_start_index = self[START_RECORDING_TIME_INDEX_UUID]
        beginning_data_acquisition_ts = self[UTC_BEGINNING_DATA_ACQUISTION_UUID]

        if reading_type == REFERENCE_SENSOR_READINGS:
            initial_timestamp = self[UTC_FIRST_REF_DATA_POINT_UUID]
            sampling_period = self[REF_SAMPLING_PERIOD_UUID]
        else:
            initial_timestamp = self[UTC_FIRST_TISSUE_DATA_POINT_UUID]
            sampling_period = self[TISSUE_SAMPLING_PERIOD_UUID]

        recording_start_index_useconds = int(recording_start_index) * MICROSECONDS_PER_CENTIMILLISECOND
        timestamp_of_start_index = beginning_data_acquisition_ts + datetime.timedelta(
            microseconds=recording_start_index_useconds
        )

        time_delta = initial_timestamp - timestamp_of_start_index
        time_delta_centimilliseconds = int(time_delta / datetime.timedelta(
            microseconds=MICROSECONDS_PER_CENTIMILLISECOND
        ))

        time_step = int(sampling_period / MICROSECONDS_PER_CENTIMILLISECOND)

        # adding `[:]` loads the data as a numpy array giving us more flexibility of multi-dimensional arrays
        data = self.file[reading_type][:]
        if len(data.shape) == 1:
            data = data.reshape(1, data.shape[0])

        # fmt: off
        # black reformatted this into a very ugly few lines of code
        times = np.mgrid[: data.shape[1],] * time_step
        # fmt: on

        if time_trimmed:
            new_times = times + time_delta_centimilliseconds
            start_index = _find_start_index(time_trimmed, new_times)
            time_delta_centimilliseconds = int(new_times[start_index])

        return np.concatenate(  # pylint: disable=unexpected-keyword-arg # Tanner (5/6/21): unsure why pylint thinks dtype is an unexpected kwarg for np.concatenate
            (times + time_delta_centimilliseconds, data), dtype=np.int32
        )


class PlateRecording:
    def __init__(self, path):
        self.path = path
        self.wells = []
        self._iter = 0

        if self.path.endswith('.zip'):
            zf = zipfile.ZipFile(self.path)
            files = [f for f in zf.namelist() if f.endswith('.h5')]
            self.wells = [None] * len(files)

            with tempfile.TemporaryDirectory() as tempdir:
                zf.extractall(path=tempdir, members=files)

                for f in files:
                    well_file = WellFile(os.path.join(tempdir, f))
                    self.wells[well_file[WELL_INDEX_UUID]] = well_file

    def __iter__(self):
        self._iter = 0
        return self

    def __next__(self):
        if self._iter < len(self.wells):
            value = self.wells[self._iter]
            self._iter += 1
            return value
        else:
            raise StopIteration


def _find_start_index(from_start: int, old_data: NDArray[(1, Any), int]) -> int:
    start_index, time_from_start = 0, 0

    while start_index + 1 < len(old_data) and from_start >= time_from_start:
        time_from_start = old_data[start_index + 1] - old_data[0]
        start_index += 1

    return start_index - 1 #loop iterates 1 past the desired index, so subtract 1

