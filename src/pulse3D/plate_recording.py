# -*- coding: utf-8 -*-
import copy
import datetime
import glob
import json
import logging
import math
import os
import tempfile
from typing import Any
from typing import List
from typing import Optional
from typing import Union
import uuid
import zipfile

import h5py
from mantarray_magnet_finding.utils import calculate_magnetic_flux_density_from_memsic
from nptyping import NDArray
import numpy as np
from openpyxl import load_workbook
from openpyxl.worksheet.worksheet import Worksheet
import pandas as pd
from scipy import interpolate
from semver import VersionInfo
from xlsxwriter.utility import xl_cell_to_rowcol

from .compression_cy import compress_filtered_magnetic_data
from .constants import *
from .magnet_finding import find_magnet_positions
from .magnet_finding import fix_dropped_samples
from .magnet_finding import format_well_file_data
from .transforms import apply_empty_plate_calibration
from .transforms import apply_noise_filtering
from .transforms import apply_sensitivity_calibration
from .transforms import calculate_displacement_from_voltage
from .transforms import calculate_force_from_displacement
from .transforms import calculate_voltage_from_gmr
from .transforms import create_filter
from .transforms import noise_cancellation
from .utils import get_experiment_id
from .utils import get_stiffness_factor
from .utils import truncate
from .utils import truncate_float

log = logging.getLogger(__name__)


class MantarrayH5FileCreator(h5py.File):
    """Creates an H5 file with the basic format/layout."""

    def __init__(
        self,
        file_name: str,
        file_format_version: str = CURRENT_BETA2_HDF5_FILE_FORMAT_VERSION,
    ) -> None:
        super().__init__(
            file_name,
            "w",
            # tried to specify this ('earliest', 'v110') to be more backward compatible
            # but it didn't work for unknown reasons (gave error when trying to set swmr_mode=True)
            libver="latest",
            userblock_size=512,  # minimum size is 512 bytes
        )

        self.attrs[FILE_FORMAT_VERSION_METADATA_KEY] = file_format_version


class WellFile:
    def __init__(
        self,
        file_path: str,
        sampling_period: Optional[Union[int, float]] = None,
        # TODO unit test the stiffness factor, auto and override
        stiffness_factor: Optional[int] = None,
        has_inverted_post_magnet: bool = False,
    ):
        self.displacement: NDArray[(2, Any), np.float64]
        self.force: NDArray[(2, Any), np.float64]

        if stiffness_factor not in (*POST_STIFFNESS_OVERRIDE_OPTIONS, None):
            raise ValueError(
                f"Invalid Post Stiffness {stiffness_factor}, must be in {POST_STIFFNESS_OVERRIDE_OPTIONS}"
            )

        self.has_inverted_post_magnet = has_inverted_post_magnet

        self.stiffness_override = stiffness_factor is not None

        if file_path.endswith(".h5"):
            with h5py.File(file_path, "r") as h5_file:
                self.file_name = os.path.basename(h5_file.filename)
                self.is_force_data = True
                self.is_magnetic_data = True

                self.attrs = {attr: h5_file.attrs[attr] for attr in list(h5_file.attrs)}
                self.version = self[FILE_FORMAT_VERSION_METADATA_KEY]

                if self.stiffness_override:
                    self.stiffness_factor = stiffness_factor
                elif not self.get(IS_CALIBRATION_FILE_UUID, False):
                    # earlier versions of files do not have the IS_CALIBRATION_FILE_UUID in their metadata
                    experiment_id = get_experiment_id(self[PLATE_BARCODE_UUID])
                    self.stiffness_factor = get_stiffness_factor(experiment_id, self[WELL_INDEX_UUID])
                else:
                    # calibration recordings do not have an associated barcode or post stiffness since they
                    # are creatd when a plate is not even on the istrument, so just set the stiffness factor to 1
                    self.stiffness_factor = CALIBRATION_STIFFNESS_FACTOR  # TODO might want to make this None

                # extract datetimes
                for uuid_ in (
                    UTC_BEGINNING_RECORDING_UUID,
                    UTC_BEGINNING_DATA_ACQUISTION_UUID,
                    UTC_FIRST_TISSUE_DATA_POINT_UUID,
                ):
                    self[uuid_] = self._extract_datetime(uuid_)

                self.tissue_sampling_period = (
                    sampling_period if sampling_period else self[TISSUE_SAMPLING_PERIOD_UUID]
                )

                if self.version < VersionInfo.parse("1.0.0"):
                    # Ref data not yet added to Beta 2 files
                    self[UTC_FIRST_REF_DATA_POINT_UUID] = self._extract_datetime(
                        UTC_FIRST_REF_DATA_POINT_UUID
                    )
                    # setup noise filter, should probably add beta 2 file support for this
                    self.noise_filter_uuid = (
                        TSP_TO_DEFAULT_FILTER_UUID[self.tissue_sampling_period]
                        if self.is_magnetic_data
                        else None
                    )
                    self.filter_coefficients = (
                        create_filter(self.noise_filter_uuid, self.tissue_sampling_period)
                        if self.noise_filter_uuid
                        else None
                    )

                    is_untrimmed = self.get(IS_FILE_ORIGINAL_UNTRIMMED_UUID, True)
                    time_trimmed = None if is_untrimmed else self.attrs[TRIMMED_TIME_FROM_ORIGINAL_START_UUID]

                    # load sensor data. This is only possible to do here for Beta 1 data files
                    self[TISSUE_SENSOR_READINGS] = self._load_reading(
                        h5_file, TISSUE_SENSOR_READINGS, time_trimmed
                    )
                    self[REFERENCE_SENSOR_READINGS] = self._load_reading(
                        h5_file, REFERENCE_SENSOR_READINGS, time_trimmed
                    )
                    self._load_magnetic_data()
                else:
                    if self.has_inverted_post_magnet:
                        raise ValueError("V1 algorithm will automatically fix this issue")

                    self.noise_filter_uuid = None
                    self.filter_coefficients = None
                    self.is_magnetic_data = False

                    for dataset in (
                        TIME_INDICES,
                        TIME_OFFSETS,
                        TISSUE_SENSOR_READINGS,
                        REFERENCE_SENSOR_READINGS,
                    ):
                        self[dataset] = h5_file[dataset][:]

        elif file_path.endswith(".xlsx"):
            self._excel_sheet = _get_single_sheet(file_path)
            self.file_name = os.path.basename(file_path)
            self.attrs = {k: v for k, v in _load_optical_file_attrs(self._excel_sheet).items()}
            self.version = self[FILE_FORMAT_VERSION_METADATA_KEY]
            self.is_magnetic_data = False
            self.is_force_data = (
                "y" in str(_get_excel_metadata_value(self._excel_sheet, TWITCHES_POINT_UP_UUID)).lower()
            )

            self.noise_filter_uuid = (
                TSP_TO_DEFAULT_FILTER_UUID[self.tissue_sampling_period] if self.is_magnetic_data else None
            )
            self.filter_coefficients = (
                create_filter(self.noise_filter_uuid, self.tissue_sampling_period)
                if self.noise_filter_uuid
                else None
            )

            self.stiffness_factor = stiffness_factor if stiffness_factor else CARDIAC_STIFFNESS_FACTOR

            self._load_magnetic_data()

    def _load_magnetic_data(self):
        adj_raw_tissue_reading = self[TISSUE_SENSOR_READINGS].copy()
        f = MICROSECONDS_PER_CENTIMILLISECOND if self.is_magnetic_data else MICRO_TO_BASE_CONVERSION
        adj_raw_tissue_reading[0] *= f

        # magnetic data is flipped
        if self.is_magnetic_data:
            adj_raw_tissue_reading[1] *= -1

        # some posts have a flipped magnet
        if self.has_inverted_post_magnet:
            adj_raw_tissue_reading[1] *= -1

        self.raw_tissue_magnetic_data: NDArray[(2, Any), int] = adj_raw_tissue_reading
        self.raw_reference_magnetic_data: NDArray[(2, Any), int] = self[REFERENCE_SENSOR_READINGS].copy()

        self.sensitivity_calibrated_tissue_gmr: NDArray[(2, Any), int] = apply_sensitivity_calibration(
            self.raw_tissue_magnetic_data
        )

        self.sensitivity_calibrated_reference_gmr: NDArray[(2, Any), int] = apply_sensitivity_calibration(
            self.raw_reference_magnetic_data
        )

        self.noise_cancelled_magnetic_data: NDArray[(2, Any), int] = noise_cancellation(
            self.sensitivity_calibrated_tissue_gmr, self.sensitivity_calibrated_reference_gmr
        )

        self.fully_calibrated_magnetic_data: NDArray[(2, Any), int] = apply_empty_plate_calibration(
            self.noise_cancelled_magnetic_data
        )

        if self.noise_filter_uuid is None:
            self.noise_filtered_magnetic_data: NDArray[(2, Any), int] = self.fully_calibrated_magnetic_data
        else:
            self.noise_filtered_magnetic_data: NDArray[(2, Any), int] = apply_noise_filtering(
                self.fully_calibrated_magnetic_data, self.filter_coefficients
            )

        self.compressed_magnetic_data: NDArray[(2, Any), int] = compress_filtered_magnetic_data(
            self.noise_filtered_magnetic_data
        )
        self.compressed_voltage: NDArray[(2, Any), np.float32] = calculate_voltage_from_gmr(
            self.compressed_magnetic_data
        )
        self.compressed_displacement: NDArray[(2, Any), np.float32] = calculate_displacement_from_voltage(
            self.compressed_voltage
        )
        self.compressed_force: NDArray[(2, Any), np.float32] = calculate_force_from_displacement(
            self.compressed_displacement, stiffness_factor=self.stiffness_factor, in_mm=False
        )

        self.voltage: NDArray[(2, Any), np.float32] = calculate_voltage_from_gmr(
            self.noise_filtered_magnetic_data
        )
        self.displacement: NDArray[(2, Any), np.float64] = calculate_displacement_from_voltage(self.voltage)
        self.force: NDArray[(2, Any), np.float64] = calculate_force_from_displacement(
            self.displacement, stiffness_factor=self.stiffness_factor, in_mm=False
        )

    def get(self, key, default):
        try:
            return self[key]
        except Exception:
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
                """The use of this proxy value is justified by the fact that
                there is a 15 second delay between when data is recorded and
                when the GUI displays it, and because the GUI will send the
                timestamp of when the recording button is pressed."""
                acquisition_timestamp_str = self[UTC_BEGINNING_DATA_ACQUISTION_UUID]

                begin_recording = datetime.datetime.strptime(
                    acquisition_timestamp_str, DATETIME_STR_FORMAT
                ).replace(tzinfo=datetime.timezone.utc) + datetime.timedelta(seconds=15)

                return begin_recording
            if metadata_uuid == UTC_FIRST_TISSUE_DATA_POINT_UUID:
                """Early file versions did not include this metadata under a
                UUID, so we have to use this string identifier instead."""
                metadata_name = "UTC Timestamp of Beginning of Recorded Tissue Sensor Data"
                timestamp_str = self[metadata_name]

                return datetime.datetime.strptime(timestamp_str, DATETIME_STR_FORMAT).replace(
                    tzinfo=datetime.timezone.utc
                )
            if metadata_uuid == UTC_FIRST_REF_DATA_POINT_UUID:
                """Early file versions did not include this metadata under a
                UUID, so we have to use this string identifier instead."""
                timestamp_str = self["UTC Timestamp of Beginning of Recorded Reference Sensor Data"]

                return datetime.datetime.strptime(timestamp_str, DATETIME_STR_FORMAT).replace(
                    tzinfo=datetime.timezone.utc
                )

        timestamp_str = self[metadata_uuid]
        return datetime.datetime.strptime(timestamp_str, DATETIME_STR_FORMAT).replace(
            tzinfo=datetime.timezone.utc
        )

    def _load_reading(self, h5_file, reading_type: str, time_trimmed) -> NDArray[(Any, Any), int]:
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
        time_delta_centimilliseconds = int(
            time_delta / datetime.timedelta(microseconds=MICROSECONDS_PER_CENTIMILLISECOND)
        )

        time_step = int(sampling_period / MICROSECONDS_PER_CENTIMILLISECOND)

        # adding `[:]` loads the data as a numpy array giving us more flexibility of multi-dimensional arrays
        data = h5_file[reading_type][:]
        if len(data.shape) == 1:
            data = data.reshape(1, data.shape[0])

        # fmt: off
        # black reformatted this into a few very ugly lines of code
        times = np.mgrid[: data.shape[1],] * time_step
        # fmt: on

        if time_trimmed:
            new_times = times + time_delta_centimilliseconds
            start_index = _find_start_index(time_trimmed, new_times)
            time_delta_centimilliseconds = int(new_times[start_index])

        return np.concatenate((times + time_delta_centimilliseconds, data), dtype=np.int32)


class PlateRecording:
    def __init__(
        self,
        path,
        use_mean_of_baseline=True,
        force_df: pd.DataFrame = None,
        start_time: Union[float, int] = 0,
        end_time: Union[float, int] = np.inf,
        # TODO unit test the stiffness factor, auto and override
        stiffness_factor: Optional[int] = None,
        inverted_post_magnet_wells: Optional[List[str]] = None,
    ):
        self.path = path
        self.wells = []
        self._iter = 0
        self.is_optical_recording = False
        self.start_time = start_time * MICRO_TO_BASE_CONVERSION
        self.end_time = end_time * MICRO_TO_BASE_CONVERSION

        if self.path.endswith(".zip"):
            with tempfile.TemporaryDirectory() as tmpdir:
                zf = zipfile.ZipFile(path)
                zf.extractall(path=tmpdir)
                self.wells, calibration_recordings = load_files(
                    tmpdir, stiffness_factor, inverted_post_magnet_wells
                )
        elif self.path.endswith(".xlsx"):  # optical file
            self.is_optical_recording = True
            well_file = WellFile(self.path, stiffness_factor=stiffness_factor)
            self.wells = [None] * (well_file[WELL_INDEX_UUID] + 1)
            self.wells[well_file[WELL_INDEX_UUID]] = well_file
        else:  # .h5 files
            self.wells, calibration_recordings = load_files(
                self.path, stiffness_factor, inverted_post_magnet_wells
            )

        if not self.wells:
            # might not be any well files in the path
            # TODO should probably raise an error here
            return

        # currently file versions 1.0.0 and above must have all their data processed together
        if not self.is_optical_recording and self.wells[0].version >= VersionInfo.parse("1.0.0"):
            if force_df is None:
                self._process_plate_data(calibration_recordings, use_mean_of_baseline=use_mean_of_baseline)
            else:
                self._load_dataframe(force_df)

    def _process_plate_data(self, calibration_recordings, use_mean_of_baseline):
        if not all(isinstance(well_file, WellFile) for well_file in self.wells) or len(self.wells) != 24:
            raise NotImplementedError("All 24 wells must have a recording file present")

        if (
            not all(isinstance(well_file, WellFile) for well_file in calibration_recordings)
            or len(calibration_recordings) != 24
        ):
            raise NotImplementedError("All 24 wells must have a calibration file present")

        initial_magnet_finding_params = json.loads(
            self.wells[0].get(INITIAL_MAGNET_FINDING_PARAMS_UUID, r"{}")
        )

        # load data
        plate_data_array = format_well_file_data(self.wells)
        fixed_plate_data_array = fix_dropped_samples(plate_data_array)
        plate_data_array_mt = calculate_magnetic_flux_density_from_memsic(fixed_plate_data_array)
        baseline_data = format_well_file_data(calibration_recordings)
        baseline_data_mt = calculate_magnetic_flux_density_from_memsic(baseline_data)

        # create baseline data array
        if use_mean_of_baseline:
            baseline_data_mt = np.mean(
                baseline_data_mt[:, :, :, -BASELINE_MEAN_NUM_DATA_POINTS:], axis=3
            ).reshape((24, 3, 3, 1))
        else:
            # extend baseline data (if necessary) so that it has at least as many samples as the tissue data
            num_samples_in_recording = plate_data_array_mt.shape[-1]
            num_samples_in_baseline = baseline_data_mt.shape[-1]
            num_times_to_duplicate = math.ceil(num_samples_in_recording / num_samples_in_baseline)
            # truncate so baseline data has exactly as many samples as the tissue data
            baseline_data_mt = np.tile(baseline_data_mt, num_times_to_duplicate)[
                :, :, :, :num_samples_in_recording
            ]

        # pass data into magnet finding alg
        log.info("Estimating magnet positions")
        estimated_magnet_positions = find_magnet_positions(
            plate_data_array_mt, baseline_data_mt, initial_magnet_finding_params
        )

        flip_data = self.wells[0].version >= VersionInfo.parse("1.1.0")

        # create displacement and force arrays for each WellFile
        log.info("Create diplacement and force data for each well")
        for well_idx in range(24):
            well_file = self.wells[well_idx]
            x = estimated_magnet_positions["X"][:, well_idx]
            if flip_data:
                x *= -1

            # have time indices start at 0
            adjusted_time_indices = well_file[TIME_INDICES] - well_file[TIME_INDICES][0]
            start_idx, end_idx = truncate(
                source_series=adjusted_time_indices, lower_bound=self.start_time, upper_bound=self.end_time
            )

            well_file.displacement = np.array([adjusted_time_indices, x])[:, start_idx : end_idx + 1]
            well_file.force = calculate_force_from_displacement(
                well_file.displacement, stiffness_factor=well_file.stiffness_factor
            )

    def _load_dataframe(self, df: pd.DataFrame) -> None:
        """Add time and force data to well files in PR.

        Args:
            df: pd.Dataframe existing time force data to be added
        """
        time = df["Time (s)"].values
        for well in self.wells:
            well_name = well[WELL_NAME_UUID]
            well_data = np.vstack((time, df[f"{well_name}__raw"])).astype(np.float64)
            well.force = well_data

    def write_time_force_csv(self, output_dir: str):
        # get recording name
        recording_name = os.path.splitext(os.path.basename(self.path))[0]
        output_path = os.path.join(output_dir, f"{recording_name}.csv")

        # set indexes to time points
        truncated_time_sec = [
            truncate_float(ms / MICRO_TO_BASE_CONVERSION, 2) for ms in self.wells[0].force[0]
        ]

        force_data = dict({"Time (s)": pd.Series(truncated_time_sec)})

        for well in self.wells:
            well_name = well.get(WELL_NAME_UUID, None)
            force_data[well_name] = pd.Series(well.force[1])

        time_force_df = pd.DataFrame(force_data)
        time_force_df.to_csv(output_path, index=False)

        return time_force_df, output_path

    def to_dataframe(self, from_dataframe: bool = False) -> pd.DataFrame:
        """Creates DataFrame from PlateRecording with all the data
        interpolated, normalized, and scaled. The returned dataframe contains
        one column for time in ms and one column for each well.

        The dataframe returned by this method can be used in calls to peak_detector by selecting the
        'time' column and the well column that peak detection should be run on after transposing the
        data, e.g.

        >>> df = to_dataframe()
        >>> peak_detector(df[['time', 'A1']].transpose().values)

        The dataframe needs to be transposed before converting to NDArray because peak_detector
        wants an ndarray of shape (2,N) while dataframe[['time', 'A1']] is in shape (N,2)
        """
        data = {}

        # get first valid well and set interpolation period
        first_well = [pw for pw in self.wells if pw][0]

        if self.is_optical_recording:
            interp_period = first_well[INTERPOLATION_VALUE_UUID]
        else:
            interp_period = INTERPOLATED_DATA_PERIOD_US

        if from_dataframe:
            time_steps = copy.deepcopy(first_well.force[0])
        else:
            max_time = max([w.force[0][-1] for w in self.wells if w])
            time_steps = np.arange(interp_period, max_time + interp_period, interp_period)

        data["Time (s)"] = pd.Series(time_steps)

        for i, w in enumerate(self.wells):

            if not from_dataframe:
                start_idx, end_idx = truncate(
                    source_series=time_steps,
                    lower_bound=w.force[0][0],
                    upper_bound=w.force[0][-1],
                )

                interp_fn = interpolate.interp1d(w.force[0, :], w.force[1, :])
                interp_force = interp_fn(time_steps[start_idx : end_idx + 1])
            else:
                interp_force = copy.deepcopy(w.force[1, 1:])

            data[f"{w.get(WELL_NAME_UUID, str(i))}__raw"] = pd.Series(w.force[1, :])

            min_value = min(interp_force)
            interp_force -= min_value
            interp_force *= MICRO_TO_BASE_CONVERSION

            data[w.get(WELL_NAME_UUID, str(i))] = pd.Series(interp_force)

        df = pd.DataFrame(data).dropna()
        return df

    @staticmethod
    def from_dataframe(path, df: pd.DataFrame):
        # multi zip files
        # only allowed for one recording at a time assuming a user would only ever pass a dataframe to one recording
        log.info(f"Loading recording from file {os.path.basename(path)}")
        yield PlateRecording(path, force_df=df)

    @staticmethod
    def from_directory(path, **kwargs):
        # multi zip files
        for zf in glob.glob(os.path.join(path, "*.zip"), recursive=True):
            zip_file = zipfile.ZipFile(zf)
            # check if this is a zip of optical files
            if ".xlsx" in zip_file.namelist()[0]:
                with tempfile.TemporaryDirectory() as tmpdir:
                    zip_file.extractall(path=tmpdir)
                    for optical_file in os.scandir(tmpdir):
                        yield PlateRecording(optical_file.path, **kwargs)
                return
            log.info(f"Loading recording from file {zf}")
            yield PlateRecording(zf, **kwargs)

        # multi optical files
        for of in glob.glob(os.path.join(path, "*.xlsx"), recursive=True):
            log.info(f"Loading optical data from file {of}")
            yield PlateRecording(of, **kwargs)

        # directory of .h5 files
        for dir in glob.glob(os.path.join(path, "*"), recursive=True):
            if glob.glob(os.path.join(dir, "*.h5"), recursive=True):
                yield PlateRecording(dir, **kwargs)

    def __iter__(self):
        self._iter = 0
        return self

    def __next__(self):
        while self._iter < len(self.wells):
            value = self.wells[self._iter]
            self._iter += 1
            if not value:
                continue
            return value
        else:
            raise StopIteration


# helpers
def load_files(
    path: str, stiffness_factor: Optional[int], inverted_post_magnet_wells: Optional[List[str]] = None
):
    if not inverted_post_magnet_wells:
        inverted_post_magnet_wells = []

    h5_files = glob.glob(os.path.join(path, "**", "*.h5"), recursive=True)

    recording_files = [f for f in h5_files if "Calibration" not in f]
    calibration_files = [f for f in h5_files if "Calibration" in f]

    tissue_well_files = [None] * len(recording_files)
    baseline_well_files = [None] * len(calibration_files)

    for f in recording_files:
        log.info(f"Loading data from {os.path.basename(f)}")
        well_name = os.path.splitext(f)[0].split("__")[1]
        well_file = WellFile(
            f,
            stiffness_factor=stiffness_factor,
            has_inverted_post_magnet=well_name in inverted_post_magnet_wells,
        )
        tissue_well_files[well_file[WELL_INDEX_UUID]] = well_file  # type: ignore

    for f in calibration_files:
        log.info(f"Loading calibration data from {os.path.basename(f)}")
        well_file = WellFile(f, stiffness_factor=stiffness_factor)
        baseline_well_files[well_file[WELL_INDEX_UUID]] = well_file  # type: ignore

    return tissue_well_files, baseline_well_files


def _find_start_index(from_start: int, old_data: NDArray[(1, Any), int]) -> int:
    start_index, time_from_start = 0, 0

    while start_index + 1 < len(old_data) and from_start >= time_from_start:
        time_from_start = old_data[start_index + 1] - old_data[0]
        start_index += 1

    return start_index - 1  # loop iterates 1 past the desired index, so subtract 1


def _get_col_as_array(sheet: Worksheet, zero_based_row: int, zero_based_col: int) -> NDArray[(2, Any), float]:
    col_array = []
    result = _get_cell_value(sheet, zero_based_row, zero_based_col)
    zero_based_row += 1
    while result:
        col_array.append(float(result))
        result = _get_cell_value(sheet, zero_based_row, zero_based_col)
        zero_based_row += 1
    return np.array(col_array)


def _get_single_sheet(file_name: str) -> Any:
    work_book = load_workbook(file_name)
    return work_book[work_book.sheetnames[0]]


def _get_cell_value(sheet: Worksheet, zero_based_row: int, zero_based_col: int) -> Optional[str]:
    result = sheet.cell(row=zero_based_row + 1, column=zero_based_col + 1).value
    if result is None:
        return result
    return str(result)


def _get_excel_metadata_value(sheet: Worksheet, metadata_uuid: uuid.UUID) -> Optional[str]:
    """Return a user-entered metadata value."""
    metadata_description = METADATA_UUID_DESCRIPTIONS[metadata_uuid]
    cell_name = EXCEL_OPTICAL_METADATA_CELLS.get(metadata_uuid, None)
    if cell_name is None:
        raise NotImplementedError(
            f"Metadata value for {metadata_description} is not contained in excel files of well data"
        )
    row, col = xl_cell_to_rowcol(cell_name)
    result = _get_cell_value(sheet, row, col)
    return result


def _load_optical_file_attrs(sheet: Worksheet):
    raw_tissue_reading = np.array((_get_col_as_array(sheet, 1, 0), _get_col_as_array(sheet, 1, 1)))

    value = _get_excel_metadata_value(sheet, TISSUE_SAMPLING_PERIOD_UUID)
    if value is None:
        raise NotImplementedError("Tissue Sampling Period should not be None here")
    sampling_period = int(round(1 / float(value), 6) * MICRO_TO_BASE_CONVERSION)

    interpolation_value_str = _get_excel_metadata_value(sheet, INTERPOLATION_VALUE_UUID)
    interpolation_value = (
        float(sampling_period)
        if interpolation_value_str is None
        else float(interpolation_value_str) * MICRO_TO_BASE_CONVERSION
    )

    begin_recording = _get_excel_metadata_value(sheet, UTC_BEGINNING_RECORDING_UUID)
    begin_recording = datetime.datetime.strptime(begin_recording, "%Y-%m-%d %H:%M:%S")  # type: ignore

    twenty_four_well = LabwareDefinition(row_count=4, column_count=6)
    well_name = _get_excel_metadata_value(sheet, WELL_NAME_UUID)

    attrs = {
        FILE_FORMAT_VERSION_METADATA_KEY: "N/A",
        TISSUE_SENSOR_READINGS: raw_tissue_reading,
        REFERENCE_SENSOR_READINGS: np.zeros(raw_tissue_reading.shape),
        str(INTERPOLATION_VALUE_UUID): interpolation_value,
        str(TISSUE_SAMPLING_PERIOD_UUID): sampling_period,
        str(UTC_BEGINNING_RECORDING_UUID): begin_recording,
        str(MANTARRAY_SERIAL_NUMBER_UUID): _get_excel_metadata_value(sheet, MANTARRAY_SERIAL_NUMBER_UUID),
        str(PLATE_BARCODE_UUID): _get_excel_metadata_value(sheet, PLATE_BARCODE_UUID),
        str(WELL_NAME_UUID): well_name,
        str(WELL_INDEX_UUID): twenty_four_well.get_well_index_from_well_name(well_name),
    }

    return attrs
