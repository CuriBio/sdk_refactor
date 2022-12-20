# -*- coding: utf-8 -*-
"""Constants for the Mantarray File Manager."""
import uuid

from immutabledict import immutabledict
from labware_domain_models import LabwareDefinition
from mantarray_magnet_finding.constants import NUM_CHANNELS as NUM_CHANNELS_PER_WELL

try:
    from importlib import metadata
except ImportError:  # pragma: no cover
    import importlib_metadata as metadata  # type: ignore
PACKAGE_VERSION = metadata.version("pulse3D")

TWENTY_FOUR_WELL_PLATE = LabwareDefinition(row_count=4, column_count=6)


NUM_CHANNELS_24_WELL_PLATE = 24 * NUM_CHANNELS_PER_WELL

MIN_SUPPORTED_FILE_VERSION = "0.1.1"
CURRENT_BETA1_HDF5_FILE_FORMAT_VERSION = "0.4.2"
CURRENT_BETA2_HDF5_FILE_FORMAT_VERSION = "1.0.0"
FILE_FORMAT_VERSION_METADATA_KEY = "File Format Version"
FILE_MIGRATION_PATHS = immutabledict({"0.3.1": "0.4.1", "0.4.1": "0.4.2"})

# Eli (1/19/21): H5 files can't store the concept of `None` in their metadata, so using this value to denote that a particular piece of metadata is not available (i.e. after migrating to a newer file format version)
NOT_APPLICABLE_H5_METADATA = uuid.UUID("59d92e00-99d5-4460-9a28-5a1a0fe9aecf")

HARDWARE_TEST_RECORDING_UUID = uuid.UUID("a2e76058-08cd-475d-a55d-31d401c3cb34")
UTC_BEGINNING_DATA_ACQUISTION_UUID = uuid.UUID("98c67f22-013b-421a-831b-0ea55df4651e")
START_RECORDING_TIME_INDEX_UUID = uuid.UUID("e41422b3-c903-48fd-9856-46ff56a6534c")
UTC_BEGINNING_RECORDING_UUID = uuid.UUID("d2449271-0e84-4b45-a28b-8deab390b7c2")
UTC_FIRST_TISSUE_DATA_POINT_UUID = uuid.UUID("b32fb8cb-ebf8-4378-a2c0-f53a27bc77cc")
UTC_FIRST_REF_DATA_POINT_UUID = uuid.UUID("7cc07b2b-4146-4374-b8f3-1c4d40ff0cf7")
CUSTOMER_ACCOUNT_ID_UUID = uuid.UUID("4927c810-fbf4-406f-a848-eba5308576e6")
USER_ACCOUNT_ID_UUID = uuid.UUID("7282cf00-2b6e-4202-9d9e-db0c73c3a71f")
SOFTWARE_BUILD_NUMBER_UUID = uuid.UUID("b4db8436-10a4-4359-932d-aa80e6de5c76")
SOFTWARE_RELEASE_VERSION_UUID = uuid.UUID("432fc3c1-051b-4604-bc3d-cc0d0bd75368")
MAIN_FIRMWARE_VERSION_UUID = uuid.UUID("faa48a0c-0155-4234-afbf-5e5dbaa59537")
SLEEP_FIRMWARE_VERSION_UUID = uuid.UUID("3a816076-90e4-4437-9929-dc910724a49d")
XEM_SERIAL_NUMBER_UUID = uuid.UUID("e5f5b134-60c7-4881-a531-33aa0edba540")
MANTARRAY_NICKNAME_UUID = uuid.UUID("0cdec9bb-d2b4-4c5b-9dd5-6a49766c5ed4")
MANTARRAY_SERIAL_NUMBER_UUID = uuid.UUID("83720d36-b941-4d85-9b39-1d817799edd6")
REFERENCE_VOLTAGE_UUID = uuid.UUID("0b3f3f56-0cc7-45f0-b748-9b9de480cba8")
WELL_NAME_UUID = uuid.UUID("6d78f3b9-135a-4195-b014-e74dee70387b")
WELL_ROW_UUID = uuid.UUID("da82fe73-16dd-456a-ac05-0b70fb7e0161")
WELL_COLUMN_UUID = uuid.UUID("7af25a0a-8253-4d32-98c4-3c2ca0d83906")
WELL_INDEX_UUID = uuid.UUID("cd89f639-1e36-4a13-a5ed-7fec6205f779")
TOTAL_WELL_COUNT_UUID = uuid.UUID("7ca73e1c-9555-4eca-8281-3f844b5606dc")
REF_SAMPLING_PERIOD_UUID = uuid.UUID("48aa034d-8775-453f-b135-75a983d6b553")
TISSUE_SAMPLING_PERIOD_UUID = uuid.UUID("f629083a-3724-4100-8ece-c03e637ac19c")
ADC_GAIN_SETTING_UUID = uuid.UUID("a3c3bb32-9b92-4da1-8ed8-6c09f9c816f8")
ADC_TISSUE_OFFSET_UUID = uuid.UUID("41069860-159f-49f2-a59d-401783c1ecb4")
ADC_REF_OFFSET_UUID = uuid.UUID("dc10066c-abf2-42b6-9b94-5e52d1ea9bfc")
PLATE_BARCODE_UUID = uuid.UUID("cf60afef-a9f0-4bc3-89e9-c665c6bb6941")
STIM_BARCODE_UUID = uuid.UUID("6fa67db1-c8b9-4937-b93f-6fe8bdc7e6d7")
BACKEND_LOG_UUID = uuid.UUID("87533deb-2495-4430-bce7-12fdfc99158e")
COMPUTER_NAME_HASH_UUID = uuid.UUID("fefd0675-35c2-45f6-855a-9500ad3f100d")
PLATE_BARCODE_IS_FROM_SCANNER_UUID = uuid.UUID("7d026e86-da70-4464-9181-dc0ce2d47bd1")
STIM_BARCODE_IS_FROM_SCANNER_UUID = uuid.UUID("6e5a4b3e-f766-4638-80f7-d95c417c0fc2")
IS_FILE_ORIGINAL_UNTRIMMED_UUID = uuid.UUID("52231a24-97a3-497a-917c-86c780d9993f")
TRIMMED_TIME_FROM_ORIGINAL_START_UUID = uuid.UUID("371996e6-5e2d-4183-a5cf-06de7058210a")
TRIMMED_TIME_FROM_ORIGINAL_END_UUID = uuid.UUID("55f6770d-c369-42ce-a437-5ed89c3cb1f8")
ORIGINAL_FILE_VERSION_UUID = uuid.UUID("cd1b4063-4a87-4a57-bc12-923ff4890844")
UTC_TIMESTAMP_OF_FILE_VERSION_MIGRATION_UUID = uuid.UUID("399b2148-09d4-418b-a132-e37df2721938")
FILE_VERSION_PRIOR_TO_MIGRATION_UUID = uuid.UUID("11b4945b-3cf3-4f67-8bee-7abc3c449756")
TWITCHES_POINT_UP_UUID = uuid.UUID("97f69f56-f1c6-4c50-8590-7332570ed3c5")
INTERPOLATION_VALUE_UUID = uuid.UUID("466d0131-06b7-4f0f-ba1e-062a771cb280")
BOOTUP_COUNTER_UUID = uuid.UUID("b9ccc724-a39d-429a-be6d-3fd29be5037d")
TOTAL_WORKING_HOURS_UUID = uuid.UUID("f8108718-2fa0-40ce-a51a-8478e5edd4b8")
TAMPER_FLAG_UUID = uuid.UUID("68d0147f-9a84-4423-9c50-228da16ba895")
PCB_SERIAL_NUMBER_UUID = uuid.UUID("5103f995-19d2-4880-8a2e-2ce9080cd2f5")
MAGNETOMETER_CONFIGURATION_UUID = uuid.UUID("921121e9-4191-4536-bedd-03186fa1e117")
UTC_BEGINNING_STIMULATION_UUID = uuid.UUID("4b310594-ded4-45fd-a1b4-b829aceb416c")
STIMULATION_PROTOCOL_UUID = uuid.UUID("ede638ce-544e-427a-b1d9-c40784d7c82d")
IS_CALIBRATION_FILE_UUID = uuid.UUID("9a6f90eb-fe34-423b-bfed-fb441d6d9e5f")
CHANNEL_FIRMWARE_VERSION_UUID = uuid.UUID("d9694cfe-824c-41f8-915e-91e41ce7af32")
BOOT_FLAGS_UUID = uuid.UUID("762f6715-ffcd-4e8d-b707-638dd5777841")
INITIAL_MAGNET_FINDING_PARAMS_UUID = uuid.UUID("da5f2f6d-6874-4e53-be10-90c4bfbd3d45")
METADATA_UUID_DESCRIPTIONS = immutabledict(
    {
        # General values
        HARDWARE_TEST_RECORDING_UUID: "Is Hardware Test Recording",
        START_RECORDING_TIME_INDEX_UUID: "Timepoint of Beginning of Recording",
        UTC_BEGINNING_DATA_ACQUISTION_UUID: "UTC Timestamp of Beginning of Data Acquisition",
        UTC_BEGINNING_RECORDING_UUID: "UTC Timestamp of Beginning of Recording",
        UTC_FIRST_TISSUE_DATA_POINT_UUID: "UTC Timestamp of Beginning of Recorded Tissue Sensor Data",
        UTC_FIRST_REF_DATA_POINT_UUID: "UTC Timestamp of Beginning of Recorded Reference Sensor Data",
        CUSTOMER_ACCOUNT_ID_UUID: "Customer Account ID",
        USER_ACCOUNT_ID_UUID: "User Account ID",
        SOFTWARE_BUILD_NUMBER_UUID: "Software Build Number",
        SOFTWARE_RELEASE_VERSION_UUID: "Software Release Version",
        MAIN_FIRMWARE_VERSION_UUID: "Firmware Version (Main Controller)",
        SLEEP_FIRMWARE_VERSION_UUID: "Firmware Version (Sleep Mode)",
        MANTARRAY_NICKNAME_UUID: "Mantarray Nickname",
        MANTARRAY_SERIAL_NUMBER_UUID: "Mantarray Serial Number",
        REFERENCE_VOLTAGE_UUID: "Reference Voltage",
        WELL_NAME_UUID: "Well Name",
        WELL_ROW_UUID: "Well Row (zero-based)",
        WELL_COLUMN_UUID: "Well Column (zero-based)",
        WELL_INDEX_UUID: "Well Index (zero-based)",
        TOTAL_WELL_COUNT_UUID: "Total Wells in Plate",
        REF_SAMPLING_PERIOD_UUID: "Reference Sensor Sampling Period (microseconds)",
        TISSUE_SAMPLING_PERIOD_UUID: "Tissue Sensor Sampling Period (microseconds)",
        ADC_GAIN_SETTING_UUID: "ADC Gain Setting",
        ADC_TISSUE_OFFSET_UUID: "ADC Tissue Sensor Offset",
        ADC_REF_OFFSET_UUID: "ADC Reference Sensor Offset",
        PLATE_BARCODE_UUID: "Plate Barcode",
        BACKEND_LOG_UUID: "Backend log file identifier",
        COMPUTER_NAME_HASH_UUID: "SHA512 digest of computer name",
        PLATE_BARCODE_IS_FROM_SCANNER_UUID: "Is this plate barcode obtained from the scanner",
        IS_FILE_ORIGINAL_UNTRIMMED_UUID: "Is this an original file straight from the instrument and untrimmed",
        TRIMMED_TIME_FROM_ORIGINAL_START_UUID: "Number of centimilliseconds that has been trimmed off the beginning of when the original data started",
        TRIMMED_TIME_FROM_ORIGINAL_END_UUID: "Number of centimilliseconds that has been trimmed off the end of when the original data ended",
        ORIGINAL_FILE_VERSION_UUID: "The original version of the file when recorded, prior to any migrations to newer versions/formats.",
        UTC_TIMESTAMP_OF_FILE_VERSION_MIGRATION_UUID: "Timestamp when this file was migrated from an earlier version.",
        FILE_VERSION_PRIOR_TO_MIGRATION_UUID: "File format version that this file was migrated from",
        # Beta 1 specific values
        XEM_SERIAL_NUMBER_UUID: "XEM Serial Number",
        # Beta 2 specific values
        BOOTUP_COUNTER_UUID: "The number of times this Mantarray Instrument has booted up",
        TOTAL_WORKING_HOURS_UUID: "The total number of hours this Mantarray Instrument has been powered on and running",
        TAMPER_FLAG_UUID: "Is it suspected the internals of the Mantarray enclosure have been tampered with",
        PCB_SERIAL_NUMBER_UUID: "The serial number of the Mantarray PCB",
        MAGNETOMETER_CONFIGURATION_UUID: "The state (on/off) of the board's magnetometers",
        UTC_BEGINNING_STIMULATION_UUID: "UTC Timestamp of Beginning of Stimulation",
        STIMULATION_PROTOCOL_UUID: "The stimulation protocol that was running on this well during recording. Empty string if stimulation was not active",
        STIM_BARCODE_UUID: "Stim Lid Barcode",
        STIM_BARCODE_IS_FROM_SCANNER_UUID: "Is this stim lid barcode obtained from the scanner",
        IS_CALIBRATION_FILE_UUID: "Is this file a calibration (empty plate) recording",
        CHANNEL_FIRMWARE_VERSION_UUID: "Firmware Version (Channel Controller)",
        BOOT_FLAGS_UUID: "Hardware/firmware flags present on device bootup",
        INITIAL_MAGNET_FINDING_PARAMS_UUID: "JSON string of the initial magnet finding params that should be used in Pulse3D",
        # Optical file values
        TWITCHES_POINT_UP_UUID: "Flag indicating whether or not the twitches in the data point up or not",
        INTERPOLATION_VALUE_UUID: "Desired value for optical well data interpolation",
    }
)

DATETIME_STR_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
MILLI_TO_BASE_CONVERSION = int(1e3)
CENTIMILLISECONDS_PER_SECOND = int(1e5)
MICRO_TO_BASE_CONVERSION = int(1e6)
MICROSECONDS_PER_CENTIMILLISECOND = 10


TISSUE_SENSOR_READINGS = "tissue_sensor_readings"
REFERENCE_SENSOR_READINGS = "reference_sensor_readings"
STIMULATION_READINGS = "stimulation_readings"
TIME_INDICES = "time_indices"
TIME_OFFSETS = "time_offsets"


TWITCH_PERIOD_UUID = uuid.UUID("6e0cd81c-7861-4c49-ba14-87b2739d65fb")
TWITCH_FREQUENCY_UUID = uuid.UUID("472d0707-ff87-4198-9374-c28900bb216c")
AMPLITUDE_UUID = uuid.UUID("89cf1105-a015-434f-b527-4169b9400e26")
AUC_UUID = uuid.UUID("e7b9a6e4-c43d-4e8b-af7e-51742e252030")
WIDTH_UUID = uuid.UUID("c4c60d55-017a-4783-9600-f19606de26f3")
WIDTH_VALUE_UUID = uuid.UUID("05041f4e-c77d-42d9-a2ae-8902f912e9ac")
WIDTH_RISING_COORDS_UUID = uuid.UUID("2a16acb6-4df7-4064-9d47-5d27ea7a98ad")
WIDTH_FALLING_COORDS_UUID = uuid.UUID("26e5637d-42c9-4060-aa5d-52209b349c84")
RELAXATION_VELOCITY_UUID = uuid.UUID("0fcc0dc3-f9aa-4f1b-91b3-e5b5924279a9")
CONTRACTION_VELOCITY_UUID = uuid.UUID("73961e7c-17ec-42b0-b503-a23195ec249c")
IRREGULARITY_INTERVAL_UUID = uuid.UUID("61046076-66b9-4b8b-bfec-1e00603743c0")
FRACTION_MAX_UUID = uuid.UUID("8fe142e2-2504-4c9e-b3dc-817b24c7447e")
TIME_DIFFERENCE_UUID = uuid.UUID("1363817a-b1fb-468e-9f1c-ec54fce72dfe")
TIME_VALUE_UUID = uuid.UUID("32f5ce6b-e311-4434-8a2a-c2b6bbd81ee6")
RELAXATION_TIME_UUID = uuid.UUID("0ad56cd1-7bcc-4b57-8076-14366d7f3c6a")
CONTRACTION_TIME_UUID = uuid.UUID("33b5b0a8-f197-46ef-a451-a254e530757b")
BASELINE_TO_PEAK_UUID = uuid.UUID("03ce2d30-3580-4129-9913-2fc2e35eddb7")
PEAK_TO_BASELINE_UUID = uuid.UUID("1ac2589d-4713-41c0-8dd0-1e6c98600e37")

ALL_METRICS = (
    TWITCH_PERIOD_UUID,
    FRACTION_MAX_UUID,
    AMPLITUDE_UUID,
    AUC_UUID,
    TWITCH_FREQUENCY_UUID,
    CONTRACTION_VELOCITY_UUID,
    RELAXATION_VELOCITY_UUID,
    IRREGULARITY_INTERVAL_UUID,
    WIDTH_UUID,
    RELAXATION_TIME_UUID,
    PEAK_TO_BASELINE_UUID,
    BASELINE_TO_PEAK_UUID,
    CONTRACTION_TIME_UUID,
)


DEFAULT_TWITCH_WIDTHS = (10, 50, 90)
DEFAULT_BASELINE_WIDTHS = (10, 90)
DEFAULT_TWITCH_WIDTH_PERCENTS = tuple(range(10, 95, 5))
DEFAULT_PROMINENCE_FACTORS = (6, 6)
DEFAULT_WIDTH_FACTORS = (7, 7)

PRIOR_PEAK_INDEX_UUID = uuid.UUID("80df90dc-21f8-4cad-a164-89436909b30a")
PRIOR_VALLEY_INDEX_UUID = uuid.UUID("72ba9466-c203-41b6-ac30-337b4a17a124")
SUBSEQUENT_PEAK_INDEX_UUID = uuid.UUID("7e37325b-6681-4623-b192-39f154350f36")
SUBSEQUENT_VALLEY_INDEX_UUID = uuid.UUID("fd47ba6b-ee4d-4674-9a89-56e0db7f3d97")

BESSEL_BANDPASS_UUID = uuid.UUID("0ecf0e52-0a29-453f-a6ff-46f5ec3ae783")
BESSEL_LOWPASS_10_UUID = uuid.UUID("7d64cac3-b841-4912-b734-c0cf20a81e7a")
BESSEL_LOWPASS_30_UUID = uuid.UUID("eee66c75-4dc4-4eb4-8d48-6c608bf28d91")
BUTTERWORTH_LOWPASS_30_UUID = uuid.UUID("de8d8cef-65bf-4119-ada7-bdecbbaa897a")

# General magnetic field to force conversion factor. Obtained 03/09/2021 by Kevin Gray
# Valid as of 11/19/21
MILLIMETERS_PER_MILLITESLA = 23.25
# Valid as of 09/30/22
NEWTONS_PER_MILLIMETER = 0.000159

CALIBRATION_STIFFNESS_FACTOR = 1
CARDIAC_STIFFNESS_FACTOR = 1
SKM_STIFFNESS_FACTOR = 12
ROW_LABEL_TO_VARIABLE_STIFFNESS_FACTOR = immutabledict({"A": 12, "B": 9, "C": 6, "D": 3})


# Beta 1 GMR to magnetic field conversion values. Valid as of 11/19/21
MILLIVOLTS_PER_MILLITESLA = 1073.6  # Obtained 03/09/2021 by Kevin Gray
MIDSCALE_CODE = 0x800000
RAW_TO_SIGNED_CONVERSION_VALUE = 2**23  # subtract this value from raw hardware data
REFERENCE_VOLTAGE = 2.5
ADC_GAIN = 2

# Beta 2 Memsic to magnetic field conversion factors. Valid as of 11/19/21
MEMSIC_CENTER_OFFSET = 2**15
MEMSIC_MSB = 2**16
MEMSIC_FULL_SCALE = 16
GAUSS_PER_MILLITESLA = 10


MIN_NUMBER_PEAKS = 1
MIN_NUMBER_VALLEYS = 1


MIN_EXPERIMENT_ID = 0
MAX_CARDIAC_EXPERIMENT_ID = 99
MAX_SKM_EXPERIMENT_ID = 199
MAX_VARIABLE_EXPERIMENT_ID = 299
MAX_EXPERIMENT_ID = 999

CARDIAC_STIFFNESS_LABEL = "Cardiac (1x)"
SKM_STIFFNESS_LABEL = "Skeletal Muscle (12x)"
VARIABLE_STIFFNESS_LABEL = "Variable (Row A: 12x, Row B: 9x, Row C: 6x, Row D: 3x)"

POST_STIFFNESS_LABEL_TO_FACTOR = immutabledict(
    {
        CARDIAC_STIFFNESS_LABEL: CARDIAC_STIFFNESS_FACTOR,
        SKM_STIFFNESS_LABEL: SKM_STIFFNESS_FACTOR,
        VARIABLE_STIFFNESS_LABEL: ROW_LABEL_TO_VARIABLE_STIFFNESS_FACTOR,
    }
)

POST_STIFFNESS_OVERRIDE_OPTIONS = (CARDIAC_STIFFNESS_FACTOR, SKM_STIFFNESS_FACTOR)


METADATA_EXCEL_SHEET_NAME = "metadata"
METADATA_RECORDING_ROW_START = 0
METADATA_INSTRUMENT_ROW_START = METADATA_RECORDING_ROW_START + 4
METADATA_OUTPUT_FILE_ROW_START = METADATA_INSTRUMENT_ROW_START + 6

CONTINUOUS_WAVEFORM_SHEET_NAME = "continuous-waveforms"
AGGREGATE_METRICS_SHEET_NAME = "aggregate-metrics"
PER_TWITCH_METRICS_SHEET_NAME = "per-twitch-metrics"
NUMBER_OF_PER_TWITCH_METRICS = 45
SNAPSHOT_CHART_SHEET_NAME = "continuous-waveform-snapshots"
FULL_CHART_SHEET_NAME = "full-continuous-waveform-plots"
TWITCH_FREQUENCIES_CHART_SHEET_NAME = "twitch-frequencies-plots"
FORCE_FREQUENCY_RELATIONSHIP_SHEET = "force-frequency-relationship"

INTERPOLATED_DATA_PERIOD_SECONDS = 1 / 100
INTERPOLATED_DATA_PERIOD_US = int(INTERPOLATED_DATA_PERIOD_SECONDS * MICRO_TO_BASE_CONVERSION)
# Tissue Sampling Period (µs) to default Pipeline Filter UUID
TSP_TO_DEFAULT_FILTER_UUID = {9600: BESSEL_LOWPASS_10_UUID, 1600: BUTTERWORTH_LOWPASS_30_UUID}

DEFAULT_CELL_WIDTH = 64
DEFAULT_CELL_HEIGHT = 20

CHART_ALPHA = 60  # for full/snapshots -- num pixels between left figure edge and plot area
CHART_GAMMA = 150  # for full/snapshots -- num pixels between right figure edge and plot area
CHART_PIXELS_PER_SECOND = 35  # for full/snapshots -- number of pixels per second
CHART_MAXIMUM_SNAPSHOT_LENGTH = 10

CHART_HEIGHT_CELLS = 15
CHART_HEIGHT = DEFAULT_CELL_HEIGHT * CHART_HEIGHT_CELLS

STIM_CHART_HEIGHT_CELLS = CHART_HEIGHT_CELLS - 2
STIM_CHART_HEIGHT = DEFAULT_CELL_HEIGHT * STIM_CHART_HEIGHT_CELLS

CHART_FIXED_WIDTH_CELLS = 8
CHART_FIXED_WIDTH = DEFAULT_CELL_WIDTH * CHART_FIXED_WIDTH_CELLS

PEAK_VALLEY_COLUMN_START = 100
STIM_DATA_COLUMN_START = 200
SECONDS_PER_CELL = 2.5

CALCULATED_METRIC_DISPLAY_NAMES = {
    TWITCH_PERIOD_UUID: "Twitch Period (seconds)",
    TWITCH_FREQUENCY_UUID: "Twitch Frequency (Hz)",
    AMPLITUDE_UUID: "Active Twitch Force (μN)",
    FRACTION_MAX_UUID: "Fraction of Maximum Active Twitch Force (μN)",
    AUC_UUID: "Area Under Curve (μN * second)",
    CONTRACTION_VELOCITY_UUID: "Twitch Contraction Velocity (μN/second)",
    RELAXATION_VELOCITY_UUID: "Twitch Relaxation Velocity (μN/second)",
    IRREGULARITY_INTERVAL_UUID: "Twitch Interval Irregularity (seconds)",
    TIME_DIFFERENCE_UUID: "Time Difference (seconds)",
    WIDTH_UUID: "Twitch Width {} (seconds)",
    CONTRACTION_TIME_UUID: "Time From Contraction {} to Peak (seconds)",
    BASELINE_TO_PEAK_UUID: "Time From Contraction {} to Peak (seconds)",
    RELAXATION_TIME_UUID: "Time From Peak to Relaxation {} (seconds)",
    PEAK_TO_BASELINE_UUID: "Time From Peak to Relaxation {} (seconds)",
}

CALCULATED_METRICS = immutabledict(
    {
        "by_width": (WIDTH_UUID, CONTRACTION_TIME_UUID, RELAXATION_TIME_UUID),
        "scalar": (
            AMPLITUDE_UUID,
            AUC_UUID,
            BASELINE_TO_PEAK_UUID,
            CONTRACTION_VELOCITY_UUID,
            FRACTION_MAX_UUID,
            IRREGULARITY_INTERVAL_UUID,
            PEAK_TO_BASELINE_UUID,
            RELAXATION_VELOCITY_UUID,
            TWITCH_FREQUENCY_UUID,
            TWITCH_PERIOD_UUID,
        ),
    }
)

EXCEL_OPTICAL_METADATA_CELLS = immutabledict(
    {
        WELL_NAME_UUID: "E2",
        UTC_BEGINNING_RECORDING_UUID: "E3",
        PLATE_BARCODE_UUID: "E4",
        TISSUE_SAMPLING_PERIOD_UUID: "E5",
        TWITCHES_POINT_UP_UUID: "E6",
        MANTARRAY_SERIAL_NUMBER_UUID: "E7",
        INTERPOLATION_VALUE_UUID: "E8",
    }
)


# 10 seconds at sampling rate of 100Hz
BASELINE_MEAN_NUM_DATA_POINTS = 10 * 100


MIN_FILE_VERSION_FOR_STIM_INTERPOLATION = "1.3.0"
STIM_COMPLETE_SUBPROTOCOL_IDX = 255
