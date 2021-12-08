# -*- coding: utf-8 -*-
import os

from pulse3D.constants import *
from pulse3D.utils import deserialize_main_dict
import pytest
from stdlib_utils import get_current_file_abs_directory

# from curibio.sdk import ExcelWellFile
# from curibio.sdk import PlateRecording
# from curibio.sdk import WellFile

PATH_OF_CURRENT_FILE = get_current_file_abs_directory()


@pytest.fixture(scope="function", name="generic_deserialized_per_twitch_metrics_output_0_3_1")
def fixture_generic_deserialized_per_twitch_metrics_output_0_3_1():

    file_name = "MA20123456__2020_08_17_145752__B3_per_twitch.json"
    metrics_path = os.path.join(PATH_OF_CURRENT_FILE, "data_metrics", "v0.3.1", file_name)

    metrics = [
        TWITCH_PERIOD_UUID,
        FRACTION_MAX_UUID,
        AMPLITUDE_UUID,
        AUC_UUID,
        TWITCH_FREQUENCY_UUID,
        CONTRACTION_VELOCITY_UUID,
        RELAXATION_VELOCITY_UUID,
        IRREGULARITY_INTERVAL_UUID,
        # BASELINE_TO_PEAK_UUID, # older set of tests outputs doesn't have these metrics
        # PEAK_TO_BASELINE_UUID,
        WIDTH_UUID,
        # RELAXATION_TIME_UUID,
        # CONTRACTION_TIME_UUID,
    ]

    yield deserialize_main_dict(metrics_path, metrics)
