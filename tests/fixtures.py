# -*- coding: utf-8 -*-
import json
import os
import tempfile

# from curibio.sdk import ExcelWellFile
# from curibio.sdk import PlateRecording
# from curibio.sdk import WellFile
from utils import deserialize_aggregate_dict
from utils import deserialize_main_dict
from constants import ALL_METRICS
from constants import BESSEL_LOWPASS_10_UUID

import pytest
from stdlib_utils import get_current_file_abs_directory

PATH_OF_CURRENT_FILE = get_current_file_abs_directory()


@pytest.fixture(scope="function", name="generic_deserialized_per_twitch_metrics_output_0_3_1")
def fixture_generic_deserialized_per_twitch_metrics_output_0_3_1():

    file_name = "MA20123456__2020_08_17_145752__B3_per_twitch.json"
    metrics_path = os.path.join(PATH_OF_CURRENT_FILE, "data_metrics", "v0.3.1", file_name)

    yield deserialize_main_dict(metrics_path, ALL_METRICS)
