# -*- coding: utf-8 -*-
"""Tests for PlateRecording subclass.

To create a file to look at: python3 -c "import os; from curibio.sdk import PlateRecording; PlateRecording([os.path.join('tests','h5','v0.3.1','MA20123456__2020_08_17_145752__A1.h5')]).write_xlsx('.',file_name='temp.xlsx')"
To create a file to look at: python3 -c "import os; from curibio.sdk import PlateRecording; PlateRecording([os.path.join('tests','h5','v0.3.1','MA201110001__2020_09_03_213024__A3.h5')]).write_xlsx('.',file_name='temp.xlsx')"
To create a file to look at: python3 -c "import os; from curibio.sdk import PlateRecording; PlateRecording.from_directory(os.path.join('tests','h5','v0.3.1')).write_xlsx('.',file_name='temp.xlsx')"

python3 -c "import os; from curibio.sdk import PlateRecording; PlateRecording([os.path.join('tests','h5','v0.3.1','MA201110001__2020_09_03_213024','MA201110001__2020_09_03_213024__A1.h5',),os.path.join('tests','h5','v0.3.1','MA201110001__2020_09_03_213024','MA201110001__2020_09_03_213024__B2.h5',),]).write_xlsx('.',file_name='temp.xlsx')"
python3 -c "import os; from curibio.sdk import PlateRecording; PlateRecording([os.path.join('tests','h5','v0.3.1','"MA20123456__2020_08_17_145752__A2.h5')]).write_xlsx('.',file_name='temp.xlsx')"

python3 -c "import os; from curibio.sdk import PlateRecording; PlateRecording([os.path.join('tests','h5','v0.3.1','MA201110001__2020_09_03_213024__A3.h5')]).write_xlsx('.',file_name='temp.xlsx',twitch_width_values=(25,), show_twitch_coordinate_values=True)"
python3 -c "import os; from curibio.sdk import PlateRecording; PlateRecording([os.path.join('tests','h5','v0.3.1','MA201110001__2020_09_03_213024__A3.h5')]).write_xlsx('.',file_name='temp.xlsx', show_twitch_coordinate_values=True, show_twitch_time_diff_values=True)"

"""

import math
import os
import uuid

from constants import ALL_METRICS
from constants import CONTRACTION_TIME_UUID
from constants import RELAXATION_TIME_UUID
from constants import TIME_DIFFERENCE_UUID
from stdlib_utils import get_current_file_abs_directory

from .fixtures import fixture_generic_deserialized_per_twitch_metrics_output_0_3_1

from plate_recording import WellFile
from peak_detection import peak_detector, data_metrics

def get_force_metrics_from_well_file(w: WellFile, metrics_to_create=ALL_METRICS):
    peak_and_valley_indices = peak_detector(w.noise_filtered_magnetic_data)
    return data_metrics(peak_and_valley_indices, w.force)

__fixtures__ = (
    fixture_generic_deserialized_per_twitch_metrics_output_0_3_1,
)

PATH_OF_CURRENT_FILE = get_current_file_abs_directory()

def test_per_twitch_metrics_for_single_well(
    generic_deserialized_per_twitch_metrics_output_0_3_1,
):
    path = os.path.join(PATH_OF_CURRENT_FILE, "h5", "v0.3.1", "MA20123456__2020_08_17_145752__B3.h5")
    w = WellFile(path)

    main_dict, _ = get_force_metrics_from_well_file(w)
    dmf = generic_deserialized_per_twitch_metrics_output_0_3_1
    twitch = 1084000

    for metric in ALL_METRICS:
        if not isinstance(main_dict[twitch][metric], dict) and not isinstance(dmf[twitch][metric], dict):
            if math.isnan(main_dict[twitch][metric]) and math.isnan(dmf[twitch][metric]):
                continue
        else:
            assert main_dict[twitch][metric] == dmf[twitch][metric]

