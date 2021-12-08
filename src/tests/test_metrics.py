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

import json
import math
import os
import uuid

import numpy as np
from pulse3D.constants import *
from pulse3D.constants import ALL_METRICS
from pulse3D.constants import MICRO_TO_BASE_CONVERSION
import pulse3D.metrics as metrics
from pulse3D.peak_detection import data_metrics
from pulse3D.peak_detection import find_twitch_indices
from pulse3D.peak_detection import peak_detector
from pulse3D.plate_recording import WellFile
from stdlib_utils import get_current_file_abs_directory

from .fixtures import fixture_generic_deserialized_per_twitch_metrics_output_0_3_1


def get_force_metrics_from_well_file(w: WellFile, metrics_to_create=ALL_METRICS):
    peak_and_valley_indices = peak_detector(w.noise_filtered_magnetic_data)
    return data_metrics(peak_and_valley_indices, w.force)


def encode_dict(d):
    result = {}
    for key, value in d.items():
        if isinstance(key, uuid.UUID):
            key = str(key)
        if isinstance(key, int):
            key = str(key)
        if isinstance(key, np.int64):
            key = str(key)
        if isinstance(value, uuid.UUID):
            value = str(value)
        if isinstance(value, tuple):
            value = list(value)
        elif isinstance(value, dict):
            value = encode_dict(value)
        result.update({key: value})
    return result


__fixtures__ = (fixture_generic_deserialized_per_twitch_metrics_output_0_3_1,)

PATH_OF_CURRENT_FILE = get_current_file_abs_directory()


def test_per_twitch_metrics_for_single_well(
    generic_deserialized_per_twitch_metrics_output_0_3_1,
):
    path = os.path.join(PATH_OF_CURRENT_FILE, "h5", "v0.3.1", "MA20123456__2020_08_17_145752__B3.h5")
    w = WellFile(path)

    main_dict, _ = get_force_metrics_from_well_file(w)
    dmf = generic_deserialized_per_twitch_metrics_output_0_3_1
    twitch = 1084000

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
        # WIDTH_UUID,
        # RELAXATION_TIME_UUID,
        # CONTRACTION_TIME_UUID,
    ]

    for metric in metrics:
        if not isinstance(main_dict[twitch][metric], dict) and not isinstance(dmf[twitch][metric], dict):
            if math.isnan(main_dict[twitch][metric]) and math.isnan(dmf[twitch][metric]):
                continue
        else:
            assert main_dict[twitch][metric] == dmf[twitch][metric]


# def test_metrics__peaks_greater_than_prior_and_subsequent_valleys(generic_well_features):

#     filtered_data, _, twitch_indices = generic_well_features

#     for twitch, pv in twitch_indices.items():
#         assert filtered_data[1, twitch] > filtered_data[1, pv[PRIOR_VALLEY_INDEX_UUID]]
#         assert filtered_data[1, twitch] > filtered_data[1, pv[SUBSEQUENT_VALLEY_INDEX_UUID]]


def test_metrics__TwitchAmplitude():
    with open(
        os.path.join(
            PATH_OF_CURRENT_FILE,
            "data_metrics",
            "v0.3.1",
            "amplitude_MA201110001__2020_09_03_213024__A1.json",
        )
    ) as f:
        expected = json.load(f)

    expected = np.asarray([float(x) for x in expected])
    expected *= MICRO_TO_BASE_CONVERSION

    w = WellFile(
        os.path.join(
            PATH_OF_CURRENT_FILE,
            "h5",
            "v0.3.1",
            "MA201110001__2020_09_03_213024",
            "MA201110001__2020_09_03_213024__A1.h5",
        )
    )
    pv = peak_detector(w.noise_filtered_magnetic_data)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchAmplitude()
    estimate = metric.fit(pv, w.force, twitch_indices)

    assert np.all(expected == estimate)  # [n for n in estimate]


def test_metrics__TwitchAUC():
    with open(
        os.path.join(
            PATH_OF_CURRENT_FILE, "data_metrics", "v0.3.1", "auc_MA201110001__2020_09_03_213024__A1.json"
        )
    ) as f:
        expected = json.load(f)

    w = WellFile(
        os.path.join(
            PATH_OF_CURRENT_FILE,
            "h5",
            "v0.3.1",
            "MA201110001__2020_09_03_213024",
            "MA201110001__2020_09_03_213024__A1.h5",
        )
    )
    pv = peak_detector(w.noise_filtered_magnetic_data)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchAUC()
    estimate = metric.fit(pv, w.force, twitch_indices)

    assert expected == [str(n) for n in estimate]


def test_metrics__TwitchFracAmp():
    with open(
        os.path.join(
            PATH_OF_CURRENT_FILE,
            "data_metrics",
            "v0.3.1",
            "fraction_max_MA201110001__2020_09_03_213024__A1.json",
        )
    ) as f:
        expected = json.load(f)

    w = WellFile(
        os.path.join(
            PATH_OF_CURRENT_FILE,
            "h5",
            "v0.3.1",
            "MA201110001__2020_09_03_213024",
            "MA201110001__2020_09_03_213024__A1.h5",
        )
    )
    pv = peak_detector(w.noise_filtered_magnetic_data)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchFractionAmplitude()
    estimate = metric.fit(pv, w.force, twitch_indices)

    assert expected == [str(n) for n in estimate]


def test_metrics__TwitchFreq():
    with open(
        os.path.join(
            PATH_OF_CURRENT_FILE,
            "data_metrics",
            "v0.3.1",
            "twitch_frequency_MA201110001__2020_09_03_213024__A1.json",
        )
    ) as f:
        expected = json.load(f)

    w = WellFile(
        os.path.join(
            PATH_OF_CURRENT_FILE,
            "h5",
            "v0.3.1",
            "MA201110001__2020_09_03_213024",
            "MA201110001__2020_09_03_213024__A1.h5",
        )
    )
    pv = peak_detector(w.noise_filtered_magnetic_data)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchFrequency()
    estimate = metric.fit(pv, w.force, twitch_indices)

    assert expected == [str(n) for n in estimate]


def test_metrics__TwitchIrregularity():
    with open(
        os.path.join(
            PATH_OF_CURRENT_FILE,
            "data_metrics",
            "v0.3.1",
            "irregularity_interval_MA201110001__2020_09_03_213024__A1.json",
        )
    ) as f:
        expected = json.load(f)

    expected = np.asarray([float(x) for x in expected])
    expected /= MICRO_TO_BASE_CONVERSION

    w = WellFile(
        os.path.join(
            PATH_OF_CURRENT_FILE,
            "h5",
            "v0.3.1",
            "MA201110001__2020_09_03_213024",
            "MA201110001__2020_09_03_213024__A1.h5",
        )
    )
    pv = peak_detector(w.noise_filtered_magnetic_data)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchIrregularity()
    estimate = metric.fit(pv, w.force, twitch_indices)

    assert [str(n) for n in expected] == [str(n) for n in estimate]


# def test_metrics__TwitchPeakTime(
#     generic_well_features, generate_twitch_peak_time_contraction, generate_twitch_peak_time_relaxation
# ):

#     [filtered_data, peak_and_valley_indices, twitch_indices] = generic_well_features
#     PARAMS = {
#         "peak_and_valley_indices": peak_and_valley_indices,
#         "filtered_data": filtered_data,
#         "twitch_indices": twitch_indices,
#     }

#     percents = range(10, 95, 5)

#     contraction_metric = metrics.TwitchPeakTime(
#         rounded=False, is_contraction=True, twitch_width_percents=percents
#     )
#     contractions = contraction_metric.fit(**PARAMS)

#     relaxation_metric = metrics.TwitchPeakTime(
#         rounded=False, is_contraction=False, twitch_width_percents=percents
#     )
#     relaxations = relaxation_metric.fit(**PARAMS)

#     for i in range(len(percents) - 1):
#         assert (
#             contractions[0][percents[i]][TIME_VALUE_UUID] > contractions[0][percents[i + 1]][TIME_VALUE_UUID]
#         )

#     for i in range(len(percents) - 1):
#         assert relaxations[0][percents[i]][TIME_VALUE_UUID] < relaxations[0][percents[i + 1]][TIME_VALUE_UUID]

#     # regression
#     assert np.all(contractions == generate_twitch_peak_time_contraction)
#     assert np.all(relaxations == generate_twitch_peak_time_relaxation)


# def test_metrics__TwitchPeakToBaseline_is_contraction(
#     generic_well_features, generate_twitch_baseline_to_peak
# ):

#     [filtered_data, peak_and_valley_indices, twitch_indices] = generic_well_features
#     PARAMS = {
#         "peak_and_valley_indices": peak_and_valley_indices,
#         "filtered_data": filtered_data,
#         "twitch_indices": twitch_indices,
#     }

#     metric = metrics.TwitchPeakToBaseline(rounded=False, is_contraction=True)
#     estimate = metric.fit(**PARAMS)

#     assert np.all(estimate > 0)
#     # regression
#     assert np.all(estimate == generate_twitch_baseline_to_peak)


# def test_metrics__TwitchPeakToBaseline_is_relaxation(generic_well_features, generate_twitch_peak_to_baseline):

#     [filtered_data, peak_and_valley_indices, twitch_indices] = generic_well_features
#     PARAMS = {
#         "peak_and_valley_indices": peak_and_valley_indices,
#         "filtered_data": filtered_data,
#         "twitch_indices": twitch_indices,
#     }

#     metric = metrics.TwitchPeakToBaseline(rounded=False, is_contraction=False)
#     estimate = metric.fit(**PARAMS)

#     assert np.all(estimate > 0)
#     # regression
#     assert np.all(estimate == generate_twitch_peak_to_baseline)


def test_metrics__TwitchPeriod():
    with open(
        os.path.join(
            PATH_OF_CURRENT_FILE,
            "data_metrics",
            "v0.3.1",
            "twitch_period_MA201110001__2020_09_03_213024__A1.json",
        )
    ) as f:
        expected = json.load(f)

    expected = np.asarray([float(n) for n in expected])
    expected /= MICRO_TO_BASE_CONVERSION

    w = WellFile(
        os.path.join(
            PATH_OF_CURRENT_FILE,
            "h5",
            "v0.3.1",
            "MA201110001__2020_09_03_213024",
            "MA201110001__2020_09_03_213024__A1.h5",
        )
    )
    pv = peak_detector(w.noise_filtered_magnetic_data)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchPeriod()
    estimate = metric.fit(pv, w.force, twitch_indices)

    assert [str(n) for n in expected] == [str(n) for n in estimate]


def test_metrics__TwitchContractionVelocity():
    with open(
        os.path.join(
            PATH_OF_CURRENT_FILE,
            "data_metrics",
            "v0.3.1",
            "contraction_velocity_MA201110001__2020_09_03_213024__A1.json",
        )
    ) as f:
        expected = json.load(f)

    expected = np.asarray([float(n) for n in expected])
    expected *= MICRO_TO_BASE_CONVERSION ** 2

    w = WellFile(
        os.path.join(
            PATH_OF_CURRENT_FILE,
            "h5",
            "v0.3.1",
            "MA201110001__2020_09_03_213024",
            "MA201110001__2020_09_03_213024__A1.h5",
        )
    )
    pv = peak_detector(w.noise_filtered_magnetic_data)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchVelocity(rounded=False, is_contraction=True)
    estimate = metric.fit(pv, w.force, twitch_indices)

    assert [str(n) for n in expected] == [str(n) for n in estimate]


def test_metrics__TwitchRelaxationVelocity():
    with open(
        os.path.join(
            PATH_OF_CURRENT_FILE,
            "data_metrics",
            "v0.3.1",
            "relaxation_velocity_MA201110001__2020_09_03_213024__A1.json",
        )
    ) as f:
        expected = json.load(f)

    expected = np.asarray([float(n) for n in expected])
    expected *= MICRO_TO_BASE_CONVERSION ** 2

    w = WellFile(
        os.path.join(
            PATH_OF_CURRENT_FILE,
            "h5",
            "v0.3.1",
            "MA201110001__2020_09_03_213024",
            "MA201110001__2020_09_03_213024__A1.h5",
        )
    )
    pv = peak_detector(w.noise_filtered_magnetic_data)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchVelocity(rounded=False, is_contraction=False)
    estimate = metric.fit(pv, w.force, twitch_indices)

    assert [str(n) for n in expected] == [str(n) for n in estimate]


# def test_metrics__TwitchWidth():
#     with open(os.path.join(PATH_OF_CURRENT_FILE, "data_metrics", "v0.3.1", "width_MA201110001__2020_09_03_213024__A1.json")) as f:
#         expected = json.load(f)

#     w = WellFile(os.path.join(PATH_OF_CURRENT_FILE, "h5", "v0.3.1", "MA201110001__2020_09_03_213024", "MA201110001__2020_09_03_213024__A1.h5"))
#     pv = peak_detector(w.noise_filtered_magnetic_data)
#     twitch_indices = find_twitch_indices(pv)

#     metric = metrics.TwitchWidth()
#     estimate = metric.fit(pv, w.force, twitch_indices)

#     assert expected == [encode_dict(n) for n in estimate]


# def test_metrics__x_interpolation():

#     with pytest.raises(ZeroDivisionError):
#         metrics.interpolate_x_for_y_between_two_points(1, 0, 10, 0, 5)


# def test_metrics__y_interpolation():

#     with pytest.raises(ZeroDivisionError):
#         metrics.interpolate_y_for_x_between_two_points(1, 0, 10, 0, 5)


# def test_metrics__create_statistics():

#     estimates = np.asarray([1, 2, 3, 4, 5])

#     statistics = metrics.BaseMetric.create_statistics_dict(estimates, rounded=False)
#     assert statistics["mean"] == np.nanmean(estimates)
#     assert statistics["std"] == np.nanstd(estimates)
#     assert statistics["min"] == np.nanmin(estimates)
#     assert statistics["max"] == np.nanmax(estimates)

#     statistics = metrics.BaseMetric.create_statistics_dict(estimates, rounded=True)
#     assert statistics["mean"] == int(round(np.nanmean(estimates)))
#     assert statistics["std"] == int(round(np.nanstd(estimates)))
#     assert statistics["min"] == int(round(np.nanmin(estimates)))
#     assert statistics["max"] == int(round(np.nanmax(estimates)))

#     estimates = []
#     statistics = metrics.BaseMetric.create_statistics_dict(estimates, rounded=False)
#     assert statistics["mean"] is None
#     assert statistics["std"] is None
#     assert statistics["min"] is None
#     assert statistics["max"] is None
