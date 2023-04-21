# -*- coding: utf-8 -*-

import os
import time
import uuid

import numpy as np
import pandas as pd
from pulse3D.constants import DEFAULT_BASELINE_WIDTHS
from pulse3D.constants import DEFAULT_TWITCH_WIDTH_PERCENTS
import pulse3D.metrics as metrics
from pulse3D.metrics import MetricCalculator
from pulse3D.peak_detection import find_twitch_indices
from pulse3D.peak_detection import peak_detector
from pulse3D.plate_recording import WellFile
import pyarrow.parquet as pq
from stdlib_utils import get_current_file_abs_directory

from ..fixtures_utils import PATH_TO_DATA_METRIC_FILES
from ..fixtures_utils import PATH_TO_H5_FILES

PATH_OF_CURRENT_FILE = get_current_file_abs_directory()

# prominence and width scaling factors for peak detection
PROMINENCE_FACTORS = (4, 4)
WIDTH_FACTORS = (2, 2)


PATH_TO_TEST_H5_FILE = os.path.join(
    PATH_TO_H5_FILES, "v0.3.1", "MA201110001__2020_09_03_213024", "MA201110001__2020_09_03_213024__A1.h5"
)

PATH_TO_EXPECTED_METRICS_FOLDER = os.path.join(
    PATH_TO_DATA_METRIC_FILES, "v0.3.1", "MA201110001__2020_09_03_213024", "A1"
)


def create_new_parquet_for_testing(estimate: pd.DataFrame, filename: str) -> None:
    """Function for creating new parquet files with new data for tests.

    ONLY USE THIS FUNCTION TO CHANGE THE PARQUET FILE IF THE RESULTS OF THE METRIC CALCULATIONS WERE INTENTIONALLY MODIFIED.

    If a test is failing and the metric calculations were only refactored or changed in any other
    way that was not meant to change the result, do not use this function, fix the calculation instead.

    The new file will show up in the root of the pulse3D repo and can then be replaced in the data_metrics folder.
    """
    f = pd.DataFrame({"0": estimate.values})
    f.to_parquet(f"NEW_{filename}")


def encode_dict(d):
    """Recursive function to encode dictionary for saving as JSON file.

    Args:
        d ([dict]): dictionary of metric values

    Returns:
        result [dict]: encoded dictionary of metric values
    """
    result = {}
    for key, value in d.items():
        # fix key
        if isinstance(key, (int, uuid.UUID, np.int64)):
            key = str(key)
        # fix value
        if isinstance(value, uuid.UUID):
            value = str(value)
        elif isinstance(value, tuple):
            value = list(value)
        elif isinstance(value, dict):
            value = encode_dict(value)
        # add to new dict
        result[key] = value
    return result


##### TESTS FOR SCALAR METRICS #####
def test_metrics__TwitchAmplitude():
    file_path = os.path.join(PATH_TO_EXPECTED_METRICS_FOLDER, "amplitude.parquet")
    expected = pq.read_table(file_path).to_pandas().squeeze()

    w = WellFile(PATH_TO_TEST_H5_FILE)

    pv = peak_detector(w.force, prominence_factors=PROMINENCE_FACTORS, width_factors=WIDTH_FACTORS)
    twitch_indices = find_twitch_indices(pv)

    # metric = metrics.TwitchAmplitude()
    # estimate = metric.fit(pv, w.force, twitch_indices)

    # 18.37362500000017
    start = time.perf_counter()
    mc = MetricCalculator(
        w.force, twitch_indices, pv, DEFAULT_TWITCH_WIDTH_PERCENTS, DEFAULT_BASELINE_WIDTHS, rounded=False
    )
    amp = mc["twitch_amplitude"]
    print("$$$", (time.perf_counter() - start) * 1000)

    np.testing.assert_array_almost_equal(amp, expected)


def test_metrics__TwitchAUC():
    file_path = os.path.join(PATH_TO_EXPECTED_METRICS_FOLDER, "auc.parquet")
    expected = pq.read_table(file_path).to_pandas().squeeze()

    w = WellFile(PATH_TO_TEST_H5_FILE)
    pv = peak_detector(w.force, prominence_factors=PROMINENCE_FACTORS, width_factors=WIDTH_FACTORS)
    twitch_indices = find_twitch_indices(pv)

    # metric = metrics.TwitchAUC()
    # estimate = metric.fit(pv, w.force, twitch_indices)

    mc = MetricCalculator(
        w.force, twitch_indices, pv, DEFAULT_TWITCH_WIDTH_PERCENTS, DEFAULT_BASELINE_WIDTHS, rounded=False
    )

    np.testing.assert_array_almost_equal(mc["twitch_auc"], expected)


def test_metrics__TwitchFractionAmplitude():
    file_path = os.path.join(PATH_TO_EXPECTED_METRICS_FOLDER, "fraction_max_amplitude.parquet")
    expected = pq.read_table(file_path).to_pandas().squeeze()

    w = WellFile(PATH_TO_TEST_H5_FILE)
    pv = peak_detector(w.force, prominence_factors=PROMINENCE_FACTORS, width_factors=WIDTH_FACTORS)
    twitch_indices = find_twitch_indices(pv)

    # metric = metrics.TwitchFractionAmplitude()
    # estimate = metric.fit(pv, w.force, twitch_indices)

    mc = MetricCalculator(
        w.force, twitch_indices, pv, DEFAULT_TWITCH_WIDTH_PERCENTS, DEFAULT_BASELINE_WIDTHS, rounded=False
    )

    np.testing.assert_array_almost_equal(mc["twitch_fraction_amplitude"], expected, decimal=3)


def test_metrics__TwitchFrequency():
    file_path = os.path.join(PATH_TO_EXPECTED_METRICS_FOLDER, "frequency.parquet")
    expected = pq.read_table(file_path).to_pandas().squeeze()

    w = WellFile(PATH_TO_TEST_H5_FILE)
    pv = peak_detector(w.force, prominence_factors=PROMINENCE_FACTORS, width_factors=WIDTH_FACTORS)
    twitch_indices = find_twitch_indices(pv)

    # metric = metrics.TwitchFrequency()
    # estimate = metric.fit(pv, w.force, twitch_indices)

    mc = MetricCalculator(
        w.force, twitch_indices, pv, DEFAULT_TWITCH_WIDTH_PERCENTS, DEFAULT_BASELINE_WIDTHS, rounded=False
    )

    np.testing.assert_array_almost_equal(mc["twitch_frequency"], expected)


def test_metrics__TwitchIrregularityInterval():
    file_path = os.path.join(PATH_TO_EXPECTED_METRICS_FOLDER, "irregularity_interval.parquet")
    expected = pq.read_table(file_path).to_pandas().squeeze()

    w = WellFile(PATH_TO_TEST_H5_FILE)
    pv = peak_detector(w.force, prominence_factors=PROMINENCE_FACTORS, width_factors=WIDTH_FACTORS)
    twitch_indices = find_twitch_indices(pv)

    # metric = metrics.TwitchIrregularity()
    # estimate = metric.fit(pv, w.force, twitch_indices)

    mc = MetricCalculator(
        w.force, twitch_indices, pv, DEFAULT_TWITCH_WIDTH_PERCENTS, DEFAULT_BASELINE_WIDTHS, rounded=False
    )

    np.testing.assert_array_almost_equal(mc["twitch_interval_irregularity"][1:-1], expected[1:-1])


def test_metrics__TwitchVelocity__contraction():
    file_path = os.path.join(PATH_TO_EXPECTED_METRICS_FOLDER, "contraction_velocity.parquet")
    expected = pq.read_table(file_path).to_pandas().squeeze()

    w = WellFile(PATH_TO_TEST_H5_FILE)
    pv = peak_detector(w.force, prominence_factors=PROMINENCE_FACTORS, width_factors=WIDTH_FACTORS)
    twitch_indices = find_twitch_indices(pv)

    # metric = metrics.TwitchVelocity(rounded=False, is_contraction=True)
    # estimate = metric.fit(pv, w.force, twitch_indices)

    mc = MetricCalculator(
        w.force, twitch_indices, pv, DEFAULT_TWITCH_WIDTH_PERCENTS, DEFAULT_BASELINE_WIDTHS, rounded=False
    )

    np.testing.assert_array_almost_equal(mc["twitch_contraction_velocity"], expected)


def test_metrics__TwitchVelocity__relaxation():
    file_path = os.path.join(PATH_TO_EXPECTED_METRICS_FOLDER, "relaxation_velocity.parquet")
    expected = pq.read_table(file_path).to_pandas().squeeze()

    w = WellFile(PATH_TO_TEST_H5_FILE)
    pv = peak_detector(w.force, prominence_factors=PROMINENCE_FACTORS, width_factors=WIDTH_FACTORS)
    twitch_indices = find_twitch_indices(pv)

    # metric = metrics.TwitchVelocity(rounded=False, is_contraction=False)
    # estimate = metric.fit(pv, w.force, twitch_indices)

    mc = MetricCalculator(
        w.force, twitch_indices, pv, DEFAULT_TWITCH_WIDTH_PERCENTS, DEFAULT_BASELINE_WIDTHS, rounded=False
    )

    np.testing.assert_array_almost_equal(mc["twitch_relaxation_velocity"], expected)


##### TESTS FOR BY-WIDTH METRICS #####
def test_metrics__TwitchWidth():
    file_path = os.path.join(PATH_TO_EXPECTED_METRICS_FOLDER, "width.parquet")
    expected = pq.read_table(file_path).to_pandas().squeeze()

    w = WellFile(PATH_TO_TEST_H5_FILE)

    pv = peak_detector(w.force)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchWidth()
    width_df, _ = metric.calculate_twitch_widths(filtered_data=w.force, twitch_indices=twitch_indices)

    np.testing.assert_array_almost_equal(width_df, expected, decimal=4)


def test_metrics__TwitchPeakTime__contraction():
    file_path = os.path.join(PATH_TO_EXPECTED_METRICS_FOLDER, "contraction_time.parquet")
    expected = pq.read_table(file_path).to_pandas().squeeze()

    w = WellFile(PATH_TO_TEST_H5_FILE)
    pv = peak_detector(w.force, prominence_factors=PROMINENCE_FACTORS, width_factors=WIDTH_FACTORS)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchPeakTime(is_contraction=True)
    estimate = metric.fit(pv, w.force, twitch_indices)

    np.testing.assert_array_almost_equal(estimate, expected, decimal=5)


def test_metrics__TwitchPeakTime__relaxation():
    file_path = os.path.join(PATH_TO_EXPECTED_METRICS_FOLDER, "relaxation_time.parquet")
    expected = pq.read_table(file_path).to_pandas().squeeze()

    w = WellFile(PATH_TO_TEST_H5_FILE)
    pv = peak_detector(w.force, prominence_factors=PROMINENCE_FACTORS, width_factors=WIDTH_FACTORS)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchPeakTime(is_contraction=False)
    estimate = metric.fit(pv, w.force, twitch_indices)

    np.testing.assert_array_almost_equal(estimate, expected, decimal=5)


def test_metrics__create_statistics():
    estimates = np.asarray([1, 2, 3, 4, 5])

    statistics = metrics.BaseMetric.create_statistics_df(estimates, rounded=False)

    assert statistics["Mean"][0] == np.nanmean(estimates)
    assert statistics["StDev"][0] == np.nanstd(estimates)
    assert statistics["Min"][0] == np.nanmin(estimates)
    assert statistics["Max"][0] == np.nanmax(estimates)

    statistics = metrics.BaseMetric.create_statistics_df(estimates, rounded=True)
    assert statistics["Mean"][0] == int(round(np.nanmean(estimates)))
    assert statistics["StDev"][0] == int(round(np.nanstd(estimates)))
    assert statistics["Min"][0] == int(round(np.nanmin(estimates)))
    assert statistics["Max"][0] == int(round(np.nanmax(estimates)))

    estimates = []
    statistics = metrics.BaseMetric.create_statistics_df(estimates, rounded=False)
    assert statistics["Mean"][0] is None
    assert statistics["StDev"][0] is None
    assert statistics["Min"][0] is None
    assert statistics["Max"][0] is None
