# -*- coding: utf-8 -*-
"""Tests for PlateRecording subclass"""

import os
import uuid

import numpy as np
import pandas as pd
import pulse3D.metrics as metrics
from pulse3D.peak_detection import find_twitch_indices
from pulse3D.peak_detection import peak_detector
from pulse3D.plate_recording import WellFile
import pyarrow.parquet as pq
import pytest
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

    metric = metrics.TwitchAmplitude()
    estimate = metric.fit(pv, w.force, twitch_indices)

    np.testing.assert_array_almost_equal(estimate, expected)


def test_metrics__TwitchAUC():
    file_path = os.path.join(PATH_TO_EXPECTED_METRICS_FOLDER, "auc.parquet")
    expected = pq.read_table(file_path).to_pandas().squeeze()

    w = WellFile(PATH_TO_TEST_H5_FILE)
    pv = peak_detector(w.force, prominence_factors=PROMINENCE_FACTORS, width_factors=WIDTH_FACTORS)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchAUC()
    estimate = metric.fit(pv, w.force, twitch_indices)
    np.testing.assert_array_almost_equal(estimate, expected)


def test_metrics__TwitchBaselineToPeak():
    file_path = os.path.join(PATH_TO_EXPECTED_METRICS_FOLDER, "baseline_to_peak.parquet")
    expected = pq.read_table(file_path).to_pandas().squeeze()

    w = WellFile(PATH_TO_TEST_H5_FILE)
    pv = peak_detector(w.force, prominence_factors=PROMINENCE_FACTORS, width_factors=WIDTH_FACTORS)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchPeakToBaseline(is_contraction=True)
    estimate = metric.fit(pv, w.force, twitch_indices)

    np.testing.assert_array_equal(estimate, expected)


def test_metrics__TwitchPeakToBaseline():
    file_path = os.path.join(PATH_TO_EXPECTED_METRICS_FOLDER, "peak_to_baseline.parquet")
    expected = pq.read_table(file_path).to_pandas().squeeze()

    w = WellFile(PATH_TO_TEST_H5_FILE)
    pv = peak_detector(w.force, prominence_factors=PROMINENCE_FACTORS, width_factors=WIDTH_FACTORS)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchPeakToBaseline(is_contraction=False)
    estimate = metric.fit(pv, w.force, twitch_indices)

    np.testing.assert_array_equal(estimate, expected)


def test_metrics__TwitchFracAmp():
    file_path = os.path.join(PATH_TO_EXPECTED_METRICS_FOLDER, "fraction_max_amplitude.parquet")
    expected = pq.read_table(file_path).to_pandas().squeeze()

    w = WellFile(PATH_TO_TEST_H5_FILE)
    pv = peak_detector(w.force, prominence_factors=PROMINENCE_FACTORS, width_factors=WIDTH_FACTORS)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchFractionAmplitude()
    estimate = metric.fit(pv, w.force, twitch_indices)

    np.testing.assert_array_almost_equal(estimate, expected)


def test_metrics__TwitchFreq():
    file_path = os.path.join(PATH_TO_EXPECTED_METRICS_FOLDER, "frequency.parquet")
    expected = pq.read_table(file_path).to_pandas().squeeze()

    w = WellFile(PATH_TO_TEST_H5_FILE)
    pv = peak_detector(w.force, prominence_factors=PROMINENCE_FACTORS, width_factors=WIDTH_FACTORS)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchFrequency()
    estimate = metric.fit(pv, w.force, twitch_indices)

    np.testing.assert_array_almost_equal(estimate, expected)


def test_metrics__TwitchIrregularity():
    file_path = os.path.join(PATH_TO_EXPECTED_METRICS_FOLDER, "irregularity_interval.parquet")
    expected = pq.read_table(file_path).to_pandas().squeeze()

    w = WellFile(PATH_TO_TEST_H5_FILE)
    pv = peak_detector(w.force, prominence_factors=PROMINENCE_FACTORS, width_factors=WIDTH_FACTORS)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchIrregularity()
    estimate = metric.fit(pv, w.force, twitch_indices)

    np.testing.assert_array_almost_equal(estimate[1:-1], expected[1:-1])


def test_metrics__TwitchPeriod():
    file_path = os.path.join(PATH_TO_EXPECTED_METRICS_FOLDER, "period.parquet")
    expected = pq.read_table(file_path).to_pandas().squeeze()

    w = WellFile(PATH_TO_TEST_H5_FILE)
    pv = peak_detector(w.force, prominence_factors=PROMINENCE_FACTORS, width_factors=WIDTH_FACTORS)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchPeriod()
    estimate = metric.fit(pv, w.force, twitch_indices)

    np.testing.assert_array_almost_equal(estimate, expected)


def test_metrics__TwitchContractionVelocity():
    file_path = os.path.join(PATH_TO_EXPECTED_METRICS_FOLDER, "contraction_velocity.parquet")
    expected = pq.read_table(file_path).to_pandas().squeeze()

    w = WellFile(PATH_TO_TEST_H5_FILE)
    pv = peak_detector(w.force, prominence_factors=PROMINENCE_FACTORS, width_factors=WIDTH_FACTORS)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchVelocity(rounded=False, is_contraction=True)
    estimate = metric.fit(pv, w.force, twitch_indices)

    np.testing.assert_array_almost_equal(estimate, expected)


def test_metrics__TwitchRelaxationVelocity():
    file_path = os.path.join(PATH_TO_EXPECTED_METRICS_FOLDER, "relaxation_velocity.parquet")
    expected = pq.read_table(file_path).to_pandas().squeeze()

    w = WellFile(PATH_TO_TEST_H5_FILE)
    pv = peak_detector(w.force, prominence_factors=PROMINENCE_FACTORS, width_factors=WIDTH_FACTORS)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchVelocity(rounded=False, is_contraction=False)
    estimate = metric.fit(pv, w.force, twitch_indices)

    np.testing.assert_array_almost_equal(estimate, expected)


##### TESTS FOR BY-WIDTH METRICS #####
def test_metrics__TwitchWidth():
    file_path = os.path.join(PATH_TO_EXPECTED_METRICS_FOLDER, "width.parquet")
    expected = pq.read_table(file_path).to_pandas().squeeze()

    w = WellFile(PATH_TO_TEST_H5_FILE)

    pv = peak_detector(w.force)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchWidth()
    estimate = metric.fit(pv, w.force, twitch_indices)

    np.testing.assert_array_almost_equal(estimate, expected)


def test_metrics__TwitchContractionTime():
    file_path = os.path.join(PATH_TO_EXPECTED_METRICS_FOLDER, "contraction_time.parquet")
    expected = pq.read_table(file_path).to_pandas().squeeze()

    w = WellFile(PATH_TO_TEST_H5_FILE)
    pv = peak_detector(w.force, prominence_factors=PROMINENCE_FACTORS, width_factors=WIDTH_FACTORS)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchPeakTime(is_contraction=True)
    estimate = metric.fit(pv, w.force, twitch_indices)

    np.testing.assert_array_almost_equal(estimate, expected)


def test_metrics__TwitchRelaxationTime():
    file_path = os.path.join(PATH_TO_EXPECTED_METRICS_FOLDER, "relaxation_time.parquet")
    expected = pq.read_table(file_path).to_pandas().squeeze()

    w = WellFile(PATH_TO_TEST_H5_FILE)
    pv = peak_detector(w.force, prominence_factors=PROMINENCE_FACTORS, width_factors=WIDTH_FACTORS)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchPeakTime(is_contraction=False)
    estimate = metric.fit(pv, w.force, twitch_indices)

    np.testing.assert_array_almost_equal(estimate, expected)


def test_metrics__x_interpolation():
    with pytest.raises(ZeroDivisionError):
        metrics.interpolate_x_for_y_between_two_points(1, 0, 10, 0, 5)


def test_metrics__y_interpolation():
    with pytest.raises(ZeroDivisionError):
        metrics.interpolate_y_for_x_between_two_points(1, 0, 10, 0, 5)


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
