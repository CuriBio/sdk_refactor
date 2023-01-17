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


def create_new_parquet_for_testing(filename_of_file_to_replace):
    """Function for creating new parquet files with new data for tests.
    The new file will show up in the root of the pulse3D repo and can then be replaced in the data_metrics folder. files

    Args:
        string: name of file in data_files/data_metrics/v0.3.1 to replace

    Returns:
        nothing
    """
    file_path = os.path.join(
        PATH_TO_DATA_METRIC_FILES,
        "v0.3.1",
        filename_of_file_to_replace,
    )
    try:
        table = pq.read_table(file_path)
    except Exception as e:
        raise FileNotFoundError("Parquet file for full twitch relaxation time not found.") from e
    else:
        expected = table.to_pandas().squeeze()

    f = pd.DataFrame({"0": expected.values})
    f.to_parquet(f"NEW_{filename_of_file_to_replace}")


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
    file_path = os.path.join(
        PATH_TO_DATA_METRIC_FILES, "v0.3.1", "amplitude_MA201110001__2020_09_03_213024__A1.parquet"
    )
    try:
        table = pq.read_table(file_path)
    except Exception as e:
        raise FileNotFoundError("Parquet file for twitch amplitude not found.") from e
    else:
        expected = table.to_pandas().squeeze()

    w = WellFile(
        os.path.join(
            PATH_TO_H5_FILES,
            "v0.3.1",
            "MA201110001__2020_09_03_213024",
            "MA201110001__2020_09_03_213024__A1.h5",
        )
    )

    pv = peak_detector(w.force, prominence_factors=PROMINENCE_FACTORS, width_factors=WIDTH_FACTORS)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchAmplitude()
    estimate = metric.fit(pv, w.force, twitch_indices)

    np.testing.assert_array_almost_equal(estimate, expected)


def test_metrics__TwitchAUC():
    file_path = os.path.join(
        PATH_TO_DATA_METRIC_FILES, "v0.3.1", "auc_MA201110001__2020_09_03_213024__A1.parquet"
    )
    try:
        table = pq.read_table(file_path)
    except Exception as e:
        raise FileNotFoundError("Parquet file for twitch AUC not found.") from e
    else:
        expected = table.to_pandas().squeeze()

    w = WellFile(
        os.path.join(
            PATH_TO_H5_FILES,
            "v0.3.1",
            "MA201110001__2020_09_03_213024",
            "MA201110001__2020_09_03_213024__A1.h5",
        )
    )
    pv = peak_detector(w.force, prominence_factors=PROMINENCE_FACTORS, width_factors=WIDTH_FACTORS)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchAUC()
    estimate = metric.fit(pv, w.force, twitch_indices)
    np.testing.assert_array_almost_equal(estimate, expected)


def test_metrics__TwitchBaselineToPeak():
    file_path = os.path.join(
        PATH_TO_DATA_METRIC_FILES,
        "v0.3.1",
        "baseline_to_peak_MA201110001__2020_09_03_213024__A1.parquet",
    )
    try:
        table = pq.read_table(file_path)
    except Exception as e:
        raise FileNotFoundError("Parquet file for full twitch contraction time not found.") from e
    else:
        expected = table.to_pandas().squeeze()

    w = WellFile(
        os.path.join(
            PATH_TO_H5_FILES,
            "v0.3.1",
            "MA201110001__2020_09_03_213024",
            "MA201110001__2020_09_03_213024__A1.h5",
        )
    )
    pv = peak_detector(w.force, prominence_factors=PROMINENCE_FACTORS, width_factors=WIDTH_FACTORS)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchPeakToBaseline(is_contraction=True)
    estimate = metric.fit(pv, w.force, twitch_indices)

    np.testing.assert_array_equal(estimate, expected)


def test_metrics__TwitchPeakToBaseline():
    file_path = os.path.join(
        PATH_TO_DATA_METRIC_FILES,
        "v0.3.1",
        "peak_to_baseline_MA201110001__2020_09_03_213024__A1.parquet",
    )
    try:
        table = pq.read_table(file_path)
    except Exception as e:
        raise FileNotFoundError("Parquet file for full twitch relaxation time not found.") from e
    else:
        expected = table.to_pandas().squeeze()

    w = WellFile(
        os.path.join(
            PATH_TO_H5_FILES,
            "v0.3.1",
            "MA201110001__2020_09_03_213024",
            "MA201110001__2020_09_03_213024__A1.h5",
        )
    )
    pv = peak_detector(w.force, prominence_factors=PROMINENCE_FACTORS, width_factors=WIDTH_FACTORS)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchPeakToBaseline(is_contraction=False)
    estimate = metric.fit(pv, w.force, twitch_indices)

    np.testing.assert_array_equal(estimate, expected)


def test_metrics__TwitchFracAmp():
    file_path = os.path.join(
        PATH_TO_DATA_METRIC_FILES,
        "v0.3.1",
        "fraction_max_MA201110001__2020_09_03_213024__A1.parquet",
    )
    try:
        table = pq.read_table(file_path)
    except Exception as e:
        raise FileNotFoundError("Parquet file for twitch fraction max amplitude not found.") from e
    else:
        expected = table.to_pandas().squeeze()

    w = WellFile(
        os.path.join(
            PATH_TO_H5_FILES,
            "v0.3.1",
            "MA201110001__2020_09_03_213024",
            "MA201110001__2020_09_03_213024__A1.h5",
        )
    )
    pv = peak_detector(w.force, prominence_factors=PROMINENCE_FACTORS, width_factors=WIDTH_FACTORS)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchFractionAmplitude()
    estimate = metric.fit(pv, w.force, twitch_indices)

    np.testing.assert_array_almost_equal(estimate, expected)


def test_metrics__TwitchFreq():
    file_path = os.path.join(
        PATH_TO_DATA_METRIC_FILES,
        "v0.3.1",
        "twitch_frequency_MA201110001__2020_09_03_213024__A1.parquet",
    )
    try:
        table = pq.read_table(file_path)
    except Exception as e:
        raise FileNotFoundError("Parquet file for twitch frequency not found.") from e
    else:
        expected = table.to_pandas().squeeze()

    w = WellFile(
        os.path.join(
            PATH_TO_H5_FILES,
            "v0.3.1",
            "MA201110001__2020_09_03_213024",
            "MA201110001__2020_09_03_213024__A1.h5",
        )
    )
    pv = peak_detector(w.force, prominence_factors=PROMINENCE_FACTORS, width_factors=WIDTH_FACTORS)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchFrequency()
    estimate = metric.fit(pv, w.force, twitch_indices)

    np.testing.assert_array_almost_equal(estimate, expected)


def test_metrics__TwitchIrregularity():
    file_path = os.path.join(
        PATH_TO_DATA_METRIC_FILES,
        "v0.3.1",
        "irregularity_interval_MA201110001__2020_09_03_213024__A1.parquet",
    )
    try:
        table = pq.read_table(file_path)
    except Exception as e:
        raise FileNotFoundError("Parquet file for twitch irregularity interval not found.") from e
    else:
        expected = table.to_pandas().squeeze()

    w = WellFile(
        os.path.join(
            PATH_TO_H5_FILES,
            "v0.3.1",
            "MA201110001__2020_09_03_213024",
            "MA201110001__2020_09_03_213024__A1.h5",
        )
    )
    pv = peak_detector(w.force, prominence_factors=PROMINENCE_FACTORS, width_factors=WIDTH_FACTORS)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchIrregularity()
    estimate = metric.fit(pv, w.force, twitch_indices)

    np.testing.assert_array_almost_equal(estimate[1:-1], expected[1:-1])


def test_metrics__TwitchPeriod():
    file_path = os.path.join(
        PATH_TO_DATA_METRIC_FILES,
        "v0.3.1",
        "twitch_period_MA201110001__2020_09_03_213024__A1.parquet",
    )
    try:
        table = pq.read_table(file_path)
    except Exception as e:
        raise FileNotFoundError("Parquet file for twitch period not found.") from e
    else:
        expected = table.to_pandas().squeeze()

    w = WellFile(
        os.path.join(
            PATH_TO_H5_FILES,
            "v0.3.1",
            "MA201110001__2020_09_03_213024",
            "MA201110001__2020_09_03_213024__A1.h5",
        )
    )
    pv = peak_detector(w.force, prominence_factors=PROMINENCE_FACTORS, width_factors=WIDTH_FACTORS)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchPeriod()
    estimate = metric.fit(pv, w.force, twitch_indices)

    np.testing.assert_array_almost_equal(estimate, expected)


def test_metrics__TwitchContractionVelocity():
    file_path = os.path.join(
        PATH_TO_DATA_METRIC_FILES,
        "v0.3.1",
        "contraction_velocity_MA201110001__2020_09_03_213024__A1.parquet",
    )
    try:
        table = pq.read_table(file_path)
    except Exception as e:
        raise FileNotFoundError("Parquet file for twitch contraction velocity not found.") from e
    else:
        expected = table.to_pandas().squeeze()

    w = WellFile(
        os.path.join(
            PATH_TO_H5_FILES,
            "v0.3.1",
            "MA201110001__2020_09_03_213024",
            "MA201110001__2020_09_03_213024__A1.h5",
        )
    )
    pv = peak_detector(w.force, prominence_factors=PROMINENCE_FACTORS, width_factors=WIDTH_FACTORS)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchVelocity(rounded=False, is_contraction=True)
    estimate = metric.fit(pv, w.force, twitch_indices)

    np.testing.assert_array_almost_equal(estimate, expected)


def test_metrics__TwitchRelaxationVelocity():
    file_path = os.path.join(
        PATH_TO_DATA_METRIC_FILES,
        "v0.3.1",
        "relaxation_velocity_MA201110001__2020_09_03_213024__A1.parquet",
    )
    try:
        table = pq.read_table(file_path)
    except Exception as e:
        raise FileNotFoundError("Parquet file for twitch relaxation velocity not found.") from e
    else:
        expected = table.to_pandas().squeeze()

    w = WellFile(
        os.path.join(
            PATH_TO_H5_FILES,
            "v0.3.1",
            "MA201110001__2020_09_03_213024",
            "MA201110001__2020_09_03_213024__A1.h5",
        )
    )
    pv = peak_detector(w.force, prominence_factors=PROMINENCE_FACTORS, width_factors=WIDTH_FACTORS)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchVelocity(rounded=False, is_contraction=False)
    estimate = metric.fit(pv, w.force, twitch_indices)

    np.testing.assert_array_almost_equal(estimate, expected)


##### TESTS FOR BY-WIDTH METRICS #####
def test_metrics__TwitchWidth():
    file_path = os.path.join(
        PATH_TO_DATA_METRIC_FILES, "v0.3.1", "width_MA201110001__2020_09_03_213024__A1.parquet"
    )
    try:
        table = pq.read_table(file_path)
    except Exception as e:
        raise FileNotFoundError("Parquet file for twitch width not found.") from e
    else:
        expected = table.to_pandas().squeeze()

    w = WellFile(
        os.path.join(
            PATH_TO_H5_FILES,
            "v0.3.1",
            "MA201110001__2020_09_03_213024",
            "MA201110001__2020_09_03_213024__A1.h5",
        )
    )

    pv = peak_detector(w.force)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchWidth()
    estimate = metric.fit(pv, w.force, twitch_indices)

    np.testing.assert_array_almost_equal(estimate, expected)


def test_metrics__TwitchContractionTime():
    file_path = os.path.join(
        PATH_TO_DATA_METRIC_FILES,
        "v0.3.1",
        "contraction_time_MA201110001__2020_09_03_213024__A1.parquet",
    )
    try:
        table = pq.read_table(file_path)
    except Exception as e:
        raise FileNotFoundError("Parquet file for twitch contraction time not found.") from e
    else:
        expected = table.to_pandas().squeeze()

    w = WellFile(
        os.path.join(
            PATH_TO_H5_FILES,
            "v0.3.1",
            "MA201110001__2020_09_03_213024",
            "MA201110001__2020_09_03_213024__A1.h5",
        )
    )
    pv = peak_detector(w.force, prominence_factors=PROMINENCE_FACTORS, width_factors=WIDTH_FACTORS)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchPeakTime(is_contraction=True)
    estimate = metric.fit(pv, w.force, twitch_indices)

    np.testing.assert_array_almost_equal(estimate, expected)


def test_metrics__TwitchRelaxationTime():
    file_path = os.path.join(
        PATH_TO_DATA_METRIC_FILES,
        "v0.3.1",
        "relaxation_time_MA201110001__2020_09_03_213024__A1.parquet",
    )
    try:
        table = pq.read_table(file_path)
    except Exception as e:
        raise FileNotFoundError("Parquet file for twitch relaxation time not found.") from e
    else:
        expected = table.to_pandas().squeeze()

    w = WellFile(
        os.path.join(
            PATH_TO_H5_FILES,
            "v0.3.1",
            "MA201110001__2020_09_03_213024",
            "MA201110001__2020_09_03_213024__A1.h5",
        )
    )
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
