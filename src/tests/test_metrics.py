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
import json

from pandas import DataFrame, MultiIndex, Series
import pyarrow.parquet as pq
import numpy as np
import pytest

from pulse3D.constants import ALL_METRICS
from pulse3D.constants import CONTRACTION_TIME_UUID
from pulse3D.constants import RELAXATION_TIME_UUID
from pulse3D.constants import TIME_DIFFERENCE_UUID
from pulse3D.constants import CENTIMILLISECONDS_PER_SECOND
from pulse3D.constants import PRIOR_VALLEY_INDEX_UUID
from pulse3D.constants import SUBSEQUENT_VALLEY_INDEX_UUID
from pulse3D.constants import TIME_VALUE_UUID
from pulse3D.constants import WIDTH_VALUE_UUID
from pulse3D.constants import MICRO_TO_BASE_CONVERSION
from pulse3D.constants import *

import pulse3D.metrics as metrics
from stdlib_utils import get_current_file_abs_directory

from .fixtures import fixture_generic_deserialized_per_twitch_metrics_output_0_3_1
from pulse3D.plate_recording import WellFile
from pulse3D.peak_detection import peak_detector, data_metrics, find_twitch_indices

def get_force_metrics_from_well_file(w: WellFile, metrics_to_create=ALL_METRICS):
    peak_and_valley_indices = peak_detector(w.noise_filtered_magnetic_data)
    return data_metrics(peak_and_valley_indices, w.force)


def encode_dict(d):
    """Recursive function to endocide dictionary for saving as JSON file.

    Args:
        d ([dict]): dictionary of metric values

    Returns:
        result [dict]: encoded dictionary of metric values
    """
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

def decode_arrow(table):
    
    df = table.to_pandas()

    tuples = []
    for index in df.columns:
        metric_uuid = uuid.UUID(index[0])
        try:
            metric_width = int(index[1])
        except:
            metric_width = ''
        tuples.append((metric_uuid, metric_width))
    
    df = DataFrame(data=df.values, 
                   columns = MultiIndex.from_tuples(tuples),
                   index=df.index)

    return df

PATH_OF_CURRENT_FILE = get_current_file_abs_directory()

##### TESTS FOR SCALAR METRICS #####
def test_metrics__TwitchAmplitude():
    file_path = os.path.join(PATH_OF_CURRENT_FILE,
                            "data_metrics",
                            "v0.3.1",
                            "amplitude_MA201110001__2020_09_03_213024__A1.parquet"
                            )
    try:
        table = pq.read_table(file_path)
    except:
        raise FileNotFoundError('Parquet file for twitch amplitude not found.')
    else:
        expected = table.to_pandas().squeeze()

    w = WellFile(os.path.join(PATH_OF_CURRENT_FILE,
                              "h5", "v0.3.1",
                              "MA201110001__2020_09_03_213024",
                              "MA201110001__2020_09_03_213024__A1.h5"))
    pv = peak_detector(w.noise_filtered_magnetic_data)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchAmplitude()
    estimate = metric.fit(pv, w.force, twitch_indices)

    assert np.all(expected == estimate)


def test_metrics__TwitchAUC():
    file_path = os.path.join(PATH_OF_CURRENT_FILE,
                            "data_metrics",
                            "v0.3.1", 
                             "auc_MA201110001__2020_09_03_213024__A1.parquet"
                            )
    try:
        table = pq.read_table(file_path)
    except:
        raise FileNotFoundError('Parquet file for twitch AUC not found.')
    else:
        expected = table.to_pandas().squeeze()

    w = WellFile(os.path.join(PATH_OF_CURRENT_FILE, 
                              "h5", "v0.3.1",
                              "MA201110001__2020_09_03_213024", 
                              "MA201110001__2020_09_03_213024__A1.h5"))
    pv = peak_detector(w.noise_filtered_magnetic_data)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchAUC()
    estimate = metric.fit(pv, w.force, twitch_indices)

    assert np.all(expected == estimate)

def test_metrics__TwitchBaselineToPeak():
    file_path = os.path.join(PATH_OF_CURRENT_FILE,
                            "data_metrics",
                            "v0.3.1", 
                             "baseline_to_peak_MA201110001__2020_09_03_213024__A1.parquet"
                            )
    try:
        table = pq.read_table(file_path)
    except:
        raise FileNotFoundError('Parquet file for full twitch contraction time not found.')
    else:
        expected = table.to_pandas().squeeze()

    w = WellFile(os.path.join(PATH_OF_CURRENT_FILE, 
                              "h5", "v0.3.1",
                              "MA201110001__2020_09_03_213024", 
                              "MA201110001__2020_09_03_213024__A1.h5"))
    pv = peak_detector(w.noise_filtered_magnetic_data)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchPeakToBaseline(is_contraction=True)
    estimate = metric.fit(pv, w.force, twitch_indices)

    assert np.all(expected == estimate)

def test_metrics__TwitchPeakToBaseline():
    file_path = os.path.join(PATH_OF_CURRENT_FILE,
                            "data_metrics",
                            "v0.3.1", 
                             "peak_to_baseline_MA201110001__2020_09_03_213024__A1.parquet"
                            )
    try:
        table = pq.read_table(file_path)
    except:
        raise FileNotFoundError('Parquet file for full twitch relaxation time not found.')
    else:
        expected = table.to_pandas().squeeze()

    w = WellFile(os.path.join(PATH_OF_CURRENT_FILE, 
                              "h5", "v0.3.1",
                              "MA201110001__2020_09_03_213024", 
                              "MA201110001__2020_09_03_213024__A1.h5"))
    pv = peak_detector(w.noise_filtered_magnetic_data)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchPeakToBaseline(is_contraction=False)
    estimate = metric.fit(pv, w.force, twitch_indices)

    assert np.all(expected == estimate)


def test_metrics__TwitchFracAmp():
    file_path = os.path.join(PATH_OF_CURRENT_FILE,
                            "data_metrics",
                            "v0.3.1",
                            "fraction_max_MA201110001__2020_09_03_213024__A1.parquet"
                            )
    try:
        table = pq.read_table(file_path)
    except:
        raise FileNotFoundError('Parquet file for twitch fraction max amplitude not found.')
    else:
        expected = table.to_pandas().squeeze()

    w = WellFile(os.path.join(PATH_OF_CURRENT_FILE, 
                              "h5", "v0.3.1",
                              "MA201110001__2020_09_03_213024", 
                              "MA201110001__2020_09_03_213024__A1.h5"))
    pv = peak_detector(w.noise_filtered_magnetic_data)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchFractionAmplitude()
    estimate = metric.fit(pv, w.force, twitch_indices)

    assert np.all(expected == estimate)


def test_metrics__TwitchFreq():
    file_path = os.path.join(PATH_OF_CURRENT_FILE, 
                             "data_metrics",
                            "v0.3.1", 
                             "twitch_frequency_MA201110001__2020_09_03_213024__A1.parquet"
                            )
    try:
        table = pq.read_table(file_path)
    except:
        raise FileNotFoundError('Parquet file for twitch frequency not found.')
    else:
        expected = table.to_pandas().squeeze()

    w = WellFile(os.path.join(PATH_OF_CURRENT_FILE, 
                              "h5", "v0.3.1",
                              "MA201110001__2020_09_03_213024", 
                              "MA201110001__2020_09_03_213024__A1.h5"))
    pv = peak_detector(w.noise_filtered_magnetic_data)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchFrequency()
    estimate = metric.fit(pv, w.force, twitch_indices)

    assert np.all(expected == estimate)


def test_metrics__TwitchIrregularity():
    file_path = os.path.join(PATH_OF_CURRENT_FILE, 
                             "data_metrics",
                            "v0.3.1", 
                            "irregularity_interval_MA201110001__2020_09_03_213024__A1.parquet"
                            )
    try:
        table = pq.read_table(file_path)
    except:
        raise FileNotFoundError('Parquet file for twitch irregularity interval not found.')
    else:
        expected = table.to_pandas().squeeze()

    w = WellFile(os.path.join(PATH_OF_CURRENT_FILE, 
                              "h5", "v0.3.1",
                              "MA201110001__2020_09_03_213024", 
                              "MA201110001__2020_09_03_213024__A1.h5"))
    pv = peak_detector(w.noise_filtered_magnetic_data)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchIrregularity()
    estimate = metric.fit(pv, w.force, twitch_indices)

    assert np.all((expected == estimate)[1:-1])


def test_metrics__TwitchPeriod():
    file_path = os.path.join(PATH_OF_CURRENT_FILE, 
                             "data_metrics",
                            "v0.3.1", 
                            "twitch_period_MA201110001__2020_09_03_213024__A1.parquet"
                            )
    try:
        table = pq.read_table(file_path)
    except:
        raise FileNotFoundError('Parquet file for twitch period not found.')
    else:
        expected = table.to_pandas().squeeze()

    w = WellFile(os.path.join(PATH_OF_CURRENT_FILE, 
                              "h5", "v0.3.1",
                              "MA201110001__2020_09_03_213024", 
                              "MA201110001__2020_09_03_213024__A1.h5"))
    pv = peak_detector(w.noise_filtered_magnetic_data)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchPeriod()
    estimate = metric.fit(pv, w.force, twitch_indices)

    assert np.all(expected == estimate)


def test_metrics__TwitchContractionVelocity():
    file_path = os.path.join(PATH_OF_CURRENT_FILE, 
                             "data_metrics",
                            "v0.3.1", 
                             "contraction_velocity_MA201110001__2020_09_03_213024__A1.parquet"
                            )
    try:
        table = pq.read_table(file_path)
    except:
        raise FileNotFoundError('Parquet file for twitch contraction velocity not found.')
    else:
        expected = table.to_pandas().squeeze()

    w = WellFile(os.path.join(PATH_OF_CURRENT_FILE, 
                              "h5", "v0.3.1",
                              "MA201110001__2020_09_03_213024", 
                              "MA201110001__2020_09_03_213024__A1.h5"))
    pv = peak_detector(w.noise_filtered_magnetic_data)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchVelocity(rounded=False, is_contraction=True)
    estimate = metric.fit(pv, w.force, twitch_indices)

    assert np.all(expected == estimate)


def test_metrics__TwitchRelaxationVelocity():
    file_path = os.path.join(PATH_OF_CURRENT_FILE, 
                             "data_metrics",
                            "v0.3.1", 
                             "relaxation_velocity_MA201110001__2020_09_03_213024__A1.parquet"
                            )
    try:
        table = pq.read_table(file_path)
    except:
        raise FileNotFoundError('Parquet file for twitch relaxation velocity not found.')
    else:
        expected = table.to_pandas().squeeze()

    w = WellFile(os.path.join(PATH_OF_CURRENT_FILE, 
                              "h5", "v0.3.1",
                              "MA201110001__2020_09_03_213024", 
                              "MA201110001__2020_09_03_213024__A1.h5"))
    pv = peak_detector(w.noise_filtered_magnetic_data)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchVelocity(rounded=False, is_contraction=False)
    estimate = metric.fit(pv, w.force, twitch_indices)

    assert np.all(expected == estimate)


##### TESTS FOR BY-WIDTH METRICS #####
def test_metrics__TwitchWidth():
    file_path = os.path.join(PATH_OF_CURRENT_FILE, 
                             "data_metrics",
                            "v0.3.1", 
                             "width_MA201110001__2020_09_03_213024__A1.parquet"
                            )
    try:
        table = pq.read_table(file_path)
    except:
        raise FileNotFoundError('Parquet file for twitch width not found.')
    else:
        expected = table.to_pandas().squeeze()

    w = WellFile(os.path.join(PATH_OF_CURRENT_FILE, 
                              "h5", "v0.3.1",
                              "MA201110001__2020_09_03_213024", 
                              "MA201110001__2020_09_03_213024__A1.h5"))
    pv = peak_detector(w.noise_filtered_magnetic_data)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchWidth()
    estimate = metric.fit(pv, w.force, twitch_indices)

    assert np.all(expected == estimate)


def test_metrics__TwitchContractionTime():
    file_path = os.path.join(PATH_OF_CURRENT_FILE,
                             "data_metrics",
                             "v0.3.1",
                             "contraction_time_MA201110001__2020_09_03_213024__A1.parquet"
                             )
    try:
        table = pq.read_table(file_path)
    except:
        raise FileNotFoundError('Parquet file for twitch contraction time not found.')
    else:
        expected = table.to_pandas().squeeze()

    w = WellFile(os.path.join(PATH_OF_CURRENT_FILE, 
                              "h5", "v0.3.1",
                              "MA201110001__2020_09_03_213024", 
                              "MA201110001__2020_09_03_213024__A1.h5"))
    pv = peak_detector(w.noise_filtered_magnetic_data)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchPeakTime(is_contraction=True)
    estimate = metric.fit(pv, w.force, twitch_indices)

    assert np.all(expected == estimate)

def test_metrics__TwitchRelaxationTime():
    file_path = os.path.join(PATH_OF_CURRENT_FILE,
                             "data_metrics",
                             "v0.3.1",
                             "relaxation_time_MA201110001__2020_09_03_213024__A1.parquet"
                             )
    try:
        table = pq.read_table(file_path)
    except:
        raise FileNotFoundError(
            'Parquet file for twitch relaxation time not found.')
    else:
        expected = table.to_pandas().squeeze()

    w = WellFile(os.path.join(PATH_OF_CURRENT_FILE, 
                              "h5", "v0.3.1",
                              "MA201110001__2020_09_03_213024", 
                              "MA201110001__2020_09_03_213024__A1.h5"))
    pv = peak_detector(w.noise_filtered_magnetic_data)
    twitch_indices = find_twitch_indices(pv)

    metric = metrics.TwitchPeakTime(is_contraction=False)
    estimate = metric.fit(pv, w.force, twitch_indices)

    assert np.all(expected == estimate)


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
