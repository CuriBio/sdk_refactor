# -*- coding: utf-8 -*-
import csv
import os
from typing import List
from typing import Optional
from typing import Tuple

import matplotlib
import numpy as np
from pulse3D.plate_recording import WellFile
import pytest
from stdlib_utils import get_current_file_abs_directory


matplotlib.use("Agg")
PATH_OF_CURRENT_FILE = get_current_file_abs_directory()


PATH_TO_DATASETS = os.path.join(PATH_OF_CURRENT_FILE, "datasets")
PATH_TO_MAGNET_FINDING_FILES = os.path.join(PATH_OF_CURRENT_FILE, "magnet_finding")
# PATH_TO_PNGS = os.path.join(PATH_OF_CURRENT_FILE, "pngs")


def _load_file(file_path: str) -> Tuple[List[str], List[str]]:
    time = []
    v = []
    header_placer = []  # used to get rid of the header
    with open(file_path, "r") as file_name:
        file_reader = csv.reader(file_name, delimiter=",")
        header = next(file_reader)
        header_placer.append(header)
        for row in file_reader:
            # row variable is a list that represents a row in csv
            time.append(row[0])
            v.append(row[1])
    return time, v


def _load_file_tsv(file_path: str) -> Tuple[List[str], List[str]]:
    time = []
    v = []
    with open(file_path, "r") as file_name:
        file_reader = csv.reader(file_name, delimiter="\t")
        for row in file_reader:
            time.append(row[0])
            v.append(row[1])
    return time, v


def _load_file_h5(
    file_path: str, sampling_rate_construct: int, x_range: Optional[Tuple[int, int]]
) -> Tuple[List[str], List[str]]:
    wf = WellFile(file_path)
    tissue_data = wf.raw_tissue_magnetic_data

    if x_range is None:
        return tissue_data[0], tissue_data[1]

    start = x_range[0] * sampling_rate_construct
    stop = x_range[1] * sampling_rate_construct

    return tissue_data[0][start:stop], tissue_data[1][start:stop]


def create_numpy_array_of_raw_gmr_from_python_arrays(time_array, gmr_array):
    time = np.array(time_array, dtype=np.int32)
    v = np.array(gmr_array, dtype=np.int32)

    data = np.array([time, v], dtype=np.int32)
    return data


def assert_percent_diff(actual, expected, threshold=0.0006):
    percent_diff = abs(actual - expected) / expected
    assert percent_diff < threshold


@pytest.fixture(scope="function", name="raw_generic_well_a1")
def fixture_raw_generic_well_a1():
    time, gmr = _load_file_tsv(os.path.join(PATH_TO_DATASETS, "new_A1_tsv.tsv"))
    raw_gmr_data = create_numpy_array_of_raw_gmr_from_python_arrays(time, gmr)
    raw_gmr_data[0] *= 10
    return raw_gmr_data


@pytest.fixture(scope="function", name="raw_generic_well_a2")
def fixture_raw_generic_well_a2():
    time, gmr = _load_file_tsv(os.path.join(PATH_TO_DATASETS, "new_A2_tsv.tsv"))
    raw_gmr_data = create_numpy_array_of_raw_gmr_from_python_arrays(time, gmr)
    return raw_gmr_data


@pytest.fixture(scope="function", name="sample_tissue_reading")
def fixture_sample_tissue_reading():
    time, gmr = _load_file_tsv(os.path.join(PATH_TO_DATASETS, "sample_tissue_reading.tsv"))
    raw_gmr_data = create_numpy_array_of_raw_gmr_from_python_arrays(time, gmr)
    return raw_gmr_data


@pytest.fixture(scope="function", name="sample_reference_reading")
def fixture_sample_reference_reading():
    time, gmr = _load_file_tsv(os.path.join(PATH_TO_DATASETS, "sample_reference_reading.tsv"))
    raw_gmr_data = create_numpy_array_of_raw_gmr_from_python_arrays(time, gmr)
    return raw_gmr_data
