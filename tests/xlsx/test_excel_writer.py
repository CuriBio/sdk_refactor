# -*- coding: utf-8 -*-
import os
import tempfile
from typing import Set

import numpy as np
import pandas as pd
from pulse3D import magnet_finding
from pulse3D.constants import BASELINE_TO_PEAK_UUID
from pulse3D.constants import CALCULATED_METRIC_DISPLAY_NAMES
from pulse3D.constants import DEFAULT_TWITCH_WIDTHS
from pulse3D.constants import PEAK_TO_BASELINE_UUID
from pulse3D.excel_writer import write_xlsx
from pulse3D.plate_recording import PlateRecording
import pytest

from ..fixtures_utils import PATH_TO_H5_FILES


TEST_FILE_PATH = os.path.join(
    PATH_TO_H5_FILES,
    "v1.1.0",
    "ML2022126006_Position 1 Baseline_2022_06_15_004655.zip",
)


DEFAULT_TWITCH_WIDTH_LABELS = set(
    CALCULATED_METRIC_DISPLAY_NAMES[metric_uuid].format(width)
    for width in DEFAULT_TWITCH_WIDTHS
    for metric_uuid in (BASELINE_TO_PEAK_UUID, PEAK_TO_BASELINE_UUID)
)

# TODO add tests for the following params:
#     normalize_y_axis: bool = True
#     max_y: Union[int, float] = None
#     start_time: Union[float, int] = 0
#     end_time: Union[float, int] = np.inf
#     prominence_factors: Tuple[Union[int, float], Union[int, float]] = DEFAULT_PROMINENCE_FACTORS
#     width_factors: Tuple[Union[int, float], Union[int, float]] = DEFAULT_WIDTH_FACTORS
#     peaks_valleys: Dict[str, List[List[int]]] = None


def get_per_twitch_labels(df) -> Set[str]:
    return {
        metric_label
        for metric in df.values.tolist()
        if isinstance(metric_label := metric[0], str)
        and ("Time From Contraction" in metric_label or "Time From Peak" in metric_label)
    }


@pytest.mark.slow
def test_write_xlsx__runs_without_error():
    # Tanner (10/4/22): do not add anything to this test, it is just meant to run a full analysis start to finish. Any and all param testing should be done in separate tests as is done below

    pr = PlateRecording(TEST_FILE_PATH)

    # save dir before switching to temp dir
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        # switch to temp dir so output file is automatically deleted
        os.chdir(tmpdir)

        output_file_name = write_xlsx(pr)

        # switch dir back to avoid causing issues with other tests
        os.chdir(cwd)

    assert isinstance(output_file_name, str)


def test_write_xlsx__correctly_handles_default_twitch_widths(mocker):
    # mock so slow function doesn't actually run
    mocker.patch.object(
        magnet_finding,
        "get_positions",
        autospec=True,
        side_effect=lambda x, **kwargs: {"X": np.zeros((x.shape[-1], 24))},
    )

    pr = PlateRecording(TEST_FILE_PATH)

    # save dir before switching to temp dir
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        # switch to temp dir so output file is automatically deleted
        os.chdir(tmpdir)

        output_file_name = write_xlsx(pr)

        output_filepath = os.path.join(tmpdir, output_file_name)
        df = pd.read_excel(output_filepath, sheet_name="per-twitch-metrics", usecols=[0])

        assert get_per_twitch_labels(df) == DEFAULT_TWITCH_WIDTH_LABELS

        # switch dir back to avoid causing issues with other tests
        os.chdir(cwd)


def test_write_xlsx__correctly_handles_custom_twitch_widths(mocker):
    # mock so slow function doesn't actually run
    mocker.patch.object(
        magnet_finding,
        "get_positions",
        autospec=True,
        side_effect=lambda x, **kwargs: {"X": np.zeros((x.shape[-1], 24))},
    )

    pr = PlateRecording(TEST_FILE_PATH)

    # save dir before switching to temp dir
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        # switch to temp dir so output file is automatically deleted
        os.chdir(tmpdir)
        output_file_name = write_xlsx(pr, twitch_widths=(23, 81), baseline_widths_to_use=(8, 93))

        output_filepath = os.path.join(tmpdir, output_file_name)
        df = pd.read_excel(output_filepath, sheet_name="per-twitch-metrics", usecols=[0])

        assert get_per_twitch_labels(df) == {
            "Time From Contraction 8 to Peak (seconds)",
            "Time From Contraction 23 to Peak (seconds)",
            "Time From Contraction 81 to Peak (seconds)",
            "Time From Peak to Relaxation 23 (seconds)",
            "Time From Peak to Relaxation 81 (seconds)",
            "Time From Peak to Relaxation 93 (seconds)",
        }

        # switch dir back to avoid causing issues with other tests
        os.chdir(cwd)


@pytest.mark.parametrize("test_value", [True, False])
def test_include_stim_protocols_runs_without_error(mocker, test_value):
    # mock so slow function doesn't actually run
    mocker.patch.object(
        magnet_finding,
        "get_positions",
        autospec=True,
        side_effect=lambda x, **kwargs: {"X": np.zeros((x.shape[-1], 24))},
    )

    pr = PlateRecording(TEST_FILE_PATH)

    # save dir before switching to temp dir
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        # switch to temp dir so output file is automatically deleted
        os.chdir(tmpdir)
        output_file_name = write_xlsx(pr, include_stim_protocols=test_value)

        # switch dir back to avoid causing issues with other tests
        os.chdir(cwd)

    assert isinstance(output_file_name, str)
