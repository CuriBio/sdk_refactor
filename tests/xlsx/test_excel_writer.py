# -*- coding: utf-8 -*-
import math
import os
from random import choice
from random import randint
import tempfile
from typing import Set

import numpy as np
import pandas as pd
from pulse3D import excel_writer
from pulse3D import magnet_finding
from pulse3D.constants import BASELINE_TO_PEAK_UUID
from pulse3D.constants import CALCULATED_METRIC_DISPLAY_NAMES
from pulse3D.constants import DEFAULT_TWITCH_WIDTHS
from pulse3D.constants import MICRO_TO_BASE_CONVERSION
from pulse3D.constants import PEAK_TO_BASELINE_UUID
from pulse3D.excel_writer import write_xlsx
from pulse3D.plate_recording import PlateRecording
import pytest

from ..fixtures_utils import PATH_TO_H5_FILES
from ..fixtures_utils import TEST_OPTICAL_FILE_96_WELL
from ..fixtures_utils import TEST_OPTICAL_FILE_NO_DUPLICATES
from ..fixtures_utils import TEST_OPTICAL_FILE_ONE_PATH
from ..fixtures_utils import TEST_OPTICAL_FILE_THREE_PATH
from ..fixtures_utils import TEST_OPTICAL_FILE_TWO_PATH
from ..fixtures_utils import TEST_SMALL_BETA_1_FILE_PATH

TEST_FILE_PATH = os.path.join(
    PATH_TO_H5_FILES, "v1.1.0", "ML2022126006_Position 1 Baseline_2022_06_15_004655.zip"
)
TEST_OLD_FILE_WITH_STIM_PROTOCOLS_PATH = os.path.join(
    PATH_TO_H5_FILES, "stim", "ML22001000-2__2022_11_17_233136.zip"
)
TEST_TWO_STIM_SESSIONS_FILE_PATH = os.path.join(
    PATH_TO_H5_FILES, "stim", "StimInterpolationTest-TwoSessions.zip"
)
TEST_NO_STIM_FILE_PATH = os.path.join(PATH_TO_H5_FILES, "stim", "SmallBeta2File-NoStim.zip")


DEFAULT_TWITCH_WIDTH_LABELS = set(
    CALCULATED_METRIC_DISPLAY_NAMES[metric_uuid].format(width)
    for width in DEFAULT_TWITCH_WIDTHS
    for metric_uuid in (BASELINE_TO_PEAK_UUID, PEAK_TO_BASELINE_UUID)
)


def get_per_twitch_labels(df) -> Set[str]:
    return {
        metric_label
        for metric in df.values.tolist()
        if isinstance(metric_label := metric[0], str)
        and ("Time From Contraction" in metric_label or "Time From Peak" in metric_label)
    }


@pytest.fixture(scope="function", name="tmp_dir_for_xlsx", autouse=True)
def fixture_write_to_tmp_dir():
    # save dir before switching to temp dir
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        # switch to temp dir so output file is automatically deleted
        try:
            os.chdir(tmpdir)
            yield tmpdir
        finally:
            # switch dir back to avoid causing issues with other tests
            os.chdir(cwd)


@pytest.fixture(scope="function", name="patch_get_positions")
def fixture_patch_get_positions(mocker):
    def se(x, **kwargs):
        x_len = x.shape[-1]
        test_data = np.empty((x_len, 24))

        for well_idx in range(24):
            test_data[:, well_idx] = (np.arange(x_len) * -((well_idx % 4) + 1)) // 1000
        return {"X": test_data}

    mocker.patch.object(magnet_finding, "get_positions", autospec=True, side_effect=se)


@pytest.mark.slow
def test_write_xlsx__runs_beta_1_file_without_error():
    # Tanner (12/8/22): do not add anything to this test, it is just meant to run a full analysis start to
    # finish with no mocking on a beta 1 file.
    # Any and all param testing should be done in separate tests and make assertions on the xlsx output as is
    # done in the tests below.

    pr = PlateRecording(TEST_SMALL_BETA_1_FILE_PATH)
    output_file_name = write_xlsx(pr)

    # this assertion isn't really necessary, but it's nice to make an assertion in a test that otherwise has none
    assert isinstance(output_file_name, str)


@pytest.mark.slow
def test_write_xlsx__runs_magnet_finding_alg_without_error():
    # Tanner (12/8/22): do not add anything to this test, it is just meant to run a full analysis start to
    # finish with no mocking. This is specifically to make sure that there are no issues with using the magnet
    # finding alg since it is often mocked in other tests to make them run faster.
    # Any and all param testing should be done in separate tests and make assertions on the xlsx output as is
    # done in the tests below.

    pr = PlateRecording(TEST_FILE_PATH)
    output_file_name = write_xlsx(pr, stim_waveform_format="stacked")

    # this assertion isn't really necessary, but it's nice to make an assertion in a test that otherwise has none
    assert isinstance(output_file_name, str)


@pytest.mark.slow
@pytest.mark.parametrize(
    "optical_file",
    [
        TEST_OPTICAL_FILE_ONE_PATH,
        TEST_OPTICAL_FILE_TWO_PATH,
        TEST_OPTICAL_FILE_THREE_PATH,
        TEST_OPTICAL_FILE_NO_DUPLICATES,
        TEST_OPTICAL_FILE_96_WELL,
    ],
)
def test_write_xlsx__runs_optical_file_without_error(optical_file):
    # Tanner (12/8/22): do not add anything to this test, it is just meant to run a full analysis start to
    # finish with no mocking on an optical file.
    # Any and all param testing should be done in separate tests and make assertions on the xlsx output as is
    # done in the tests below.

    pr = PlateRecording(optical_file)
    output_file_name = write_xlsx(pr)

    # this assertion isn't really necessary, but it's nice to make an assertion in a test that otherwise has none
    assert isinstance(output_file_name, str)


@pytest.mark.parametrize("test_normalize_y_axis", [None, True, False])
@pytest.mark.parametrize("test_max_y", [None, randint(1, 1000)])
def test_write_xlsx__sets_tissue_y_axis_correctly_based_on_normalize_y_axis_and_max_y(
    test_normalize_y_axis, test_max_y, patch_get_positions, tmp_dir_for_xlsx, mocker
):
    mocked_create_waveform_charts = mocker.patch.object(excel_writer, "create_waveform_charts", autospec=True)

    pr = PlateRecording(TEST_FILE_PATH)

    kwargs = {}
    if test_normalize_y_axis is not None:
        kwargs["normalize_y_axis"] = test_normalize_y_axis
    if test_max_y is not None:
        kwargs["max_y"] = test_max_y

    output_filename = write_xlsx(pr, **kwargs)
    output_filepath = os.path.join(tmp_dir_for_xlsx, output_filename)

    expected_max = None
    if test_normalize_y_axis is not False:
        if test_max_y:
            expected_max = test_max_y
        else:
            tissue_waveform_data = pd.read_excel(
                output_filepath, sheet_name="continuous-waveforms", usecols=list(range(1, 25))
            )
            expected_max = math.ceil(max([max(tissue_waveform_data[col]) for col in tissue_waveform_data]))

    expected_tissue_chart_bounds = {"max": expected_max, "min": 0}

    for call in mocked_create_waveform_charts.call_args_list:
        assert call[0][0]["tissue"] == expected_tissue_chart_bounds


def test_write_xlsx__raises_error_if_start_time_less_than_zero(patch_get_positions):
    test_start_time = -0.01
    with pytest.raises(ValueError, match=rf"Window start time \({test_start_time}s\) cannot be negative"):
        write_xlsx(PlateRecording(TEST_FILE_PATH), start_time=test_start_time)


@pytest.mark.parametrize("test_start_time", [33.0, 33.1])
def test_write_xlsx__raises_error_if_start_time_greater_than_or_equal_to_final_timepoint_of_recording(
    test_start_time, patch_get_positions
):
    with pytest.raises(
        ValueError,
        match=rf"Window start time \({test_start_time}s\) greater than the max timepoint of this recording \(33.0s\)",
    ):
        write_xlsx(PlateRecording(TEST_FILE_PATH), start_time=test_start_time)


@pytest.mark.parametrize("test_end_time", [10, 9.9])
def test_write_xlsx__raises_error_if_end_time_less_than_or_equal_to_start_time(
    test_end_time, patch_get_positions
):
    with pytest.raises(ValueError, match="Window end time must be greater than window start time"):
        write_xlsx(PlateRecording(TEST_FILE_PATH), start_time=10, end_time=test_end_time)


def test_write_xlsx__uses_default_start_and_end_time_values_correctly(patch_get_positions, tmp_dir_for_xlsx):
    pr = PlateRecording(TEST_FILE_PATH)

    output_filename = write_xlsx(pr)
    assert "full" in output_filename

    output_filepath = os.path.join(tmp_dir_for_xlsx, output_filename)
    tissue_waveform_timepoints = np.array(
        pd.read_excel(output_filepath, sheet_name="continuous-waveforms", usecols=[0])["Time (seconds)"]
    )

    first_well_timepoints_secs = pr.wells[0].force[0] // MICRO_TO_BASE_CONVERSION
    np.testing.assert_almost_equal(tissue_waveform_timepoints[0], first_well_timepoints_secs[0], decimal=1)
    np.testing.assert_almost_equal(tissue_waveform_timepoints[-1], first_well_timepoints_secs[-1], decimal=1)


def test_write_xlsx__uses_custom_start_and_end_time_values_correctly(patch_get_positions, tmp_dir_for_xlsx):
    pr = PlateRecording(TEST_FILE_PATH)

    test_start_time = 5.0
    test_end_time = 10.0

    output_filename = write_xlsx(pr, start_time=test_start_time, end_time=test_end_time)
    assert f"{test_start_time}-{test_end_time}" in output_filename

    output_filepath = os.path.join(tmp_dir_for_xlsx, output_filename)
    tissue_waveform_timepoints = np.array(
        pd.read_excel(output_filepath, sheet_name="continuous-waveforms", usecols=[0])["Time (seconds)"]
    )

    np.testing.assert_almost_equal(tissue_waveform_timepoints[0], test_start_time, decimal=1)
    np.testing.assert_almost_equal(tissue_waveform_timepoints[-1], test_end_time, decimal=1)


@pytest.mark.parametrize(
    "test_start_time,test_end_time, expected_width",
    [[0.0, 33.0, 10], [15.0, 30.0, 10], [5.0, 10.0, 4.99], [25.0, 27.0, 1.9899999999999984]],
)
def test_write_xlsx__correctly_sets_snapshot_width_when_using_custom_start_and_end_time_values(
    patch_get_positions, mocker, test_start_time, test_end_time, expected_width
):
    snapshot_plotter = mocker.spy(excel_writer, "plotting_parameters")

    pr = PlateRecording(TEST_FILE_PATH)
    write_xlsx(pr, start_time=test_start_time, end_time=test_end_time)

    snapshot_plotter.assert_any_call(expected_width)


def test_write_xlsx__correctly_handles_default_twitch_widths(patch_get_positions, tmp_dir_for_xlsx):
    pr = PlateRecording(TEST_FILE_PATH)

    output_file_name = write_xlsx(pr)

    output_filepath = os.path.join(tmp_dir_for_xlsx, output_file_name)
    df = pd.read_excel(output_filepath, sheet_name="per-twitch-metrics", usecols=[0])

    assert get_per_twitch_labels(df) == DEFAULT_TWITCH_WIDTH_LABELS


def test_write_xlsx__correctly_handles_custom_twitch_widths(patch_get_positions, tmp_dir_for_xlsx):
    pr = PlateRecording(TEST_FILE_PATH)

    output_file_name = write_xlsx(pr, twitch_widths=(23, 81), baseline_widths_to_use=(8, 93))

    output_filepath = os.path.join(tmp_dir_for_xlsx, output_file_name)
    df = pd.read_excel(output_filepath, sheet_name="per-twitch-metrics", usecols=[0])

    assert get_per_twitch_labels(df) == {
        "Time From Contraction 8 to Peak (seconds)",
        "Time From Contraction 23 to Peak (seconds)",
        "Time From Contraction 81 to Peak (seconds)",
        "Time From Peak to Relaxation 23 (seconds)",
        "Time From Peak to Relaxation 81 (seconds)",
        "Time From Peak to Relaxation 93 (seconds)",
    }


# TODO add tests for the following params:
#     prominence_factors: Tuple[Union[int, float], Union[int, float]] = DEFAULT_PROMINENCE_FACTORS
#     width_factors: Tuple[Union[int, float], Union[int, float]] = DEFAULT_WIDTH_FACTORS
#     peaks_valleys: Dict[str, List[List[int]]] = None


@pytest.mark.parametrize("test_include_stim_protocols", [None, False])
def test_write_xlsx__correctly_handles_include_stim_protocols_param_when_false(
    test_include_stim_protocols, patch_get_positions, tmp_dir_for_xlsx
):
    pr = PlateRecording(TEST_OLD_FILE_WITH_STIM_PROTOCOLS_PATH)

    kwargs = {}
    if test_include_stim_protocols is not None:
        kwargs["include_stim_protocols"] = test_include_stim_protocols

    output_file_name = write_xlsx(pr, **kwargs)

    output_filepath = os.path.join(tmp_dir_for_xlsx, output_file_name)
    df = pd.read_excel(output_filepath, None)

    # check that all sheets are present except stimulation-protocols sheet
    assert len(df.keys()) == 8


def test_write_xlsx__correctly_writes_to_provided_output_directory(patch_get_positions, tmp_dir_for_xlsx):
    pr = PlateRecording(TEST_NO_STIM_FILE_PATH)

    expected_output_dir = os.path.join(tmp_dir_for_xlsx, "test_subdir")
    os.makedirs(expected_output_dir)

    output_file_name = write_xlsx(pr, output_dir=expected_output_dir)
    output_filepath = os.path.join(expected_output_dir, output_file_name)
    default_path = os.path.join(tmp_dir_for_xlsx, "SmallBeta2File-NoStim_full.xlsx")

    assert os.path.exists(output_filepath)
    assert not os.path.exists(default_path)


def test_write_xlsx__correctly_handles_include_stim_protocols_when_true_for_file_with_stim_protocols(
    patch_get_positions, tmp_dir_for_xlsx
):
    pr = PlateRecording(TEST_OLD_FILE_WITH_STIM_PROTOCOLS_PATH)

    output_file_name = write_xlsx(pr, include_stim_protocols=True)
    output_filepath = os.path.join(tmp_dir_for_xlsx, output_file_name)

    df = pd.read_excel(output_filepath, sheet_name="stimulation-protocols", usecols=[1])
    # check that stimulation-protocols sheet has unassigned well lables
    assert df.keys()[0] == "D5, D6, "
    # check first protocol has correct assigned wells
    assert df["D5, D6, "][4] == "A1, A2, B2, C2, B3, "

    # check second protocol has correct assigned wells
    df = pd.read_excel(output_filepath, sheet_name="stimulation-protocols", usecols=[2])
    assert df[df.keys()[0]][4] == "B1, C1, D1, D2, A3, D3, A4, D4, C5, A6, B6, C6, "

    # check final protocol has correct assigned wells
    df = pd.read_excel(output_filepath, sheet_name="stimulation-protocols", usecols=[3])
    assert df[df.keys()[0]][4] == "C3, B4, C4, A5, B5, "


def test_write_xlsx__correctly_handles_include_stim_protocols_when_true_for_beta_2_file_with_no_stim_protocols(
    patch_get_positions, tmp_dir_for_xlsx, mocker
):
    # mock to speed up test
    mocker.patch.object(excel_writer, "_write_metadata", autospec=True)
    mocker.patch.object(excel_writer, "_write_aggregate_metrics", autospec=True)
    mocker.patch.object(excel_writer, "_write_per_twitch_metrics", autospec=True)
    mocker.patch.object(excel_writer, "_write_continuous_waveforms", autospec=True)

    pr = PlateRecording(TEST_FILE_PATH)

    output_file_name = write_xlsx(pr, include_stim_protocols=True)

    output_filepath = os.path.join(tmp_dir_for_xlsx, output_file_name)
    df = pd.read_excel(output_filepath, sheet_name="stimulation-protocols", usecols=[0])

    # check that stimulation-protocols sheet has correct message
    assert df.keys()[0] == "No stimulation protocols applied"


def test_write_xlsx__correctly_handles_include_stim_protocols_when_true_for_beta_1_file(tmp_dir_for_xlsx):
    pr = PlateRecording(TEST_SMALL_BETA_1_FILE_PATH)

    output_file_name = write_xlsx(pr, include_stim_protocols=True)

    output_filepath = os.path.join(tmp_dir_for_xlsx, output_file_name)
    df = pd.read_excel(output_filepath, sheet_name="stimulation-protocols", usecols=[0])

    # check that stimulation-protocols sheet has correct message
    assert df.keys()[0] == "No stimulation protocols applied"


def test_write_xlsx__raise_error_with_invalid_stim_waveform_format_option(patch_get_positions):
    with pytest.raises(ValueError, match="Invalid stim_waveform_format:"):
        write_xlsx(PlateRecording(TEST_TWO_STIM_SESSIONS_FILE_PATH), stim_waveform_format="bad")


@pytest.mark.parametrize(
    "test_file_path",
    [TEST_NO_STIM_FILE_PATH, TEST_SMALL_BETA_1_FILE_PATH, TEST_OLD_FILE_WITH_STIM_PROTOCOLS_PATH],
)
def test_write_xlsx__ignores_stim_waveform_format_option_with_incompatible_file(
    test_file_path, mocker, patch_get_positions
):
    mocked_write_xlsx_helper = mocker.patch.object(excel_writer, "_write_xlsx", autospec=True)
    write_xlsx(PlateRecording(test_file_path), stim_waveform_format=choice(["overlayed", "stacked"]))
    assert mocked_write_xlsx_helper.call_args[1]["stim_plotting_info"] == {}


@pytest.mark.parametrize("test_normalize_y_axis", [None, True, False])
@pytest.mark.parametrize("test_stim_waveform_format", ["overlayed", "stacked"])
def test_write_xlsx__stim_chart_axis_bounds_set_correctly(
    test_normalize_y_axis, test_stim_waveform_format, patch_get_positions, mocker
):
    mocked_create_waveform_charts = mocker.patch.object(excel_writer, "create_waveform_charts", autospec=True)

    # mock to speed up test
    mocker.patch.object(excel_writer, "_write_metadata", autospec=True)
    mocker.patch.object(excel_writer, "_write_aggregate_metrics", autospec=True)
    mocker.patch.object(excel_writer, "_write_per_twitch_metrics", autospec=True)
    mocker.patch.object(excel_writer, "_write_continuous_waveforms", autospec=True)

    pr = PlateRecording(TEST_TWO_STIM_SESSIONS_FILE_PATH)

    kwargs = {"stim_waveform_format": test_stim_waveform_format}

    if test_normalize_y_axis is not None:
        kwargs["normalize_y_axis"] = test_normalize_y_axis

    write_xlsx(pr, **kwargs)

    if test_normalize_y_axis is not False:
        expected_max = max([max(ss[1]) for wf in pr for ss in wf.stim_sessions])
        expected_min = min([min(ss[1]) for wf in pr for ss in wf.stim_sessions])
        expected_stim_chart_bounds = {"max": expected_max, "min": expected_min}
    else:
        expected_stim_chart_bounds = {"max": None, "min": None}

    for call in mocked_create_waveform_charts.call_args_list:
        assert call[0][0]["stim"] == expected_stim_chart_bounds
        assert call[0][-4]["chart_format"] == test_stim_waveform_format
