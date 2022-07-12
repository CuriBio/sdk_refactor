# -*- coding: utf-8 -*-
import os
import tempfile
import pandas as pd

from pulse3D.excel_writer import write_xlsx
from pulse3D.plate_recording import PlateRecording
import pytest
from stdlib_utils import get_current_file_abs_directory

PATH_OF_CURRENT_FILE = get_current_file_abs_directory()


# @pytest.mark.slow
# def test_write_xlsx__runs_without_error():
#     pr = PlateRecording(
#         os.path.join(
#             PATH_OF_CURRENT_FILE,
#             "h5",
#             "v1.1.0",
#             "ML2022126006_Position 1 Baseline_2022_06_15_004655.zip",
#         )
#     )

#     # save dir before switching to temp dir
#     cwd = os.getcwd()
#     with tempfile.TemporaryDirectory() as tmpdir:
#         # switch to temp dir so output file is automatically deleted
#         os.chdir(tmpdir)
#         output_file_name = write_xlsx(pr)

#         output_filepath = os.path.join(tmpdir, output_file_name)
#         # column = pd.read_excel(output_filepath, index_col=None, usecols=[0])

#         df = pd.read_excel(output_filepath, sheet_name="per-twitch-metrics", usecols=[0])

#         # by default, should contain three contraction to peak metrics and two peak to relaxation metrics
#         for metric in (
#             "Time From Contraction 10 to Peak (seconds)",
#             "Time From Contraction 50 to Peak (seconds)",
#             "Time From Contraction 90 to Peak (seconds)",
#             "Time From Peak to Relaxation 50 (seconds)",
#             "Time From Peak to Relaxation 90 (seconds)",
#         ):
#             assert [metric] in df.values.tolist()

#         # switch dir back to avoid causing issues with other tests
#         os.chdir(cwd)

#     assert isinstance(output_file_name, str)


@pytest.mark.slow
def test_write_xlsx__correctly_handles_new_twitch_widths():
    pr = PlateRecording(
        os.path.join(
            PATH_OF_CURRENT_FILE,
            "h5",
            "v1.1.0",
            "ML2022126006_Position 1 Baseline_2022_06_15_004655.zip",
        )
    )

    # save dir before switching to temp dir
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        # switch to temp dir so output file is automatically deleted
        os.chdir(tmpdir)
        output_file_name = write_xlsx(pr, twitch_widths=(25, 75), baseline_widths_to_use=(5, 95))

        output_filepath = os.path.join(tmpdir, output_file_name)
        # column = pd.read_excel(output_filepath, index_col=None, usecols=[0])

        df = pd.read_excel(output_filepath, sheet_name="per-twitch-metrics", usecols=[0])

        # by default, should contain three contraction to peak metrics and two peak to relaxation metrics
        for metric in (
            "Time From Contraction 10 to Peak (seconds)",
            "Time From Contraction 50 to Peak (seconds)",
            "Time From Contraction 90 to Peak (seconds)",
            "Time From Peak to Relaxation 50 (seconds)",
            "Time From Peak to Relaxation 90 (seconds)",
        ):
            assert [metric] not in df.values.tolist()
        
        for metric in (
            "Time From Contraction 5 to Peak (seconds)",
            "Time From Contraction 25 to Peak (seconds)",
            "Time From Contraction 75 to Peak (seconds)",
            "Time From Peak to Relaxation 25 (seconds)",
            "Time From Peak to Relaxation 75 (seconds)",
            "Time From Peak to Relaxation 95 (seconds)",
        ):
            assert [metric] in df.values.tolist()

        # switch dir back to avoid causing issues with other tests
        os.chdir(cwd)

    assert isinstance(output_file_name, str)
