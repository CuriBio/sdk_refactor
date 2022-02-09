# -*- coding: utf-8 -*-
import os
import tempfile

from pulse3D.excel_writer import write_xlsx
from pulse3D.plate_recording import PlateRecording
import pytest
from stdlib_utils import get_current_file_abs_directory

PATH_OF_CURRENT_FILE = get_current_file_abs_directory()


@pytest.mark.slow
def test_write_xlsx__runs_without_error():
    pr = PlateRecording(
        os.path.join(
            PATH_OF_CURRENT_FILE,
            "h5",
            "v0.3.2",
            "MA20223322__2020_09_02_173919",
        )
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = os.path.join(tmpdir, "test.xlsx")
        write_xlsx(pr, name=test_path)
