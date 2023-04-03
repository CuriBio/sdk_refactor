# -*- coding: utf-8 -*-
import logging

from pulse3D.excel_writer import write_xlsx
from pulse3D.plate_recording import PlateRecording

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")

recordings = PlateRecording.from_directory("../../v2Data")

# recordings = PlateRecording.from_directory(
#     "../newxlsx_file/WellB4_1hr-contraction_data_for_sdk (1)"
# )
for r in recordings:
    write_xlsx(r)
