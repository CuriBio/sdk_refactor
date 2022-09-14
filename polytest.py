import logging

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")

from pulse3D.plate_recording import PlateRecording
from pulse3D.excel_writer import write_xlsx

recordings = PlateRecording.from_directory("../optical_files")

for r in recordings:
    write_xlsx(r)
