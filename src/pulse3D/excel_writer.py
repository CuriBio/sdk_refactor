# -*- coding: utf-8 -*-
import datetime
import logging

import numpy as np
import pandas as pd
from scipy import interpolate

from .constants import *
from .exceptions import *
from .metrics import TWITCH_WIDTH_PERCENTS
from .peak_detection import data_metrics
from .peak_detection import find_twitch_indices
from .peak_detection import peak_detector
from .utils import xl_col_to_name

INTERPOLATED_DATA_PERIOD_SECONDS = 1 / 100
INTERPOLATED_DATA_PERIOD_US = INTERPOLATED_DATA_PERIOD_SECONDS * MICRO_TO_BASE_CONVERSION

DEFAULT_CELL_WIDTH = 64
CHART_HEIGHT = 300
CHART_BASE_WIDTH = 120
CHART_HEIGHT_CELLS = 15
CHART_FIXED_WIDTH_CELLS = 8
CHART_FIXED_WIDTH = DEFAULT_CELL_WIDTH * CHART_FIXED_WIDTH_CELLS
PEAK_VALLEY_COLUMN_START = 100
CHART_WINDOW_NUM_SECONDS = 10
CHART_WINDOW_NUM_DATA_POINTS = CHART_WINDOW_NUM_SECONDS / INTERPOLATED_DATA_PERIOD_SECONDS
SECONDS_PER_CELL = 2.5

log = logging.getLogger(__name__)


def add_peak_detection_series(
    waveform_charts,
    continuous_waveform_sheet,
    detector_type: str,
    well_index: int,
    well_name: str,
    upper_x_bound_cell: int,
    indices,
    interpolated_data_function: interpolate.interpolate.interp1d,
    time_values,
    is_optical_recording,
    minimum_value: float,
) -> None:
    label = "Relaxation" if detector_type == "Valley" else "Contraction"
    offset = 1 if detector_type == "Valley" else 0
    marker_color = "#D95F02" if detector_type == "Valley" else "#7570B3"

    result_column = xl_col_to_name(PEAK_VALLEY_COLUMN_START + (well_index * 2) + offset)
    continuous_waveform_sheet.write(f"{result_column}1", f"{well_name} {detector_type} Values")

    for idx in indices:
        idx_time = time_values[idx] / MICRO_TO_BASE_CONVERSION
        shifted_idx_time = idx_time - time_values[0] / MICRO_TO_BASE_CONVERSION

        uninterpolated_time_seconds = round(idx_time, 2)
        shifted_time_seconds = round(shifted_idx_time, 2)

        if is_optical_recording:
            row = int(shifted_time_seconds * MICRO_TO_BASE_CONVERSION / INTERPOLATED_DATA_PERIOD_US)

            value = (
                interpolated_data_function(uninterpolated_time_seconds * MICRO_TO_BASE_CONVERSION)
                - minimum_value
            ) * MICRO_TO_BASE_CONVERSION
        else:
            row = shifted_time_seconds * int(1 / INTERPOLATED_DATA_PERIOD_SECONDS) + 1

            interpolated_data = interpolated_data_function(
                uninterpolated_time_seconds * MICRO_TO_BASE_CONVERSION
            )

            interpolated_data -= minimum_value
            value = interpolated_data * MICRO_TO_BASE_CONVERSION

        continuous_waveform_sheet.write(f"{result_column}{row}", value)

    if waveform_charts is not None:  # Tanner (11/11/20): chart is None when skipping chart creation
        for chart in waveform_charts:
            chart.add_series(
                {
                    "name": label,
                    "categories": f"='continuous-waveforms'!$A$2:$A${upper_x_bound_cell}",
                    "values": f"='continuous-waveforms'!${result_column}$2:${result_column}${upper_x_bound_cell}",
                    "marker": {
                        "type": "circle",
                        "size": 8,
                        "border": {"color": marker_color, "width": 1.5},
                        "fill": {"none": True},
                    },
                    "line": {"none": True},
                }
            )


def create_force_frequency_relationship_charts(
    force_frequency_sheet,
    force_frequency_chart,
    well_index: int,
    well_name: str,
    num_data_points: int,
    num_per_twitch_metrics,
) -> None:
    well_row = well_index * num_per_twitch_metrics
    last_column = xl_col_to_name(num_data_points)

    force_frequency_chart.add_series(
        {
            "categories": f"='{PER_TWITCH_METRICS_SHEET_NAME}'!$B${well_row + 7}:${last_column}${well_row + 7}",
            "values": f"='{PER_TWITCH_METRICS_SHEET_NAME}'!$B${well_row + 5}:${last_column}${well_row + 5}",
            "marker": {
                "type": "diamond",
                "size": 7,
            },
            "line": {"none": True},
        }
    )

    force_frequency_chart.set_legend({"none": True})
    x_axis_label = CALCULATED_METRIC_DISPLAY_NAMES[TWITCH_FREQUENCY_UUID]

    force_frequency_chart.set_x_axis({"name": x_axis_label})
    y_axis_label = CALCULATED_METRIC_DISPLAY_NAMES[AMPLITUDE_UUID]

    force_frequency_chart.set_y_axis({"name": y_axis_label, "major_gridlines": {"visible": 0}})
    force_frequency_chart.set_size({"width": CHART_FIXED_WIDTH, "height": CHART_HEIGHT})
    force_frequency_chart.set_title({"name": f"Well {well_name}"})

    well_row, well_col = TWENTY_FOUR_WELL_PLATE.get_row_and_column_from_well_index(well_index)

    force_frequency_sheet.insert_chart(
        1 + well_row * (CHART_HEIGHT_CELLS + 1),
        1 + well_col * (CHART_FIXED_WIDTH_CELLS + 1),
        force_frequency_chart,
    )


def create_frequency_vs_time_charts(
    frequency_chart_sheet,
    frequency_chart,
    well_index: int,
    well_name: str,
    num_data_points: int,
    time_values,
    num_per_twitch_metrics,
) -> None:
    well_row = well_index * num_per_twitch_metrics
    last_column = xl_col_to_name(num_data_points)

    frequency_chart.add_series(
        {
            "categories": f"='{PER_TWITCH_METRICS_SHEET_NAME}'!$B${well_row + 2}:${last_column}${well_row + 2}",
            "values": f"='{PER_TWITCH_METRICS_SHEET_NAME}'!$B${well_row + 7}:${last_column}${well_row + 7}",
            "marker": {
                "type": "diamond",
                "size": 7,
            },
            "line": {"none": True},
        }
    )

    frequency_chart.set_legend({"none": True})

    x_axis_settings: Dict[str, Any] = {"name": "Time (seconds)"}
    x_axis_settings["min"] = 0
    x_axis_settings["max"] = time_values[-1] // MICRO_TO_BASE_CONVERSION

    frequency_chart.set_x_axis(x_axis_settings)

    y_axis_label = CALCULATED_METRIC_DISPLAY_NAMES[TWITCH_FREQUENCY_UUID]

    frequency_chart.set_y_axis({"name": y_axis_label, "min": 0, "major_gridlines": {"visible": 0}})

    frequency_chart.set_size({"width": CHART_FIXED_WIDTH, "height": CHART_HEIGHT})
    frequency_chart.set_title({"name": f"Well {well_name}"})

    well_row, well_col = TWENTY_FOUR_WELL_PLATE.get_row_and_column_from_well_index(well_index)

    frequency_chart_sheet.insert_chart(
        1 + well_row * (CHART_HEIGHT_CELLS + 1),
        1 + well_col * (CHART_FIXED_WIDTH_CELLS + 1),
        frequency_chart,
    )


def write_xlsx(plate_recording, name=None):
    # get metadata from first well file
    w = [pw for pw in plate_recording if pw][0]
    if name is None:
        name = f"{w[PLATE_BARCODE_UUID]}__{w[UTC_BEGINNING_RECORDING_UUID].strftime('%Y_%m_%d_%H%M%S')}.xlsx"

    metadata = {
        "A": [
            "Recording Information:",
            "",
            "",
            "",
            "Device Information:",
            "",
            "",
            "",
            "",
            "",
            "Output Format:",
            "",
            "",
        ],
        "B": [
            "",
            "Plate Barcode",
            "UTC Timestamp of Beginning of Recording",
            "",
            "",
            "H5 File Layout Version",
            "Mantarray Serial Number",
            "Software Release Version",
            "Software Build Version",
            "Firmware Version (Main Controller)",
            "",
            "SDK Version",
            "File Creation Timestamp",
        ],
        "C": [
            "",
            w[PLATE_BARCODE_UUID],
            str(w[UTC_BEGINNING_RECORDING_UUID].replace(tzinfo=None)),
            "",
            "",
            w["File Format Version"],
            w.get(MANTARRAY_SERIAL_NUMBER_UUID, ""),
            w.get(SOFTWARE_RELEASE_VERSION_UUID, ""),
            w.get(SOFTWARE_BUILD_NUMBER_UUID, ""),
            w.get(MAIN_FIRMWARE_VERSION_UUID, ""),
            "",
            PACKAGE_VERSION,
            str(datetime.datetime.utcnow().replace(microsecond=0)),
        ],
    }
    metadata_df = pd.DataFrame(metadata)

    # get max_time from all wells
    max_time = max([w.raw_tissue_magnetic_data[0][-1] for w in plate_recording if w])
    interpolated_data_period = (
        w[INTERPOLATION_VALUE_UUID] if plate_recording.is_optical_recording else INTERPOLATED_DATA_PERIOD_US
    )
    time_points = np.arange(interpolated_data_period, max_time, interpolated_data_period)

    data = []
    # calculate metrics for each well
    for i, well_file in enumerate(plate_recording):
        error_msg = None
        peaks_and_valleys = None
        twitch_indices = None
        metrics = None

        if well_file is None:
            continue

        well_index = well_file[WELL_INDEX_UUID]
        well_name = TWENTY_FOUR_WELL_PLATE.get_well_name_from_well_index(well_index)

        try:
            log.info(f"Finding peaks and valleys for well {well_name}")
            peaks_and_valleys = peak_detector(well_file.noise_filtered_magnetic_data)

            log.info(f"Finding twitch indices for well {well_name}")
            twitch_indices = find_twitch_indices(peaks_and_valleys)

            log.info(f"Calculating metrics for well {well_name}")
            metrics = data_metrics(peaks_and_valleys, well_file.force)
        except TwoPeaksInARowError:
            error_msg = "Error: Two Contractions in a Row Detected"
        except TwoValleysInARowError:
            error_msg = "Error: Two Relaxations in a Row Detected"
        except TooFewPeaksDetectedError:
            error_msg = "Not Enough Twitches Detected"
        except Exception as e:
            raise NotImplementedError("Unknown PeakDetectionError") from e

        first_idx, last_idx = 0, len(time_points) - 1
        while well_file.force[0][-1] < time_points[last_idx]:
            last_idx -= 1

        while well_file.force[0][0] > time_points[first_idx]:
            first_idx += 1

        interp_data_fn = interpolate.interp1d(well_file.force[0], well_file.force[1])

        # normalize and scale data
        interp_data = interp_data_fn(time_points[first_idx:last_idx])
        min_value = min(interp_data)
        interp_data -= min_value
        interp_data *= MICRO_TO_BASE_CONVERSION

        data.append(
            {
                "error_msg": error_msg,
                "peaks_and_valleys": peaks_and_valleys,
                "twitch_indices": twitch_indices,
                "metrics": metrics,
                "well_index": well_index,
                "well_name": TWENTY_FOUR_WELL_PLATE.get_well_name_from_well_index(well_index),
                "max_time": max_time,
                "min_value": min_value,
                "interp_data": interp_data,
                "interp_data_fn": interp_data_fn,
                "force": well_file.force,
                "time_points": time_points,
                "num_data_points": len(well_file.force[0, first_idx:last_idx]),
            }
        )

    # waveform table
    continuous_waveforms = {"Time (seconds)": time_points[first_idx:last_idx] / MICRO_TO_BASE_CONVERSION}
    for d in data:
        continuous_waveforms[f"{d['well_name']} - Active Twitch Force (μN)"] = pd.Series(d["interp_data"])
    continuous_waveforms_df = pd.DataFrame(continuous_waveforms)

    _write_xlsx(name, metadata_df, continuous_waveforms_df, data, plate_recording.is_optical_recording)
    log.info("Done")


def _write_xlsx(name: str, metadata_df, continuous_waveforms_df, data, is_optical_recording):
    with pd.ExcelWriter(name) as writer:
        log.info(f"Writing H5 file metadata")
        metadata_df.to_excel(writer, sheet_name="metadata", index=False, header=False)
        ws = writer.sheets["metadata"]

        for i_col_idx, i_col_width in ((0, 25), (1, 40), (2, 25)):
            ws.set_column(i_col_idx, i_col_idx, i_col_width)

        log.info(f"Creating waveform data sheet")
        continuous_waveforms_df.to_excel(writer, sheet_name="continuous-waveforms", index=False)
        continuous_waveforms_sheet = writer.sheets["continuous-waveforms"]
        continuous_waveforms_sheet.set_column(0, 0, 18)

        for iter_well_idx in range(1, 24):
            continuous_waveforms_sheet.set_column(iter_well_idx, iter_well_idx, 13)

        # waveform snapshot/full
        wb = writer.book
        snapshot_sheet = wb.add_worksheet("continuous-waveform-snapshot")
        full_sheet = wb.add_worksheet("full-continuous-waveform-plots")

        for i, dm in enumerate(data):
            log.info(f'Creating waveform charts for well {dm["well_name"]}')
            create_waveform_charts(
                i,
                dm,
                continuous_waveforms_df,
                wb,
                continuous_waveforms_sheet,
                snapshot_sheet,
                full_sheet,
                is_optical_recording,
            )

        # aggregate metrics sheet
        log.info("Creating aggregate metrics data sheet")
        aggregate_df = aggregate_metrics_df(data)
        aggregate_df.to_excel(writer, sheet_name="aggregate-metrics", index=False, header=False)

        # per twitch metrics sheet
        log.info("Creating per-twitch metrics data sheet")
        (pdf, num_metrics) = per_twitch_df(data)
        pdf.to_excel(writer, sheet_name="per-twitch-metrics", index=False, header=False)

        # freq/force charts
        force_freq_sheet = wb.add_worksheet(FORCE_FREQUENCY_RELATIONSHIP_SHEET)
        freq_vs_time_sheet = wb.add_worksheet(TWITCH_FREQUENCIES_CHART_SHEET_NAME)

        for i, d in enumerate(data):
            dm = d["metrics"]
            if dm:
                force_freq_chart = wb.add_chart({"type": "scatter", "subtype": "straight"})
                freq_vs_time_chart = wb.add_chart({"type": "scatter", "subtype": "straight"})

                log.info(f'Creating frequency vs time chart for well {d["well_name"]}')
                create_frequency_vs_time_charts(
                    freq_vs_time_sheet,
                    freq_vs_time_chart,
                    i,
                    d["well_name"],
                    dm[1][AMPLITUDE_UUID]["n"],  # number of twitches
                    list(dm[0]),  # time values
                    num_metrics,
                )

                log.info(f'Creating force frequency relationship chart for well {d["well_name"]}')
                create_force_frequency_relationship_charts(
                    force_freq_sheet,
                    force_freq_chart,
                    i,
                    d["well_name"],
                    dm[1][AMPLITUDE_UUID]["n"],  # number of twitches
                    num_metrics,
                )
        log.info(f"Writing {name}")


def create_waveform_charts(
    iter_idx,
    dm,
    continuous_waveforms_df,
    wb,
    continuous_waveforms_sheet,
    snapshot_sheet,
    full_sheet,
    is_optical_recording,
):
    recording_stop_time = dm["time_points"][-1] // MICRO_TO_BASE_CONVERSION
    lower_x_bound = (
        0
        if recording_stop_time <= CHART_WINDOW_NUM_SECONDS
        else int((recording_stop_time - CHART_WINDOW_NUM_SECONDS) // 2)
    )

    upper_x_bound = (
        recording_stop_time
        if recording_stop_time <= CHART_WINDOW_NUM_SECONDS
        else int((recording_stop_time + CHART_WINDOW_NUM_SECONDS) // 2)
    )

    df_column = continuous_waveforms_df.columns.get_loc(f"{dm['well_name']} - Active Twitch Force (μN)")

    well_column = xl_col_to_name(df_column)
    full_chart = wb.add_chart({"type": "scatter", "subtype": "straight"})
    snapshot_chart = wb.add_chart({"type": "scatter", "subtype": "straight"})

    snapshot_chart.set_x_axis({"name": "Time (seconds)", "min": lower_x_bound, "max": upper_x_bound})
    snapshot_chart.set_y_axis({"name": "Active Twitch Force (μN)", "major_gridlines": {"visible": 0}})
    snapshot_chart.set_size({"width": CHART_FIXED_WIDTH, "height": CHART_HEIGHT})
    snapshot_chart.set_title({"name": f"Well {dm['well_name']}"})

    # full_chart.set_x_axis({"name": "Time (seconds)", "min": 0, "max": continuous_waveforms_df['Time (seconds)'].iloc[-1]})
    full_chart.set_x_axis({"name": "Time (seconds)", "min": 0, "max": recording_stop_time})
    full_chart.set_y_axis({"name": "Active Twitch Force (μN)", "major_gridlines": {"visible": 0}})

    full_chart.set_size(
        {
            "width": CHART_FIXED_WIDTH // 2
            + (DEFAULT_CELL_WIDTH * int(recording_stop_time / SECONDS_PER_CELL)),
            "height": CHART_HEIGHT,
        }
    )
    full_chart.set_title({"name": f"Well {dm['well_name']}"})

    snapshot_chart.add_series(
        {
            "name": "Waveform Data",
            "categories": f"='continuous-waveforms'!$A$2:$A${len(continuous_waveforms_df)}",
            "values": f"='continuous-waveforms'!${well_column}$2:${well_column}${len(continuous_waveforms_df)}",
            "line": {"color": "#1B9E77"},
        }
    )

    full_chart.add_series(
        {
            "name": "Waveform Data",
            "categories": f"='continuous-waveforms'!$A$2:$A${len(continuous_waveforms_df)}",
            "values": f"='continuous-waveforms'!${well_column}$2:${well_column}${len(continuous_waveforms_df)}",
            "line": {"color": "#1B9E77"},
        }
    )

    (peaks, valleys) = dm["peaks_and_valleys"]
    log.info(f'Adding peak detection series for well {dm["well_name"]}')
    add_peak_detection_series(
        [snapshot_chart, full_chart],
        continuous_waveforms_sheet,
        "Peak",
        iter_idx,
        f"{dm['well_name']}",
        dm["num_data_points"],
        peaks,
        dm["interp_data_fn"],
        dm["force"][0],
        is_optical_recording,
        dm["min_value"],
    )
    add_peak_detection_series(
        [snapshot_chart, full_chart],
        continuous_waveforms_sheet,
        "Valley",
        iter_idx,
        f"{dm['well_name']}",
        dm["num_data_points"],
        valleys,
        dm["interp_data_fn"],
        dm["force"][0],
        is_optical_recording,
        dm["min_value"],
    )

    (well_row, well_col) = TWENTY_FOUR_WELL_PLATE.get_row_and_column_from_well_index(df_column - 1)
    snapshot_sheet.insert_chart(
        well_row * (CHART_HEIGHT_CELLS + 1), well_col * (CHART_FIXED_WIDTH_CELLS + 1), snapshot_chart
    )
    full_sheet.insert_chart(1 + iter_idx * (CHART_HEIGHT_CELLS + 1), 1, full_chart)


def aggregate_metrics_df(data):
    #dms = [d["metrics"] for d in data if d["metrics"]]

    df = pd.DataFrame()
    df = df.append(pd.Series(["", ""] + [d["well_name"] for d in data]), ignore_index=True)
    df = df.append(pd.Series(["", "Treatment Description"]), ignore_index=True)
    df = df.append(
        pd.Series(["", "n (twitches)"] + ["" if d["error_msg"] else len(d["metrics"][0]) for d in data]),
        ignore_index=True,
    )
    df = df.append(
        pd.Series(["", ""] + [d["error_msg"] if d["error_msg"] else "" for d in data]), ignore_index=True
    )  # empty row

    for m in ALL_METRICS:
        if m in [WIDTH_UUID, RELAXATION_TIME_UUID, CONTRACTION_TIME_UUID]:
            #for k in dms[0][1][m].keys():
            for k in TWITCH_WIDTH_PERCENTS:
                nm = CALCULATED_METRIC_DISPLAY_NAMES[m].format(k)

                series_mean = ["N/A" if d["error_msg"] else d["metrics"][1][m][k]["mean"] for d in data]
                df = df.append(pd.Series([nm, "Mean"] + series_mean), ignore_index=True)

                series_std = ["N/A" if d["error_msg"] else d["metrics"][1][m][k]["std"] for d in data]
                df = df.append(pd.Series(["", "StDev"] + series_std), ignore_index=True)

                series_cov = ["N/A" if d["error_msg"] else d["metrics"][1][m][k]["cov"] for d in data]
                df = df.append(pd.Series(["", "CoV"] + series_cov), ignore_index=True)

                series_sem = ["N/A" if d["error_msg"] else d["metrics"][1][m][k]["sem"] for d in data]
                df = df.append(pd.Series(["", "SEM"] + series_sem), ignore_index=True)

                series_min = ["N/A" if d["error_msg"] else d["metrics"][1][m][k]["min"] for d in data]
                df = df.append(pd.Series(["", "Min"] + series_min), ignore_index=True)

                series_max = ["N/A" if d["error_msg"] else d["metrics"][1][m][k]["max"] for d in data]
                df = df.append(pd.Series(["", "Max"] + series_max), ignore_index=True)

                # empty row
                df = df.append(pd.Series([""]), ignore_index=True)
        else:
            nm = CALCULATED_METRIC_DISPLAY_NAMES[m]
            series_mean = ["N/A" if d["error_msg"] else d["metrics"][1][m]["mean"] for d in data]
            df = df.append(pd.Series([nm, "Mean"] + series_mean), ignore_index=True)

            series_std = ["N/A" if d["error_msg"] else d["metrics"][1][m]["std"] for d in data]
            df = df.append(pd.Series(["", "StDev"] + series_std), ignore_index=True)

            series_cov = ["N/A" if d["error_msg"] else d["metrics"][1][m]["cov"] for d in data]
            df = df.append(pd.Series(["", "CoV"] + series_cov), ignore_index=True)

            series_sem = ["N/A" if d["error_msg"] else d["metrics"][1][m]["sem"] for d in data]
            df = df.append(pd.Series(["", "SEM"] + series_sem), ignore_index=True)

            series_min = ["N/A" if d["error_msg"] else d["metrics"][1][m]["min"] for d in data]
            df = df.append(pd.Series(["", "Min"] + series_min), ignore_index=True)

            series_max = ["N/A" if d["error_msg"] else d["metrics"][1][m]["max"] for d in data]
            df = df.append(pd.Series(["", "Max"] + series_max), ignore_index=True)

            # empty row
            df = df.append(pd.Series([""]), ignore_index=True)

    return df


def per_twitch_df(data):
    dms = [d["metrics"] for d in data if not d["error_msg"]]

    idx = list(dms[0][0].keys())[0]
    keys = list(dms[0][0][idx].keys())
    num_per_twitch_metrics = 0  # len(labels)

    df = pd.DataFrame()
    for j, d in enumerate(data):  # for each well
        num_per_twitch_metrics = 0  # len(labels)

        dm = d["metrics"]

        tw_series = [f"Twitch {i+1}" for i in range(len(dm[0]))] if dm else []
        df = df.append(pd.Series([d["well_name"]] + tw_series), ignore_index=True)

        tp_series = [k / MICRO_TO_BASE_CONVERSION for k in dm[0].keys()] if dm else ["N/A"]
        df = df.append(pd.Series(["Timepoint of Twitch Contraction"] + tp_series), ignore_index=True)

        num_per_twitch_metrics += 2

        for m in ALL_METRICS:
            if m in keys:
                if m == WIDTH_UUID:
                    if d["error_msg"]:
                        for q in TWITCH_WIDTH_PERCENTS:
                            values = [f"{CALCULATED_METRIC_DISPLAY_NAMES[m].format(q)}"]
                            df = df.append(pd.Series(values + [""]), ignore_index=True)
                            num_per_twitch_metrics += 1
                    else:
                        key = list(dm[0].keys())[0]
                        for q in dm[0][key][m].keys():
                            values = [f"{CALCULATED_METRIC_DISPLAY_NAMES[m].format(q)}"]
                            value_series = [dm[0][k][m][q][WIDTH_VALUE_UUID] for k in dm[0].keys()]
                            df = df.append(pd.Series(values + value_series), ignore_index=True)
                            num_per_twitch_metrics += 1
                elif m in [RELAXATION_TIME_UUID, CONTRACTION_TIME_UUID]:
                    if d["error_msg"]:
                        for q in TWITCH_WIDTH_PERCENTS:
                            values = [f"{CALCULATED_METRIC_DISPLAY_NAMES[m].format(q)}"]
                            df = df.append(pd.Series(values + [""]), ignore_index=True)
                            num_per_twitch_metrics += 1
                    else:
                        key = list(dm[0].keys())[0]
                        for q in dm[0][key][m].keys():
                            values = [f"{CALCULATED_METRIC_DISPLAY_NAMES[m].format(q)}"]
                            value_series = [dm[0][k][m][q][TIME_VALUE_UUID] for k in dm[0].keys()]
                            df = df.append(pd.Series(values + value_series), ignore_index=True)
                            num_per_twitch_metrics += 1
                else:
                    values = [CALCULATED_METRIC_DISPLAY_NAMES[m]]
                    value_series = [dm[0][k][m] for k in dm[0].keys()] if dm else []
                    df = df.append(pd.Series(values + value_series), ignore_index=True)
                    num_per_twitch_metrics += 1

        for _ in range(5):
            df = df.append(pd.Series([""]), ignore_index=True)
            num_per_twitch_metrics += 1

    df.fillna("", inplace=True)
    return (df, num_per_twitch_metrics)
