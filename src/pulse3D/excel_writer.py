# -*- coding: utf-8 -*-
import datetime
import logging
import os
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from scipy import interpolate

from .constants import *
from .exceptions import *
from .peak_detection import concat
from .peak_detection import data_metrics
from .peak_detection import find_twitch_indices
from .peak_detection import get_windowed_peaks_valleys
from .peak_detection import init_dfs
from .peak_detection import peak_detector
from .plate_recording import PlateRecording
from .plotting import plotting_parameters
from .utils import get_experiment_id
from .utils import get_stiffness_label
from .utils import truncate
from .utils import truncate_float
from .utils import xl_col_to_name


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
        # convert peak/valley index to seconds
        idx_time = time_values[idx] / MICRO_TO_BASE_CONVERSION
        uninterpolated_time_seconds = round(idx_time, 2)

        # we can use the peak/valley indices directly because we are using the interpolated data
        row = idx + 2

        if is_optical_recording:
            value = (
                interpolated_data_function(uninterpolated_time_seconds * MICRO_TO_BASE_CONVERSION)
                - minimum_value
            ) * MICRO_TO_BASE_CONVERSION

        else:
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
    num_per_twitch_metrics: int,
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
    d: Dict[str, Any],
    well_name: str,
    num_data_points: int,
    num_per_twitch_metrics,
) -> None:

    well_row = well_index * num_per_twitch_metrics
    last_column = xl_col_to_name(num_data_points)

    frequency_chart.add_series(
        {
            "categories": f"='{PER_TWITCH_METRICS_SHEET_NAME}'!$B${well_row + 2}:${last_column}${well_row + 2}",
            "values": f"='{PER_TWITCH_METRICS_SHEET_NAME}'!$B${well_row + 7}:${last_column}${well_row + 7}",
            "marker": {"type": "diamond", "size": 7},
            "line": {"none": True},
        }
    )

    frequency_chart.set_legend({"none": True})

    x_axis_settings: Dict[str, Any] = {"name": "Time (seconds)"}
    x_axis_settings["min"] = d["start_time"]
    x_axis_settings["max"] = d["end_time"]

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


def write_xlsx(
    plate_recording: PlateRecording,
    normalize_y_axis: bool = True,
    max_y: Union[int, float] = None,
    start_time: Union[float, int] = 0,
    end_time: Union[float, int] = np.inf,
    twitch_widths: Tuple[int, ...] = DEFAULT_TWITCH_WIDTHS,
    baseline_widths_to_use: Tuple[int, ...] = DEFAULT_BASELINE_WIDTHS,
    prominence_factors: Tuple[Union[int, float], Union[int, float]] = DEFAULT_PROMINENCE_FACTORS,
    width_factors: Tuple[Union[int, float], Union[int, float]] = DEFAULT_WIDTH_FACTORS,
    peaks_valleys: Dict[str, List[List[int]]] = None,
):
    """Write plate recording waveform and computed metrics to Excel spredsheet.

    Args:
        plate_recording: loaded PlateRecording object
        normalize_y_axis: whether or not to set the max bound of the y-axis of all waveform graphs to the same value
        max_y: Sets the maximum bound for y-axis in the output graphs. Ignored if normalize_y_axis is False
        start_time: Start time of windowed analysis. Defaults to 0.
        end_time: End time of windowed analysis. Defaults to infinity.
        twitch_widths: Requested widths to add to output file
        baseline_widths_to_use: Twitch widths to use as baseline metrics
        prominence_factors: factors used to determine the prominence peaks/valleys must have
        width_factors: factors used to determine the width peaks/valleys must have
        peaks_valleys: User-defined peaks and valleys to use instead of peak_detector
    Raises:
        NotImplementedError: if peak finding algorithm fails for unexpected reason
        ValueError: if start and end times are outside of expected bounds, or do not ?
    """
    # make sure windows bounds are floats
    start_time = float(start_time)
    end_time = float(end_time)

    # get metadata from first well file
    w = [pw for pw in plate_recording if pw][0]
    interpolated_data_period = (
        w[INTERPOLATION_VALUE_UUID] if plate_recording.is_optical_recording else INTERPOLATED_DATA_PERIOD_US
    )
    # get max and min of final timepoints across each well
    raw_timepoints = [w.force[0][-1] for w in plate_recording if w]
    max_final_time_secs = max(raw_timepoints)
    interpolated_timepoints_secs = np.arange(
        interpolated_data_period, max_final_time_secs, interpolated_data_period
    )

    max_final_time_us = max_final_time_secs / MICRO_TO_BASE_CONVERSION
    # produce min final time truncated to 1 decimal place
    min_final_time_us = truncate_float(min(raw_timepoints) / MICRO_TO_BASE_CONVERSION, 1)

    if start_time < 0:
        raise ValueError(f"Window start time ({start_time}s) cannot be negative")
    if start_time >= round(min_final_time_us, 1):
        raise ValueError(
            f"Window start time ({start_time}s) greater than the max timepoint of this recording ({min_final_time_us:.1f}s)"
        )
    if end_time <= start_time:
        raise ValueError(
            f"Window end time ({end_time}s) must be greater than window start time ({start_time}s)"
        )

    end_time = min(end_time, max_final_time_us)
    is_full_analysis = start_time == 0 and end_time == max_final_time_us

    # create output file name
    input_file_name_no_ext = os.path.splitext(os.path.basename(plate_recording.path))[0]
    file_suffix = "full" if is_full_analysis else f"{start_time}-{end_time}"
    output_file_name = f"{input_file_name_no_ext}_{file_suffix}.xlsx"

    if w.stiffness_override:
        # reverse dict to use the stiffness factor as a key and get the label value
        post_stiffness_factor = {v: k for k, v in POST_STIFFNESS_LABEL_TO_FACTOR.items()}[w.stiffness_factor]
    else:
        post_stiffness_factor = get_stiffness_label(get_experiment_id(w[PLATE_BARCODE_UUID]))

    # create metadata sheet format as DataFrame
    metadata_rows = [
        ("Recording Information:", "", ""),
        ("", "Plate Barcode", w[PLATE_BARCODE_UUID]),
        (
            "",
            "UTC Timestamp of Beginning of Recording",
            str(w[UTC_BEGINNING_RECORDING_UUID].replace(tzinfo=None)),
        ),
        ("", "Post Stiffness Factor", post_stiffness_factor),
        ("", "", ""),
        ("Device Information:", "", ""),
        ("", "H5 File Layout Version", w[FILE_FORMAT_VERSION_METADATA_KEY]),
        ("", "Mantarray Serial Number", w.get(MANTARRAY_SERIAL_NUMBER_UUID, "")),
        ("", "Software Release Version", w.get(SOFTWARE_RELEASE_VERSION_UUID, "")),
        ("", "Firmware Version (Main Controller)", w.get(MAIN_FIRMWARE_VERSION_UUID, "")),
        ("Output Format:", "", ""),
        ("", "Pulse3D Version", PACKAGE_VERSION),
        ("", "File Creation Timestamp", str(datetime.datetime.utcnow().replace(microsecond=0))),
        ("", "Analysis Type (Full or Windowed)", "Full" if is_full_analysis else "Windowed"),
        ("", "Analysis Start Time (seconds)", f"{start_time:.1f}"),
        ("", "Analysis End Time (seconds)", f"{end_time:.1f}"),
    ]
    metadata_df = pd.DataFrame(
        {col: [row[i] for row in metadata_rows] for i, col in enumerate(("A", "B", "C"))}
    )

    twitch_width_percents = tuple(
        sorted(
            set([*twitch_widths, *(100 - np.array(twitch_widths, dtype=int)), *DEFAULT_TWITCH_WIDTH_PERCENTS])
        )
    )

    log.info("Computing data metrics for each well.")

    data = []
    max_force_of_recording = 0
    for well_file in plate_recording:
        # initialize some data structures
        error_msg = None

        # necessary for concatenating DFs together, in event that peak-finding fails and produces empty DF
        dfs = init_dfs(twitch_widths_range=twitch_width_percents)
        metrics = tuple(
            concat([dfs[k][j] for j in dfs[k].keys()], axis=1) for k in ("per_twitch", "aggregate")
        )

        if well_file is None:
            continue

        well_index = well_file[WELL_INDEX_UUID]
        well_name = TWENTY_FOUR_WELL_PLATE.get_well_name_from_well_index(well_index)

        # find bounding indices with respect to well recording
        well_start_idx, well_end_idx = truncate(
            source_series=interpolated_timepoints_secs,
            lower_bound=well_file.force[0][0],
            upper_bound=well_file.force[0][-1],
        )

        # find bounding indices of specified start/end windows
        window_start_idx, window_end_idx = truncate(
            source_series=interpolated_timepoints_secs / MICRO_TO_BASE_CONVERSION,
            lower_bound=start_time,
            upper_bound=end_time,
        )

        start_idx = max(window_start_idx, well_start_idx)
        end_idx = min(window_end_idx, well_end_idx)

        # fit interpolation function on recorded data
        interp_data_fn = interpolate.interp1d(well_file.force[0, :], well_file.force[1, :])

        # interpolate, normalize, and scale data
        interpolated_force = interp_data_fn(interpolated_timepoints_secs[start_idx:end_idx])
        interpolated_well_data = np.row_stack(
            [interpolated_timepoints_secs[start_idx:end_idx], interpolated_force]
        )

        min_value = min(interpolated_force)
        interpolated_force -= min_value
        interpolated_force *= MICRO_TO_BASE_CONVERSION

        # find the biggest activation twitch force over all
        max_force_of_well = max(interpolated_force)
        max_force_of_recording = max(max_force_of_recording, max_force_of_well)

        try:
            # compute peaks / valleys on interpolated well data
            log.info(f"Finding peaks and valleys for well {well_name}")

            if peaks_valleys is None:
                log.info("No user defined peaks and valleys were found, so calculating with peak_detector")
                peaks_and_valleys = peak_detector(
                    interpolated_well_data,
                    prominence_factors=prominence_factors,
                    width_factors=width_factors,
                    start_time=start_time,
                    end_time=end_time,
                )
            else:
                # convert peak and valley lists into a format compatible with find_twitch_indices
                peaks, valleys = [np.array(peaks_or_valleys) for peaks_or_valleys in peaks_valleys[well_name]]
                # get correct indices specific to windowed start and end
                peaks_and_valleys = get_windowed_peaks_valleys(
                    window_start_idx, window_end_idx, peaks, valleys
                )

            log.info(f"Finding twitch indices for well {well_name}")
            # Tanner (2/8/22): the value returned from this function isn't used, assuming it is only being called to raise PeakDetectionErrors
            find_twitch_indices(peaks_and_valleys)

            # compute metrics on interpolated well data
            log.info(f"Calculating metrics for well {well_name}")
            metrics = data_metrics(
                peaks_and_valleys,
                interpolated_well_data,
                twitch_width_percents=twitch_width_percents,
                baseline_widths_to_use=baseline_widths_to_use,
            )

        except TwoPeaksInARowError:
            error_msg = "Error: Two Contractions in a Row Detected"
        except TwoValleysInARowError:
            error_msg = "Error: Two Relaxations in a Row Detected"
        except TooFewPeaksDetectedError:
            error_msg = "Not Enough Twitches Detected"

        data.append(
            {
                "error_msg": error_msg,
                "peaks_and_valleys": peaks_and_valleys,
                "metrics": metrics,
                "well_index": well_index,
                "well_name": TWENTY_FOUR_WELL_PLATE.get_well_name_from_well_index(well_index),
                "min_value": min_value,
                "interp_data": interpolated_force,
                "interp_data_fn": interp_data_fn,
                "force": interpolated_well_data,
                "num_data_points": len(interpolated_well_data[0]),
                "start_time": start_time,
                "end_time": np.min(
                    [
                        interpolated_timepoints_secs[-1] / MICRO_TO_BASE_CONVERSION,
                        end_time,
                    ]
                ),
            }
        )
    # if the max y was not set by user and normalize_y_axis was not disabled by user, then set max_y to be the max twitch force between all wells
    if max_y is None and normalize_y_axis:
        max_y = int(max_force_of_recording)

    # waveform table
    continuous_waveforms = {
        "Time (seconds)": pd.Series(
            interpolated_timepoints_secs[start_idx:end_idx] / MICRO_TO_BASE_CONVERSION
        )
    }

    for d in data:
        continuous_waveforms[f"{d['well_name']} - Active Twitch Force (μN)"] = pd.Series(d["interp_data"])

    continuous_waveforms_df = pd.DataFrame(continuous_waveforms)

    _write_xlsx(
        output_file_name,
        metadata_df,
        continuous_waveforms_df,
        data,
        max_y,
        is_optical_recording=plate_recording.is_optical_recording,
        twitch_widths=twitch_widths,
        baseline_widths_to_use=baseline_widths_to_use,
    )

    log.info("Done")
    return output_file_name


def _write_xlsx(
    output_file_name: str,
    metadata_df: pd.DataFrame,
    continuous_waveforms_df: pd.DataFrame,
    data: List[Dict[Any, Any]],
    max_y: Optional[Union[float, int]],
    is_optical_recording: bool = False,
    twitch_widths: Tuple[int, ...] = DEFAULT_TWITCH_WIDTHS,
    baseline_widths_to_use: Tuple[int, ...] = DEFAULT_BASELINE_WIDTHS,
):
    with pd.ExcelWriter(output_file_name) as writer:
        log.info("Writing H5 file metadata")
        metadata_df.to_excel(writer, sheet_name="metadata", index=False, header=False)
        ws = writer.sheets["metadata"]

        for i_col_idx, i_col_width in ((0, 25), (1, 40), (2, 25)):
            ws.set_column(i_col_idx, i_col_idx, i_col_width)

        log.info("Writing continuous waveforms.")
        continuous_waveforms_df.to_excel(writer, sheet_name="continuous-waveforms", index=False)
        continuous_waveforms_sheet = writer.sheets["continuous-waveforms"]
        continuous_waveforms_sheet.set_column(0, 0, 18)

        for iter_well_idx in range(1, 24):
            continuous_waveforms_sheet.set_column(iter_well_idx, iter_well_idx, 13)

        # waveform snapshot/full
        wb = writer.book
        snapshot_sheet = wb.add_worksheet("continuous-waveform-snapshot")
        full_sheet = wb.add_worksheet("full-continuous-waveform-plots")

        for well_idx, dm in enumerate(data):
            log.info(f'Creating waveform charts for well {dm["well_name"]}')
            create_waveform_charts(
                max_y,
                well_idx,
                dm,
                continuous_waveforms_df,
                wb,
                continuous_waveforms_sheet,
                snapshot_sheet,
                full_sheet,
                is_optical_recording,
            )

        # aggregate metrics sheet
        log.info("Writing aggregate metrics.")
        aggregate_df = aggregate_metrics_df(data, twitch_widths, baseline_widths_to_use)
        aggregate_df.to_excel(writer, sheet_name="aggregate-metrics", index=False, header=False)

        # per twitch metrics sheet
        log.info("Writing per-twitch metrics.")
        pdf, num_metrics = per_twitch_df(data, twitch_widths, baseline_widths_to_use)
        pdf.to_excel(writer, sheet_name="per-twitch-metrics", index=False, header=False)

        # freq/force charts
        force_freq_sheet = wb.add_worksheet(FORCE_FREQUENCY_RELATIONSHIP_SHEET)
        freq_vs_time_sheet = wb.add_worksheet(TWITCH_FREQUENCIES_CHART_SHEET_NAME)

        for well_index, d in enumerate(data):
            dm = d["metrics"]
            if dm:
                force_freq_chart = wb.add_chart({"type": "scatter", "subtype": "straight"})
                freq_vs_time_chart = wb.add_chart({"type": "scatter", "subtype": "straight"})

                num_data_points = len(dm[0])

                log.info(f"Creating frequency vs time chart for well {d['well_name']}")
                create_frequency_vs_time_charts(
                    freq_vs_time_sheet,
                    freq_vs_time_chart,
                    well_index,
                    d,
                    d["well_name"],
                    num_data_points,
                    num_metrics,
                )

                log.info(f"Creating force frequency relationship chart for well {d['well_name']}")
                create_force_frequency_relationship_charts(
                    force_freq_sheet,
                    force_freq_chart,
                    well_index,
                    d["well_name"],
                    num_data_points,  # number of twitches
                    num_metrics,
                )
        log.info(f"Writing {output_file_name}")


def create_waveform_charts(
    max_y,
    well_idx,
    dm,
    continuous_waveforms_df,
    wb,
    continuous_waveforms_sheet,
    snapshot_sheet,
    full_sheet,
    is_optical_recording,
):
    # maximum snapshot size is 10 seconds
    lower_x_bound = dm["start_time"]

    upper_x_bound = (
        dm["end_time"]
        if dm["end_time"] - dm["start_time"] <= CHART_MAXIMUM_SNAPSHOT_LENGTH
        else dm["start_time"] + CHART_MAXIMUM_SNAPSHOT_LENGTH
    )

    df_column = continuous_waveforms_df.columns.get_loc(f"{dm['well_name']} - Active Twitch Force (μN)")

    well_column = xl_col_to_name(df_column)
    full_chart = wb.add_chart({"type": "scatter", "subtype": "straight"})

    # plot snapshot of waveform
    snapshot_plot_params = plotting_parameters(upper_x_bound - lower_x_bound)
    snapshot_chart = wb.add_chart({"type": "scatter", "subtype": "straight"})

    snapshot_chart.set_x_axis({"name": "Time (seconds)", "min": lower_x_bound, "max": upper_x_bound})
    snapshot_chart.set_y_axis(
        {"name": "Active Twitch Force (μN)", "major_gridlines": {"visible": 0}, "max": max_y}
    )
    snapshot_chart.set_title({"name": f"Well {dm['well_name']}"})

    snapshot_chart.add_series(
        {
            "name": "Waveform Data",
            "categories": f"='continuous-waveforms'!$A$2:$A${len(continuous_waveforms_df)}",
            "values": f"='continuous-waveforms'!${well_column}$2:${well_column}${len(continuous_waveforms_df)}",
            "line": {"color": "#1B9E77"},
        }
    )

    snapshot_chart.set_size({"width": snapshot_plot_params["chart_width"], "height": CHART_HEIGHT})
    snapshot_chart.set_plotarea(
        {
            "layout": {
                "x": snapshot_plot_params["x"],
                "y": 0.1,
                "width": snapshot_plot_params["plot_width"],
                "height": 0.7,
            }
        }
    )

    # plot full waveform
    full_plot_params = plotting_parameters(dm["end_time"] - dm["start_time"])

    full_chart.set_x_axis({"name": "Time (seconds)", "min": dm["start_time"], "max": dm["end_time"]})
    full_chart.set_y_axis(
        {"name": "Active Twitch Force (μN)", "major_gridlines": {"visible": 0}, "max": max_y}
    )
    full_chart.set_title({"name": f"Well {dm['well_name']}"})

    full_chart.add_series(
        {
            "name": "Waveform Data",
            "categories": f"='continuous-waveforms'!$A$2:$A${len(continuous_waveforms_df)}",
            "values": f"='continuous-waveforms'!${well_column}$2:${well_column}${len(continuous_waveforms_df)+1}",
            "line": {"color": "#1B9E77"},
        }
    )

    full_chart.set_size({"width": full_plot_params["chart_width"], "height": CHART_HEIGHT})
    full_chart.set_plotarea(
        {
            "layout": {
                "x": full_plot_params["x"],
                "y": 0.1,
                "width": full_plot_params["plot_width"],
                "height": 0.7,
            }
        }
    )

    peaks, valleys = dm["peaks_and_valleys"]
    log.info(f'Adding peak detection series for well {dm["well_name"]}')

    add_peak_detection_series(
        waveform_charts=[snapshot_chart, full_chart],
        continuous_waveform_sheet=continuous_waveforms_sheet,
        detector_type="Peak",
        well_index=well_idx,
        well_name=f"{dm['well_name']}",
        upper_x_bound_cell=dm["num_data_points"],
        indices=peaks,
        interpolated_data_function=dm["interp_data_fn"],
        time_values=dm["force"][0],
        is_optical_recording=is_optical_recording,
        minimum_value=dm["min_value"],
    )

    add_peak_detection_series(
        waveform_charts=[snapshot_chart, full_chart],
        continuous_waveform_sheet=continuous_waveforms_sheet,
        detector_type="Valley",
        well_index=well_idx,
        well_name=f"{dm['well_name']}",
        upper_x_bound_cell=dm["num_data_points"],
        indices=valleys,
        interpolated_data_function=dm["interp_data_fn"],
        time_values=dm["force"][0],
        is_optical_recording=is_optical_recording,
        minimum_value=dm["min_value"],
    )

    well_row, well_col = TWENTY_FOUR_WELL_PLATE.get_row_and_column_from_well_index(df_column - 1)
    snapshot_sheet.insert_chart(
        well_row * (CHART_HEIGHT_CELLS + 1),
        well_col * (CHART_FIXED_WIDTH_CELLS + 1),
        snapshot_chart,
    )
    full_sheet.insert_chart(1 + well_idx * (CHART_HEIGHT_CELLS + 1), 1, full_chart)


def aggregate_metrics_df(
    data: List[Dict[Any, Any]],
    widths: Tuple[int, ...] = DEFAULT_TWITCH_WIDTHS,
    baseline_widths_to_use: Tuple[int, ...] = DEFAULT_BASELINE_WIDTHS,
):
    """Combine aggregate metrics for each well into single DataFrame.

    Args:
        data (list): list of data metrics and metadata associated with each well
        widths (tuple of ints, optional): twitch-widths to return data for.
        baseline_widths_to_use: twitch widths to use as baseline metrics
    Returns:
        df (DataFrame): aggregate data frame of all metric aggregate measures
    """
    df = pd.DataFrame()
    df = df.append(pd.Series(["", "", *[d["well_name"] for d in data]]), ignore_index=True)
    df = df.append(pd.Series(["", "Treatment Description"]), ignore_index=True)
    df = df.append(
        pd.Series(
            [
                "",
                "n (twitches)",
                *[(len(d["metrics"][0]) if not d["error_msg"] else d["error_msg"]) for d in data],
            ]
        ),
        ignore_index=True,
    )
    df = df.append(pd.Series([""]), ignore_index=True)  # empty row

    combined = pd.concat([d["metrics"][1] for d in data])

    for metric_id in ALL_METRICS:
        if metric_id in (WIDTH_UUID, RELAXATION_TIME_UUID, CONTRACTION_TIME_UUID):
            for width in widths:
                name = CALCULATED_METRIC_DISPLAY_NAMES[metric_id].format(width)
                metric_df = combined[metric_id][width].drop(columns=["n"]).T
                df = _append_aggregate_measures_df(df, metric_df, name)

        elif metric_id in (BASELINE_TO_PEAK_UUID, PEAK_TO_BASELINE_UUID):
            baseline_width = (
                baseline_widths_to_use[0] if metric_id == BASELINE_TO_PEAK_UUID else baseline_widths_to_use[1]
            )
            # prevents duplicate entries in file if entered baseline(s) is/are the same as the entered twitch widths
            if baseline_width not in widths:
                name = CALCULATED_METRIC_DISPLAY_NAMES[metric_id].format(baseline_width)
                metric_df = combined[metric_id].drop(columns=["n"]).T.droplevel(level=-1, axis=0)
                df = _append_aggregate_measures_df(df, metric_df, name)
        else:
            name = CALCULATED_METRIC_DISPLAY_NAMES[metric_id]
            metric_df = combined[metric_id].drop(columns=["n"]).T.droplevel(level=-1, axis=0)
            df = _append_aggregate_measures_df(df, metric_df, name)

    return df


def _append_aggregate_measures_df(main_df: pd.DataFrame, metrics: pd.DataFrame, name: str):
    """Append metric-specific aggregate measures to aggregate data frame.

    Includes an empty row after aggregate measures

    Args:
        main_df (DataFrame): aggregate data frame
        metrics (DataFrame): metric-specific aggregate measures
        name (str): the display name of the metric
    Returns:
        main_df (DataFrame): aggregate data frame
    """
    metrics.reset_index(inplace=True)
    metrics.insert(0, "level_0", [name] + [""] * 5)
    metrics.columns = np.arange(metrics.shape[1])

    main_df = main_df.append(metrics, ignore_index=True)

    # empty row
    main_df = main_df.append(pd.Series([""]), ignore_index=True)

    return main_df


def per_twitch_df(
    data: List[Dict[Any, Any]],
    widths: Tuple[int, ...] = DEFAULT_TWITCH_WIDTHS,
    baseline_widths_to_use: Tuple[int, ...] = DEFAULT_BASELINE_WIDTHS,
):
    """Combine per-twitch metrics for each well into single DataFrame.

    Args:
        data (list): list of data metrics and metadata associated with each well
        widths (tuple of ints, optional): twitch-widths to return data for.
        baseline_widths_to_use: twitch widths to use as baseline metrics
    Returns:
        df (DataFrame): per-twitch data frame of all metrics
    """
    # append to a list instead of to a dataframe directly because it's faster and construct the dataframe at the end
    series_list = []

    for d in data:  # for each well
        num_per_twitch_metrics = 0  # len(labels)
        twitch_times = [d["force"][0, i] / MICRO_TO_BASE_CONVERSION for i in d["metrics"][0].index]

        # get metrics for single well
        dm = d["metrics"][0]

        series_list.append(pd.Series([d["well_name"]] + [f"Twitch {i+1}" for i in range(len(dm))]))
        series_list.append(pd.Series(["Timepoint of Twitch Contraction"] + twitch_times))

        num_per_twitch_metrics += 2

        for metric_id in ALL_METRICS:
            if metric_id in (WIDTH_UUID, RELAXATION_TIME_UUID, CONTRACTION_TIME_UUID):
                for twitch_width in widths:
                    values = [f"{CALCULATED_METRIC_DISPLAY_NAMES[metric_id].format(twitch_width)}"]
                    temp = pd.Series(values + list(dm[metric_id][twitch_width]))
                    series_list.append(temp)
                    num_per_twitch_metrics += 1
            elif metric_id in (BASELINE_TO_PEAK_UUID, PEAK_TO_BASELINE_UUID):
                baseline_width = (
                    baseline_widths_to_use[0]
                    if metric_id == BASELINE_TO_PEAK_UUID
                    else baseline_widths_to_use[1]
                )
                # prevents duplicate entries in file if entered baseline(s) is/are the same as the entered twitch widths
                if baseline_width not in widths:
                    values = [CALCULATED_METRIC_DISPLAY_NAMES[metric_id].format(baseline_width)]
                    temp = pd.Series(values + list(dm[metric_id]))
                    series_list.append(temp)
                    num_per_twitch_metrics += 1
            else:
                values = [CALCULATED_METRIC_DISPLAY_NAMES[metric_id]]
                temp = pd.Series(values + list(dm[metric_id]))
                series_list.append(temp)
                num_per_twitch_metrics += 1

        for _ in range(5):
            series_list.append(pd.Series([""]))
            num_per_twitch_metrics += 1

    df = pd.concat(series_list, axis=1).T
    df.fillna("", inplace=True)
    return df, num_per_twitch_metrics
