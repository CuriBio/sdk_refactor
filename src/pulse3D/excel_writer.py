# -*- coding: utf-8 -*-
import datetime
import json
import logging
import os
import string
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
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
from .stimulation import aggregate_timepoints
from .stimulation import realign_interpolated_stim_data
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
    if detector_type == "Valley":
        label = "Relaxation"
        offset = 1
        marker_color = "#D95F02"
    else:
        label = "Contraction"
        offset = 0
        marker_color = "#7570B3"

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


def add_stim_data_series(
    charts, format, charge_unit, col_offset: int, series_label: str, upper_x_bound_cell: int, y_axis_bounds
) -> None:
    stim_timepoints_col = xl_col_to_name(STIM_DATA_COLUMN_START)
    stim_session_col = xl_col_to_name(STIM_DATA_COLUMN_START + col_offset)

    series_params: Dict[str, Any] = {
        "name": series_label,
        "categories": f"='continuous-waveforms'!${stim_timepoints_col}$2:${stim_timepoints_col}${upper_x_bound_cell}",
        "values": f"='continuous-waveforms'!${stim_session_col}$2:${stim_session_col}${upper_x_bound_cell}",
        "line": {"color": "#d1d128"},
    }
    y_axis_params: Dict[str, Any] = {"name": f"Stimulator Output ({charge_unit})", **y_axis_bounds}

    if format == "overlayed":
        series_params["y2_axis"] = 1
    else:
        y_axis_params["major_gridlines"] = {"visible": 0}

    for chart in charts:
        chart.add_series(series_params)
        (chart.set_y2_axis if format == "overlayed" else chart.set_y_axis)(y_axis_params)
        chart.show_blanks_as("span")


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
    include_stim_protocols: bool = False,
    stim_waveform_format: Optional[Union[Literal["stacked"], Literal["overlayed"]]] = None,
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
        include_stim_protocols: Toggles the addition of stimulation-protocols sheet in the output excel
        stim_waveform_format: Toggles the output format of the stim waveforms if provided, o/w no waveforms are displayed
    Raises:
        NotImplementedError: if peak finding algorithm fails for unexpected reason
        ValueError: if start and end times are outside of expected bounds, or do not ?
    """
    # get metadata from first well file
    first_wf = next(iter(plate_recording))

    if stim_waveform_format is not None:
        if stim_waveform_format not in ("stacked", "overlayed"):
            raise ValueError(f"Invalid stim_waveform_format: {stim_waveform_format}")

    # make sure windows bounds are floats
    start_time = float(start_time)
    end_time = float(end_time)

    interpolated_data_period_us = (
        first_wf[INTERPOLATION_VALUE_UUID]
        if plate_recording.is_optical_recording
        else INTERPOLATED_DATA_PERIOD_US
    )

    # get stim metadata
    stim_protocols_df = _create_stim_protocols_df(plate_recording) if include_stim_protocols else None

    # get max and min of final timepoints across each well
    raw_timepoints = [w.force[0, -1] for w in plate_recording if w]
    max_final_time_us = max(raw_timepoints)
    interpolated_timepoints_us = np.arange(
        interpolated_data_period_us, max_final_time_us, interpolated_data_period_us
    )

    max_final_time_secs = max_final_time_us / MICRO_TO_BASE_CONVERSION
    # produce min final time truncated to 1 decimal place
    min_final_time_secs = truncate_float(min(raw_timepoints) / MICRO_TO_BASE_CONVERSION, 1)

    if start_time < 0:
        raise ValueError(f"Window start time ({start_time}s) cannot be negative")
    if start_time >= round(min_final_time_secs, 1):
        raise ValueError(
            f"Window start time ({start_time}s) greater than the max timepoint of this recording ({min_final_time_secs:.1f}s)"
        )
    if end_time <= start_time:
        raise ValueError("Window end time must be greater than window start time")

    end_time = min(end_time, max_final_time_secs)
    is_full_analysis = start_time == 0 and end_time == max_final_time_secs

    # create output file name
    input_file_name_no_ext = os.path.splitext(os.path.basename(plate_recording.path))[0]
    file_suffix = "full" if is_full_analysis else f"{start_time}-{end_time}"
    output_file_name = f"{input_file_name_no_ext}_{file_suffix}.xlsx"

    if first_wf.stiffness_override:
        # reverse dict to use the stiffness factor as a key and get the label value
        post_stiffness_factor = {v: k for k, v in POST_STIFFNESS_LABEL_TO_FACTOR.items()}[
            first_wf.stiffness_factor
        ]
    else:
        post_stiffness_factor = get_stiffness_label(get_experiment_id(first_wf[PLATE_BARCODE_UUID]))

    # create metadata sheet format as DataFrame
    metadata_rows = [
        ("Recording Information:", "", ""),
        ("", "Plate Barcode", first_wf[PLATE_BARCODE_UUID]),
        ("", "Stimulation Lid Barcode", first_wf.get(STIM_BARCODE_UUID)),
        (
            "",
            "UTC Timestamp of Beginning of Recording",
            str(first_wf[UTC_BEGINNING_RECORDING_UUID].replace(tzinfo=None)),
        ),
        ("", "Post Stiffness Factor", post_stiffness_factor),
        ("", "", ""),
        ("Device Information:", "", ""),
        ("", "H5 File Layout Version", first_wf.version),
        ("", "Mantarray Serial Number", first_wf.get(MANTARRAY_SERIAL_NUMBER_UUID, "")),
        ("", "Software Release Version", first_wf.get(SOFTWARE_RELEASE_VERSION_UUID, "")),
        ("", "Firmware Version (Main Controller)", first_wf.get(MAIN_FIRMWARE_VERSION_UUID, "")),
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

    tissue_data = []
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
            source_series=interpolated_timepoints_us,
            lower_bound=well_file.force[0][0],
            upper_bound=well_file.force[0][-1],
        )

        # find bounding indices of specified start/end windows
        window_start_idx, window_end_idx = truncate(
            source_series=interpolated_timepoints_us / MICRO_TO_BASE_CONVERSION,
            lower_bound=start_time,
            upper_bound=end_time,
        )

        start_idx = max(window_start_idx, well_start_idx)
        end_idx = min(window_end_idx, well_end_idx)

        # fit interpolation function on recorded data
        interp_data_fn = interpolate.interp1d(well_file.force[0, :], well_file.force[1, :])
        # interpolate, normalize, and scale data
        interpolated_force = interp_data_fn(interpolated_timepoints_us[start_idx:end_idx])
        interpolated_well_data = np.row_stack(
            [interpolated_timepoints_us[start_idx:end_idx], interpolated_force]
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

        tissue_data.append(
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
                "end_time": np.min([interpolated_timepoints_us[-1] / MICRO_TO_BASE_CONVERSION, end_time]),
            }
        )

    continuous_waveforms_df = _create_continuous_waveforms_df(
        interpolated_timepoints_us, tissue_data, start_idx, end_idx
    )

    if not normalize_y_axis:
        # override given value since y-axis normalization is disabled
        max_y = None
    elif max_y is None:
        # if y-axis normalization enabled but no max Y given, then set it to the max twitch force across all wells
        max_y = int(max_force_of_recording)

    # Tanner (12/15/22): setting min to zero right now since the tissue data will never be < 0. If this ever needs to change, may want to also take the new min into account when setting the y2 axis bounds for stim data
    y_axis_bounds = {"tissue": {"max": max_y, "min": 0}}

    stim_plotting_info: Dict[str, Any] = _get_stim_plotting_data(
        plate_recording, start_time, end_time, stim_waveform_format, normalize_y_axis, y_axis_bounds
    )

    _write_xlsx(
        output_file_name=output_file_name,
        metadata_df=metadata_df,
        continuous_waveforms_df=continuous_waveforms_df,
        stim_protocols_df=stim_protocols_df,
        stim_plotting_info=stim_plotting_info,
        data=tissue_data,
        y_axis_bounds=y_axis_bounds,
        include_stim_protocols=include_stim_protocols,
        is_optical_recording=plate_recording.is_optical_recording,
        twitch_widths=twitch_widths,
        baseline_widths_to_use=baseline_widths_to_use,
    )

    log.info("Done")
    return output_file_name


def _create_stim_protocols_df(plate_recording):
    unassigned_wells = []
    stim_protocols_dict = {
        "Title": {
            "Unassigned Wells": "Unassigned Wells:",
            "title_break": "",
            "Protocol ID": "Protocol ID:",
            "Stimulation Type": "Stimulation Type:",
            "Run Until Stopped": "Run Until Stopped:",
            "Wells": "Wells:",
            "Subprotocols": "Subprotocols:",
        }
    }

    for well in plate_recording:
        if well_data := json.loads(well[STIMULATION_PROTOCOL_UUID]):
            protocol_id = well_data.get("protocol_id")
            well_id = well[WELL_NAME_UUID]
            if protocol_id not in stim_protocols_dict:
                # functions as the scheme for entering data into this sheet
                stim_protocols_dict[protocol_id] = {
                    "Protocol ID": protocol_id,
                    "Stimulation Type": "Current"
                    if well_data.get("stimulation_type") == "C"
                    else "Voltage"
                    if well_data.get("stimulation_type") == "V"
                    else well_data.get("stimulation_type"),
                    "Run Until Stopped": "Active" if well_data.get("run_until_stopped") else "Disabled",
                    "Wells": f"{well_id}, ",
                    "Subprotocols": "",
                }
                stim_protocols_dict[protocol_id]["Subprotocols"] = well_data.get("subprotocols")
            else:
                stim_protocols_dict[protocol_id]["Wells"] += f"{well_id}, "
        else:
            unassigned_wells.append(f"{well[WELL_NAME_UUID]}, ")

    if len(unassigned_wells) == 24:  # if all wells are unassigned
        stim_protocols_dict["Title"] = {"message": "No stimulation protocols applied"}
    elif len(unassigned_wells) > 0:  # if some of the wells are unassigned
        stim_protocols_dict[list(stim_protocols_dict.keys())[1]]["Unassigned Wells"] = "".join(
            unassigned_wells
        )

    return pd.DataFrame(stim_protocols_dict)


def _create_continuous_waveforms_df(interpolated_timepoints_us, tissue_data, start_idx, end_idx):
    continuous_waveforms = {
        "Time (seconds)": pd.Series(interpolated_timepoints_us[start_idx:end_idx] / MICRO_TO_BASE_CONVERSION)
    }
    continuous_waveforms.update(
        {f"{d['well_name']} - Active Twitch Force (μN)": pd.Series(d["interp_data"]) for d in tissue_data}
    )
    return pd.DataFrame(continuous_waveforms)


def _get_stim_plotting_data(
    plate_recording, start_time, end_time, stim_waveform_format, normalize_y_axis, y_axis_bounds
):
    if not stim_waveform_format:
        return {}

    charge_units = {}

    start_time_us = int(start_time * MICRO_TO_BASE_CONVERSION)
    end_time_us = int(end_time * MICRO_TO_BASE_CONVERSION)

    # insert this first since dict insertion order matters for the data frame creation
    stim_waveforms_dict = {"Stim Time (seconds)": None}

    for well_idx, wf in enumerate(plate_recording):
        well_name = TWENTY_FOUR_WELL_PLATE.get_well_name_from_well_index(well_idx)

        if not wf.stim_sessions:
            continue

        stim_protocol = json.loads(wf[STIMULATION_PROTOCOL_UUID])

        charge_units[well_name] = "mA" if stim_protocol["stimulation_type"] == "C" else "mV"

        stim_session_idx = 0
        for waveform in wf.stim_sessions:
            stim_session_idx += 1
            stim_waveforms_dict[f"{well_name} - Stim Session {stim_session_idx}"] = waveform[
                :, (start_time_us <= waveform[0]) & (waveform[0] <= end_time_us)
            ]

    stim_timepoints_aggregate_us = aggregate_timepoints(
        [waveform[0] for title, waveform in stim_waveforms_dict.items() if "Stim Session" in title]  # type: ignore
    )
    stim_timepoints_for_plotting_us = np.repeat(stim_timepoints_aggregate_us, 2)

    max_stim_amplitude = 0
    min_stim_amplitude = 0

    # convert all to series, remove timepoints and convert to correct unit in stim sessions
    for title, arr in stim_waveforms_dict.items():
        if title == "Stim Time (seconds)":
            new_arr = stim_timepoints_for_plotting_us / MICRO_TO_BASE_CONVERSION
        else:
            max_stim_amplitude = max(max_stim_amplitude, max(arr[1]))  # type: ignore
            min_stim_amplitude = min(min_stim_amplitude, min(arr[1]))  # type: ignore
            new_arr = realign_interpolated_stim_data(stim_timepoints_for_plotting_us, arr)
        stim_waveforms_dict[title] = pd.Series(new_arr)

    stim_waveform_df = pd.DataFrame(stim_waveforms_dict)
    if stim_waveform_df.empty:
        return {}

    y_axis_bounds["stim"] = {"max": None, "min": None}
    if normalize_y_axis:
        y_axis_bounds["stim"] = {"max": max_stim_amplitude, "min": min_stim_amplitude}

    return {
        "chart_format": stim_waveform_format,
        "charge_units": charge_units,
        "stim_waveform_df": stim_waveform_df,
    }


def _write_xlsx(
    output_file_name: str,
    metadata_df: pd.DataFrame,
    continuous_waveforms_df: pd.DataFrame,
    stim_protocols_df: pd.DataFrame,
    stim_plotting_info: Dict[str, Any],
    data: List[Dict[Any, Any]],
    y_axis_bounds: Dict[str, Dict[str, Any]],
    include_stim_protocols: bool = False,
    is_optical_recording: bool = False,
    twitch_widths: Tuple[int, ...] = DEFAULT_TWITCH_WIDTHS,
    baseline_widths_to_use: Tuple[int, ...] = DEFAULT_BASELINE_WIDTHS,
):
    log.info(f"Writing {output_file_name}")
    with pd.ExcelWriter(output_file_name) as writer:
        _write_metadata(writer, metadata_df)

        if include_stim_protocols:
            _write_stim_protocols(writer, stim_protocols_df)

        continuous_waveforms_sheet = _write_continuous_waveforms(writer, continuous_waveforms_df)

        if stim_plotting_info:
            _write_stim_waveforms(writer, stim_plotting_info["stim_waveform_df"])

        # waveform snapshot/full
        wb = writer.book
        snapshot_sheet = wb.add_worksheet("continuous-waveform-snapshot")
        full_sheet = wb.add_worksheet("full-continuous-waveform-plots")

        for well_idx, dm in enumerate(data):
            log.info(f'Creating waveform charts for well {dm["well_name"]}')
            create_waveform_charts(
                y_axis_bounds,
                well_idx,
                dm,
                continuous_waveforms_df,
                wb,
                continuous_waveforms_sheet,
                snapshot_sheet,
                full_sheet,
                is_optical_recording,
                stim_plotting_info,
            )

        _write_aggregate_metrics(writer, data, twitch_widths, baseline_widths_to_use)

        num_metrics = _write_per_twitch_metrics(writer, data, twitch_widths, baseline_widths_to_use)

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


def _write_metadata(writer, metadata_df):
    log.info("Writing H5 file metadata")
    metadata_df.to_excel(writer, sheet_name="metadata", index=False, header=False)
    metadata_sheet = writer.sheets["metadata"]

    for i_col_idx, i_col_width in ((0, 25), (1, 40), (2, 25)):
        metadata_sheet.set_column(i_col_idx, i_col_idx, i_col_width)


def _write_stim_protocols(writer, stim_protocols_df):
    log.info("Writing stimulation protocols.")
    stim_protocols_df.to_excel(writer, sheet_name="stimulation-protocols", index=False, header=False)
    stim_protocols_sheet = writer.sheets["stimulation-protocols"]
    stim_protocols_sheet.set_column(0, 0, 18)
    stim_protocols_sheet.set_column(1, stim_protocols_df.shape[1] - 1, 45)
    # if the length is one then protocols sheet was requested but no protocols have been used
    # add each subprotocols to each column with formats
    if len(stim_protocols_df) > 1:
        column_counter = 0
        for _, protocol_data in stim_protocols_df.iteritems():
            if column_counter != 0:
                stim_protocols_sheet.merge_range(
                    6, column_counter, len(protocol_data["Subprotocols"]) * 10 + 6, column_counter, ""
                )
            subprotocols = protocol_data["Subprotocols"]
            subprotocols_format = writer.book.add_format()
            subprotocols_format.set_text_wrap()
            subprotocols_format.set_align("top")
            stim_protocols_sheet.write(
                f"{string.ascii_uppercase[column_counter]}7",
                json.dumps(subprotocols, indent=4),
                subprotocols_format,
            )
            column_counter += 1


def _write_continuous_waveforms(writer, continuous_waveforms_df):
    log.info("Writing continuous waveforms.")
    continuous_waveforms_df.to_excel(writer, sheet_name="continuous-waveforms", index=False)
    continuous_waveforms_sheet = writer.sheets["continuous-waveforms"]

    for iter_well_idx in range(1, 24):
        continuous_waveforms_sheet.set_column(iter_well_idx, iter_well_idx, 13)

    return continuous_waveforms_sheet


def _write_stim_waveforms(writer, stim_waveform_df):
    log.info("Writing stim data")
    stim_waveform_df.to_excel(
        writer, sheet_name="continuous-waveforms", index=False, startcol=STIM_DATA_COLUMN_START
    )


def _write_aggregate_metrics(writer, data, twitch_widths, baseline_widths_to_use):
    log.info("Writing aggregate metrics.")
    aggregate_df = aggregate_metrics_df(data, twitch_widths, baseline_widths_to_use)
    aggregate_df.to_excel(writer, sheet_name="aggregate-metrics", index=False, header=False)


def _write_per_twitch_metrics(writer, data, twitch_widths, baseline_widths_to_use):
    log.info("Writing per-twitch metrics.")
    pdf, num_metrics = per_twitch_df(data, twitch_widths, baseline_widths_to_use)
    pdf.to_excel(writer, sheet_name="per-twitch-metrics", index=False, header=False)
    return num_metrics


def create_waveform_charts(
    y_axis_bounds,
    well_idx,
    dm,
    continuous_waveforms_df,
    wb,
    continuous_waveforms_sheet,
    snapshot_sheet,
    full_sheet,
    is_optical_recording,
    stim_plotting_info,
):
    well_name = dm["well_name"]

    # maximum snapshot size is 10 seconds
    lower_x_bound = dm["start_time"]

    upper_x_bound = (
        dm["end_time"]
        if dm["end_time"] - dm["start_time"] <= CHART_MAXIMUM_SNAPSHOT_LENGTH
        else dm["start_time"] + CHART_MAXIMUM_SNAPSHOT_LENGTH
    )

    df_column = continuous_waveforms_df.columns.get_loc(f"{well_name} - Active Twitch Force (μN)")

    well_column = xl_col_to_name(df_column)

    # plot snapshot of waveform
    snapshot_plot_params = plotting_parameters(upper_x_bound - lower_x_bound)

    snapshot_chart = wb.add_chart({"type": "scatter", "subtype": "straight"})

    snapshot_chart.set_x_axis({"name": "Time (seconds)", "min": lower_x_bound, "max": upper_x_bound})
    snapshot_chart.set_y_axis(
        {"name": "Active Twitch Force (μN)", "major_gridlines": {"visible": 0}, **y_axis_bounds["tissue"]}
    )
    snapshot_chart.set_title({"name": f"Well {well_name}"})

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
    full_plot_include_y2_axis = stim_plotting_info.get("chart_format") == "overlayed"

    full_plot_params = plotting_parameters(
        dm["end_time"] - dm["start_time"], include_y2_axis=full_plot_include_y2_axis
    )

    full_chart = wb.add_chart({"type": "scatter", "subtype": "straight"})

    full_chart.set_x_axis({"name": "Time (seconds)", "min": dm["start_time"], "max": dm["end_time"]})
    full_chart.set_y_axis(
        {"name": "Active Twitch Force (μN)", "major_gridlines": {"visible": 0}, **y_axis_bounds["tissue"]}
    )
    full_chart.set_title({"name": f"Well {well_name}"})

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

    stim_chart_format = stim_plotting_info.get("chart_format")

    if stim_plotting_info:
        log.info(f"Adding stim data series for well {well_name}")

        if stim_chart_format == "overlayed":
            chart = full_chart
        else:
            chart = stim_chart = wb.add_chart({"type": "scatter", "subtype": "straight"})

            chart.set_x_axis({"name": "Time (seconds)", "min": dm["start_time"], "max": dm["end_time"]})

            chart.set_size({"width": full_plot_params["chart_width"], "height": STIM_CHART_HEIGHT})
            chart.set_plotarea(
                {
                    "layout": {
                        "x": full_plot_params["x"],
                        "y": 0.1,
                        "width": full_plot_params["plot_width"],
                        "height": 0.7,
                    }
                }
            )

        stim_waveform_df = stim_plotting_info["stim_waveform_df"]
        for col_idx, col_title in enumerate(stim_waveform_df):
            if not col_title.startswith(well_name):
                continue

            series_label = col_title.split("-")[-1].strip()
            add_stim_data_series(
                charts=[chart],
                format=stim_chart_format,
                charge_unit=stim_plotting_info["charge_units"][well_name],
                col_offset=col_idx,
                series_label=series_label,
                upper_x_bound_cell=len(stim_waveform_df["Stim Time (seconds)"]),
                y_axis_bounds=y_axis_bounds["stim"],
            )

    peaks, valleys = dm["peaks_and_valleys"]
    log.info(f"Adding peak detection series for well {well_name}")

    add_peak_detection_series(
        waveform_charts=[snapshot_chart, full_chart],
        continuous_waveform_sheet=continuous_waveforms_sheet,
        detector_type="Peak",
        well_index=well_idx,
        well_name=well_name,
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
        well_name=well_name,
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

    cells_per_well = CHART_HEIGHT_CELLS + 1
    if stim_chart_format == "stacked":
        cells_per_well += STIM_CHART_HEIGHT_CELLS
        if stim_chart.series:
            full_sheet.insert_chart(1 + CHART_HEIGHT_CELLS + well_idx * cells_per_well, 1, stim_chart)

    full_sheet.insert_chart(1 + well_idx * cells_per_well, 1, full_chart)


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
    df = pd.concat([df, pd.Series(["", "", *[d["well_name"] for d in data]])], ignore_index=True)
    df = pd.concat([df, pd.Series(["", "Treatment Description"])], ignore_index=True)
    df = pd.concat(
        [
            df,
            pd.Series(
                [
                    "",
                    "n (twitches)",
                    *[(len(d["metrics"][0]) if not d["error_msg"] else d["error_msg"]) for d in data],
                ]
            ),
        ],
        ignore_index=True,
    )
    df = pd.concat([df, pd.Series([""])], ignore_index=True)  # empty row

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

    main_df = pd.concat([main_df, metrics], ignore_index=True)

    # empty row
    main_df = pd.concat([main_df, pd.Series([""])], ignore_index=True)

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
