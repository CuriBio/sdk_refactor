# -*- coding: utf-8 -*-
import datetime
import logging

import numpy as np
import pandas as pd
from scipy import interpolate
from typing import Any
from typing import List
from typing import Union

from .plate_recording import WellFile, PlateRecording
from .constants import *
from .exceptions import *
from .peak_detection import concat
from .peak_detection import data_metrics
from .peak_detection import find_twitch_indices
from .peak_detection import init_dfs
from .peak_detection import peak_detector
from .plotting import plotting_parameters
from .utils import xl_col_to_name
from .utils import truncate

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

    start_time=time_values[0]/MICRO_TO_BASE_CONVERSION

    for idx in indices:
        # convert peak/valley index to seconds
        idx_time = time_values[idx] / MICRO_TO_BASE_CONVERSION
        # subtract start time
        shifted_idx_time = idx_time - start_time

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
            "marker": {
                "type": "diamond",
                "size": 7,
            },
            "line": {"none": True},
        }
    )

    frequency_chart.set_legend({"none": True})

    x_axis_settings: Dict[str, Any] = {"name": "Time (seconds)"}
    x_axis_settings["min"] = d['start_time']
    x_axis_settings["max"] = d['end_time']

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


def write_xlsx(plate_recording,
               name=None,
               start_time: float = 0.0,
               end_time: float = np.inf,
               twitch_widths: Tuple[int] = tuple([50,90])):
    """Write plate recording waveform and computed metrics to Excel spredsheet.

    Args:
        plate_recording (PlateRecording): loaded plate recording object
        name (str, optional): File name of Excel spreadsheet. Defaults to None.
        start_time (float): Start time of windowed analysis. Defaults to 0.0.
        end_time (float): End time of windowed analysis. Defaults to np.inf.

    Raises:
        NotImplementedError: if peak finding algorithm fails for unexpected reason
        ValueError: if start and end times are outside of expected bounds, or do not
    """
    # get metadata from first well file
    w = [pw for pw in plate_recording if pw][0]
    if name is None:
        name = f"{w[PLATE_BARCODE_UUID]}__{w[UTC_BEGINNING_RECORDING_UUID].strftime('%Y_%m_%d_%H%M%S')}.xlsx"
    interpolated_data_period = (
        w[INTERPOLATION_VALUE_UUID] if plate_recording.is_optical_recording else INTERPOLATED_DATA_PERIOD_US
    )

    # get max_time from all wells
    max_time = max([w.force[0][-1] for w in plate_recording if w])
    min_time = min([w.force[0][-1] for w in plate_recording if w])
    time_points = np.arange(interpolated_data_period, max_time, interpolated_data_period)

    # Kristian 1/25/22
    try:
        assert start_time >= 0
    except Exception as e:
        raise ValueError(f"Window start time ({start_time}) cannot be negative.") from e
    try:
        assert start_time < (min_time/MICRO_TO_BASE_CONVERSION)
    except Exception as e:
        raise ValueError(f"Window start time ({start_time}s) cannot be after longest time of shortest recording ({(min_time/MICRO_TO_BASE_CONVERSION):.1f}).") from e
    try:
        assert end_time > start_time
    except Exception as e:
        raise ValueError(f"Window end time ({end_time}) must come after window start time ({start_time}).") from e

    end_time=np.min([end_time, max_time/MICRO_TO_BASE_CONVERSION])
    is_full_analysis=np.all([(start_time==0),(end_time==max_time/MICRO_TO_BASE_CONVERSION)])

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
            "",
            "",
            ""
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
            "Analysis Type (Full or Windowed)",
            "Analysis Start Time (seconds)",
            "Analysis End Time (seconds)"
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
            "Full" if is_full_analysis else "Windowed",
            "%.1f" % (start_time),
            "%.1f" % (end_time)
        ],
    }
    metadata_df = pd.DataFrame(metadata)

    data = []

    log.info("Computing data metrics for each well.")

    for i, well_file in enumerate(plate_recording):

        # initialize some data structures
        error_msg = None
        peaks_and_valleys = None
        twitch_indices = None
        # necessary for concatenating DFs together, in event that peak-finding fails and produces empty DF
        dfs = init_dfs()
        metrics = [concat([dfs[k][j] for j in dfs[k].keys()], axis=1) for k in ['per-twitch','aggregate']]

        if well_file is None:
            continue

        well_index = well_file[WELL_INDEX_UUID]
        well_name = TWENTY_FOUR_WELL_PLATE.get_well_name_from_well_index(well_index)

        # find bounding indices with respect to well recording
        well_start_idx, well_end_idx = truncate(source_series=time_points,
                                                lower_bound=well_file.force[0][0],
                                                upper_bound=well_file.force[0][-1])

        # find bounding indices of specified start/end windows
        window_start_idx, window_end_idx = truncate(source_series=time_points/MICRO_TO_BASE_CONVERSION,
                                                    lower_bound=start_time,
                                                    upper_bound=end_time)

        start_idx = np.max([window_start_idx, well_start_idx])
        end_idx = np.min([window_end_idx, well_end_idx])

        # fit interpolation function on recorded data
        interp_data_fn = interpolate.interp1d(well_file.force[0,:],
                                              well_file.force[1,:])

        # interpolate, normalize, and scale data
        interpolated_force = interp_data_fn(time_points[start_idx:end_idx])
        interpolated_well_data = np.row_stack([time_points[start_idx:end_idx],
                                               interpolated_force])

        min_value = min(interpolated_force)
        interpolated_force -= min_value
        interpolated_force *= MICRO_TO_BASE_CONVERSION

        try:
            # compute peaks / valleys on interpolated well data
            log.info(f"Finding peaks and valleys for well {well_name}")
            peaks_and_valleys = peak_detector(interpolated_well_data, start_time=start_time, end_time=end_time)

            log.info(f"Finding twitch indices for well {well_name}")
            twitch_indices = list(find_twitch_indices(peaks_and_valleys).keys())

            # compute metrics on interpolated well data
            log.info(f"Calculating metrics for well {well_name}")
            metrics = data_metrics(peaks_and_valleys, interpolated_well_data)

        except TwoPeaksInARowError:
            error_msg = "Error: Two Contractions in a Row Detected"
        except TwoValleysInARowError:
            error_msg = "Error: Two Relaxations in a Row Detected"
        except TooFewPeaksDetectedError:
            error_msg = "Not Enough Twitches Detected"
        except Exception as e:
            raise NotImplementedError("Unknown PeakDetectionError") from e

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
                "end_time": np.min([time_points[-1]/MICRO_TO_BASE_CONVERSION, end_time])
            }
        )

    # waveform table
    continuous_waveforms = {"Time (seconds)": pd.Series(time_points[start_idx:end_idx] / MICRO_TO_BASE_CONVERSION)}
    for d in data:
        continuous_waveforms[f"{d['well_name']} - Active Twitch Force (μN)"] = pd.Series(d["interp_data"])
    continuous_waveforms_df = pd.DataFrame(continuous_waveforms)

    _write_xlsx(name=name,
                metadata_df=metadata_df,
                continuous_waveforms_df=continuous_waveforms_df,
                data=data,
                is_optical_recording=plate_recording.is_optical_recording,
                twitch_widths=twitch_widths)
    log.info("Done")


def _write_xlsx(name: str,
                metadata_df: pd.DataFrame,
                continuous_waveforms_df: pd.DataFrame,
                data: List[Dict[Any,Any]],
                is_optical_recording: bool = False,
                twitch_widths: Tuple[int] = tuple([50,90])):

    with pd.ExcelWriter(name) as writer:
        log.info(f"Writing H5 file metadata")
        metadata_df.to_excel(writer, sheet_name="metadata", index=False, header=False)
        ws = writer.sheets["metadata"]

        for i_col_idx, i_col_width in ((0, 25), (1, 40), (2, 25)):
            ws.set_column(i_col_idx, i_col_idx, i_col_width)

        log.info("Writing continuous waveforms.")
        continuous_waveforms_df.to_excel(writer, sheet_name='continuous-waveforms', index=False)
        continuous_waveforms_sheet = writer.sheets['continuous-waveforms']
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
        log.info("Writing aggregate metrics.")
        aggregate_df = aggregate_metrics_df(data)
        aggregate_df.to_excel(writer, sheet_name="aggregate-metrics", index=False, header=False)

        # per twitch metrics sheet
        log.info('Writing per-twitch metrics.')
        (pdf, num_metrics) = per_twitch_df(data, twitch_widths)
        pdf.to_excel(writer, sheet_name='per-twitch-metrics', index=False, header=False)

        # freq/force charts
        force_freq_sheet = wb.add_worksheet(FORCE_FREQUENCY_RELATIONSHIP_SHEET)
        freq_vs_time_sheet = wb.add_worksheet(TWITCH_FREQUENCIES_CHART_SHEET_NAME)

        for well_index, d in enumerate(data):
            dm = d["metrics"]
            if dm:
                force_freq_chart = wb.add_chart({"type": "scatter", "subtype": "straight"})
                freq_vs_time_chart = wb.add_chart({"type": "scatter", "subtype": "straight"})

                num_data_points=len(dm[0])

                log.info(f'Creating frequency vs time chart for well {d["well_name"]}')
                create_frequency_vs_time_charts(
                    freq_vs_time_sheet,
                    freq_vs_time_chart,
                    well_index,
                    d,
                    d['well_name'],
                    num_data_points,
                    num_metrics,
                )

                log.info(f'Creating force frequency relationship chart for well {d["well_name"]}')
                create_force_frequency_relationship_charts(
                    force_freq_sheet,
                    force_freq_chart,
                    well_index,
                    d['well_name'],
                    num_data_points,  # number of twitches
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

    # maximum snapshot size is 10 seconds
    lower_x_bound = (dm['start_time'])

    upper_x_bound = (
        dm['end_time']
        if (dm['end_time'] - dm['start_time']) <= CHART_MAXIMUM_SNAPSHOT_LENGTH
        else dm['start_time'] + CHART_MAXIMUM_SNAPSHOT_LENGTH
    )

    df_column = continuous_waveforms_df.columns.get_loc(f"{dm['well_name']} - Active Twitch Force (μN)")

    well_column = xl_col_to_name(df_column)
    full_chart = wb.add_chart({"type": "scatter", "subtype": "straight"})

    # plot snapshot of waveform
    snapshot_plot_params = plotting_parameters(upper_x_bound-lower_x_bound)
    snapshot_chart = wb.add_chart({"type": "scatter", "subtype": "straight"})

    snapshot_chart.set_x_axis({"name": "Time (seconds)", "min": lower_x_bound, "max": upper_x_bound})
    snapshot_chart.set_y_axis({"name": "Active Twitch Force (μN)", "major_gridlines": {"visible": 0}})
    snapshot_chart.set_title({"name": f"Well {dm['well_name']}"})

    snapshot_chart.add_series({
            "name": "Waveform Data",
            "categories": f"='continuous-waveforms'!$A$2:$A${len(continuous_waveforms_df)}",
            "values": f"='continuous-waveforms'!${well_column}$2:${well_column}${len(continuous_waveforms_df)}",
            "line": {"color": "#1B9E77"}}
    )

    snapshot_chart.set_size({
           "width": snapshot_plot_params['chart_width'],
           "height": CHART_HEIGHT}
    )
    snapshot_chart.set_plotarea({
        'layout': {
            'x': snapshot_plot_params['x'],
            'y': 0.1,
            'width':  snapshot_plot_params['plot_width'],
            'height': 0.7}
    })

    # plot full waveform
    full_plot_params = plotting_parameters(dm['end_time'] - dm['start_time'])

    full_chart.set_x_axis({"name": "Time (seconds)", "min": dm['start_time'], "max": dm['end_time']})
    full_chart.set_y_axis({"name": "Active Twitch Force (μN)", "major_gridlines": {"visible": 0}})
    full_chart.set_title({"name": f"Well {dm['well_name']}"})

    full_chart.add_series({
            "name": "Waveform Data",
            "categories": f"='continuous-waveforms'!$A$2:$A${len(continuous_waveforms_df)}",
            "values": f"='continuous-waveforms'!${well_column}$2:${well_column}${len(continuous_waveforms_df)+1}",
            "line": {"color": "#1B9E77"} }
    )

    full_chart.set_size({
           "width": full_plot_params['chart_width'],
           "height": CHART_HEIGHT}
    )
    full_chart.set_plotarea({
        'layout': {
            'x': full_plot_params['x'],
            'y': 0.1,
            'width':  full_plot_params['plot_width'],
            'height': 0.7}
    })

    (peaks, valleys) = dm["peaks_and_valleys"]

    log.info(f'Adding peak detection series for well {dm["well_name"]}')

    add_peak_detection_series(
        waveform_charts=[snapshot_chart, full_chart],
        continuous_waveform_sheet=continuous_waveforms_sheet,
        detector_type="Peak",
        well_index=iter_idx,
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
        well_index=iter_idx,
        well_name=f"{dm['well_name']}",
        upper_x_bound_cell=dm["num_data_points"],
        indices=valleys,
        interpolated_data_function=dm["interp_data_fn"],
        time_values=dm["force"][0],
        is_optical_recording=is_optical_recording,
        minimum_value=dm["min_value"],
    )

    (well_row, well_col) = TWENTY_FOUR_WELL_PLATE.get_row_and_column_from_well_index(df_column - 1)
    snapshot_sheet.insert_chart(
        well_row * (CHART_HEIGHT_CELLS + 1), well_col * (CHART_FIXED_WIDTH_CELLS + 1), snapshot_chart
    )
    full_sheet.insert_chart(1 + iter_idx * (CHART_HEIGHT_CELLS + 1), 1, full_chart)


def aggregate_metrics_df(data: List[Dict[Any,Any]], widths: Tuple[int] = tuple([50,90])):
    """Combine aggregate metrics for each well into single DataFrame for writing.

    Args:
        data (list): list of data metrics and metadata associated with each well
        widths (tuple of ints, optional): twitch-widths to return data for. Defaults to tuple(50,90).

    Returns:
        df (DataFrame): aggregate data frame of all metric aggregate measures
    """
    def append2df(main_df: pd.DataFrame, metrics: pd.DataFrame):
        """Wrapper to append metric-specific aggregate measures to aggregate data frame.

        Args:
            main_df (DataFrame): aggregate data frame
            metrics (DataFrame): metric-specific aggregate measures
        Returns:
            main_df (DataFrame): aggregate data frame
        """
        metrics.reset_index(inplace=True)
        metrics.insert(0, 'level_0', [nm] + ['']*5)
        metrics.columns = np.arange(metrics.shape[1])

        main_df = main_df.append(metrics, ignore_index=True)

        #empty row
        main_df = main_df.append(pd.Series(['']), ignore_index=True)

        return main_df

    df = pd.DataFrame()
    df = df.append(pd.Series(['', '', ] + [d['well_name'] for d in data]), ignore_index=True)
    df = df.append(pd.Series(['', 'Treatment Description']), ignore_index=True)
    df = df.append(pd.Series(['', 'n (twitches)'] + [len(d['metrics'][0]) if not d['error_msg'] else d['error_msg'] for d in data]), ignore_index=True)
    df = df.append(pd.Series(['']), ignore_index=True)  # empty row

    combined = pd.concat([d['metrics'][1] for d in data])

    for metric_id in ALL_METRICS:
        if metric_id in [WIDTH_UUID, RELAXATION_TIME_UUID, CONTRACTION_TIME_UUID]:
            for k in widths:
                nm = CALCULATED_METRIC_DISPLAY_NAMES[metric_id].format(k)
                metric_df = combined[metric_id][k].drop(columns=['n']).T
                df = append2df(df, metric_df)
        else:
            nm = CALCULATED_METRIC_DISPLAY_NAMES[metric_id]
            metric_df = combined[metric_id].drop(columns=['n']).T.droplevel(level=-1, axis=0)
            df = append2df(df, metric_df)

    return df


def per_twitch_df(data: List[Dict[Any,Any]], widths: Tuple[int] = tuple([50,90])):
    """Combine per-twitch metrics for each well into single DataFrame for writing.

    Args:
        data (list): list of data metrics and metadata associated with each well
        widths (tuple of ints, optional): twitch-widths to return data for. Defaults to tuple([50,90]).

    Returns:
        df (DataFrame): per-twitch data frame of all metrics
    """

    df = pd.DataFrame()
    for j, d in enumerate(data): #for each well
        num_per_twitch_metrics = 0 #len(labels)
        twitch_times = [d['force'][0,i]/MICRO_TO_BASE_CONVERSION for i in d['metrics'][0].index]

        # get metrics for single well
        dm = d['metrics'][0]
        df = df.append(pd.Series([d['well_name']] + [f'Twitch {i+1}' for i in range( len(dm))]), ignore_index=True)
        df = df.append(pd.Series(['Timepoint of Twitch Contraction'] + twitch_times), ignore_index=True)
        num_per_twitch_metrics += 2

        for metric_id in ALL_METRICS:
            if metric_id in [WIDTH_UUID, RELAXATION_TIME_UUID, CONTRACTION_TIME_UUID]:
                for twitch_width in widths:
                    values = [f'{CALCULATED_METRIC_DISPLAY_NAMES[metric_id].format(twitch_width)}']
                    df = df.append(pd.Series(values + list(dm[metric_id][twitch_width])), ignore_index=True)
                    num_per_twitch_metrics += 1
            else:
                values = [CALCULATED_METRIC_DISPLAY_NAMES[metric_id]]
                df = df.append(pd.Series(values + list(dm[metric_id])), ignore_index=True)
                num_per_twitch_metrics += 1

        for _ in range(5):
            df = df.append(pd.Series([""]), ignore_index=True)
            num_per_twitch_metrics += 1

    df.fillna("", inplace=True)
    return (df, num_per_twitch_metrics)
