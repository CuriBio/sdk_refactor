import numpy as np
import datetime
import pandas as pd
from scipy import interpolate

from plate_recording import WellFile
from peak_detection import peak_detector, find_twitch_indices, data_metrics
from constants import *

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

w = WellFile('./MA201110001__2020_09_03_213024__A1.h5')
pv = peak_detector(w.noise_filtered_magnetic_data)
twitch_indices = find_twitch_indices(pv)
dm = data_metrics(pv, w.force)

metadata = {
    'A': ['Recording Information:', None, None, None, 'Device Information:', None, None, None, None, None, 'Output Format:', None, None],
    'B': [None, 'Plate Barcode', 'UTC Timestamp of Beginning of Recording', None, None, 'H5 File Layout Version', 'Mantarray Serial Number', 'Software Release Version', 'Software Build Version', 'Firmware Version (Main Controller)', None, 'SDK Version', 'File Creation Timestamp'],
    'C': [None, w[PLATE_BARCODE_UUID], w[UTC_BEGINNING_RECORDING_UUID].replace(tzinfo=None), None, None, w['File Format Version'], w[MANTARRAY_SERIAL_NUMBER_UUID], w[SOFTWARE_RELEASE_VERSION_UUID], w[SOFTWARE_BUILD_NUMBER_UUID], w[MAIN_FIRMWARE_VERSION_UUID], None, '0.1', datetime.datetime.utcnow().replace(microsecond=0)],
    }

metadata_df = pd.DataFrame(metadata)

max_time = w.raw_tissue_magnetic_data[0][-1]
q = np.arange(INTERPOLATED_DATA_PERIOD_US, max_time, INTERPOLATED_DATA_PERIOD_US)
interp_data_fn = interpolate.interp1d(w.force[0], w.force[1])

interp_data = interp_data_fn(q)
min_value = min(interp_data)
interp_data -= min_value
interp_data *= MICRO_TO_BASE_CONVERSION

continuous_waveforms = {
    'Time (seconds)': q / MICRO_TO_BASE_CONVERSION,
    'A1 - Active Twitch Force': interp_data  #interp_data_fn(q) #w.force[1][:-1],
    }
continuous_waveforms_df = pd.DataFrame(continuous_waveforms)


def xl_col_to_name(col, col_abs=False):
    """
    Convert a zero indexed column cell reference to a string.

    Args:
       col:     The cell column. Int.
       col_abs: Optional flag to make the column absolute. Bool.

    Returns:
        Column style string.
    """
    col_num = col
    if col_num < 0:
        warn("Col number %d must be >= 0" % col_num)
        return None

    col_num += 1  # Change to 1-index.
    col_str = ''
    col_abs = '$' if col_abs else ''

    while col_num:
        # Set remainder from 1 .. 26
        remainder = col_num % 26

        if remainder == 0:
            remainder = 26

        # Convert the remainder to a character.
        col_letter = chr(ord('A') + remainder - 1)

        # Accumulate the column letters, right to left.
        col_str = col_letter + col_str

        # Get the next order of magnitude.
        col_num = int((col_num - 1) / 26)

    return col_abs + col_str


def add_peak_detection_series(
    waveform_chart,
    continuous_waveform_sheet,
    detector_type: str,
    well_index: int,
    well_name: str,
    upper_x_bound_cell: int,
    indices,
    interpolated_data_function: interpolate.interpolate.interp1d,
    time_values,
    minimum_value: float,
) -> None:
    label = "Relaxation" if detector_type == "Valley" else "Contraction"
    offset = 1 if detector_type == "Valley" else 0
    marker_color = "#D95F02" if detector_type == "Valley" else "#7570B3"

    #continuous_waveform_sheet = self._workbook.get_worksheet_by_name(CONTINUOUS_WAVEFORM_SHEET_NAME)
    result_column = xl_col_to_name(PEAK_VALLEY_COLUMN_START + (well_index * 2) + offset)
    continuous_waveform_sheet.write(f"{result_column}1", f"{well_name} {detector_type} Values")

    for idx in indices:
        idx_time = time_values[idx] / MICRO_TO_BASE_CONVERSION
        shifted_idx_time = idx_time - time_values[0] / MICRO_TO_BASE_CONVERSION

        uninterpolated_time_seconds = round(idx_time, 2)
        shifted_time_seconds = round(shifted_idx_time, 2)

        if False:#self._is_optical_recording:
            row = int(shifted_time_seconds * MICRO_TO_BASE_CONVERSION / INTERPOLATED_DATA_PERIOD_US) #self._interpolated_data_period)

            value = (
                interpolated_data_function(uninterpolated_time_seconds * MICRO_TO_BASE_CONVERSION)
                - minimum_value
            ) * MICRO_TO_BASE_CONVERSION
        else:
            row = shifted_time_seconds * int(1 / INTERPOLATED_DATA_PERIOD_SECONDS) + 1

            interpolated_data = interpolated_data_function(
                uninterpolated_time_seconds * MICRO_TO_BASE_CONVERSION
            )
            #interpolated_data *= -1  # magnetic waveform is flipped
            interpolated_data -= minimum_value
            value = interpolated_data * MICRO_TO_BASE_CONVERSION

        continuous_waveform_sheet.write(f"{result_column}{row}", value)

    if waveform_chart is not None:  # Tanner (11/11/20): chart is None when skipping chart creation
        waveform_chart.add_series(
            {
                "name": label,
                "categories": f"='continuous-waveforms'!$A$2:$A${upper_x_bound_cell}",
                "values": f"='continuous-waveforms'!${result_column}$2:${result_column}${upper_x_bound_cell}",
                #"values": f"='continuous-waveforms'!$CY$2:$CY${upper_x_bound_cell}",
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

    # well_row, well_col = TWENTY_FOUR_WELL_PLATE.get_row_and_column_from_well_index(well_index)

    # force_frequency_sheet.insert_chart(
    #     1 + well_row * (CHART_HEIGHT_CELLS + 1),
    #     1 + well_col * (CHART_FIXED_WIDTH_CELLS + 1),
    #     force_frequency_chart,
    # )

    force_frequency_sheet.insert_chart(
        1 + (CHART_HEIGHT_CELLS + 1),
        1 + (CHART_FIXED_WIDTH_CELLS + 1),
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
    breakpoint()

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

    #well_row, well_col = TWENTY_FOUR_WELL_PLATE.get_row_and_column_from_well_index(well_index)

    # frequency_chart_sheet.insert_chart(
    #     1 + well_row * (CHART_HEIGHT_CELLS + 1),
    #     1 + well_col * (CHART_FIXED_WIDTH_CELLS + 1),
    #     frequency_chart,
    # )
    frequency_chart_sheet.insert_chart(
        1 + (CHART_HEIGHT_CELLS + 1),
        1 + (CHART_FIXED_WIDTH_CELLS + 1),
        frequency_chart,
    )


def write_xlsx():
    with pd.ExcelWriter('test.xlsx') as writer:
        metadata_df.to_excel(writer, sheet_name='metadata', index=False, header=False)
        ws = writer.sheets['metadata']
        for i_col_idx, i_col_width in ((0,25), (1,40), (2,25)):
            ws.set_column(i_col_idx, i_col_idx, i_col_width)


        continuous_waveforms_df.to_excel(writer, sheet_name='continuous-waveforms', index=False)
        ws = writer.sheets['continuous-waveforms']
        ws.set_column(0, 0, 18)

        for iter_well_idx in range(1,24):
            ws.set_column(iter_well_idx, iter_well_idx, 13)

        #waveform snapshot/full
        lower_x_bound = (
            0 if continuous_waveforms_df['Time (seconds)'].iloc[-1] <= CHART_WINDOW_NUM_SECONDS 
            else int((continuous_waveforms_df['Time (seconds)'].iloc[-1] - CHART_WINDOW_NUM_SECONDS) // 2)
        )

        upper_x_bound = (
            0 if continuous_waveforms_df['Time (seconds)'].iloc[-1] <= CHART_WINDOW_NUM_SECONDS 
            else int((continuous_waveforms_df['Time (seconds)'].iloc[-1] + CHART_WINDOW_NUM_SECONDS) // 2)
        )

        wb = writer.book
        snapshot_chart = wb.add_chart({"type": "scatter", "subtype": "straight"})
        snapshot_sheet = wb.add_worksheet('continuous-waveform-snapshot')

        full_chart = wb.add_chart({"type": "scatter", "subtype": "straight"})
        full_sheet = wb.add_worksheet('full-continuous-waveform-plots')

        snapshot_chart.set_x_axis({"name": "Time (seconds)", "min": lower_x_bound, "max": upper_x_bound})
        snapshot_chart.set_y_axis({"name": "Active Twitch Force (μN)", "major_gridlines": {"visible": 0}})
        snapshot_chart.set_size({"width": CHART_FIXED_WIDTH, "height": CHART_HEIGHT})
        snapshot_chart.set_title({"name": f"Well A1"})

        full_chart.set_x_axis({"name": "Time (seconds)", "min": 0, "max": continuous_waveforms_df['Time (seconds)'].iloc[-1]})
        full_chart.set_y_axis({"name": "Active Twitch Force (μN)", "major_gridlines": {"visible": 0}})

        full_chart.set_size({
            "width": CHART_FIXED_WIDTH // 2 + (DEFAULT_CELL_WIDTH * continuous_waveforms_df['Time (seconds)'].iloc[-1] // SECONDS_PER_CELL),
            "height": CHART_HEIGHT,
        })
        full_chart.set_title({"name": f"Well A1"})


        snapshot_chart.add_series({
            "name": "Waveform Data",
            "categories": f"='continuous-waveforms'!$A$2:$A${len(continuous_waveforms_df)}",
            #"values": f"='continuous-waveforms'!${continuous_waveforms_df.columns[1]}$2:${continuous_waveforms_df.columns[1]}${len(continuous_waveforms_df)}",
            "values": f"='continuous-waveforms'!$B$2:$B${len(continuous_waveforms_df)}",
            "line": {"color": "#1B9E77"},
        })

        full_chart.add_series({
            "name": "Waveform Data",
            "categories": f"='continuous-waveforms'!$A$2:$A${len(continuous_waveforms_df)}",
            "values": f"='continuous-waveforms'!$B$2:$B${len(continuous_waveforms_df)}",
            "line": {"color": "#1B9E77"},
        })

        (peaks, valleys) = pv
        add_peak_detection_series(snapshot_chart, writer.sheets['continuous-waveforms'], "Peak", 1, "A1", len(w.force[0]), peaks, interp_data_fn, w.force[0], min_value)
        add_peak_detection_series(snapshot_chart, writer.sheets['continuous-waveforms'], "Valley", 1, "A1", len(w.force[0]), valleys, interp_data_fn, w.force[0], min_value)

        add_peak_detection_series(full_chart, writer.sheets['continuous-waveforms'], "Peak", 1, "A1", len(w.force[0]), peaks, interp_data_fn, w.force[0], min_value)
        add_peak_detection_series(full_chart, writer.sheets['continuous-waveforms'], "Valley", 1, "A1", len(w.force[0]), valleys, interp_data_fn, w.force[0], min_value)

        snapshot_sheet.insert_chart(1+(CHART_HEIGHT_CELLS + 1), 1 + (CHART_FIXED_WIDTH_CELLS + 1), snapshot_chart)
        full_sheet.insert_chart(1+(CHART_HEIGHT_CELLS + 1), 1, full_chart)

        #aggregate metrics
        aggregate_df = aggregate_metrics_df([dm])
        aggregate_df.to_excel(writer, sheet_name='aggregate-metrics', index=False, header=False)

        (pdf, num_metrics) = per_twitch_df([dm])
        pdf.to_excel(writer, sheet_name='per-twitch-metrics', index=False, header=False)

        freq_vs_time_sheet = wb.add_worksheet(FORCE_FREQUENCY_RELATIONSHIP_SHEET)
        freq_vs_time_chart = wb.add_chart({"type": "scatter", "subtype": "straight"})

        create_frequency_vs_time_charts(
            freq_vs_time_sheet,
            freq_vs_time_chart,
            0,
            "A1",
            dm[1][AMPLITUDE_UUID]["n"], #number of twitches
            list(dm[0]), # time values
            num_metrics,
        )

        force_freq_sheet = wb.add_worksheet(TWITCH_FREQUENCIES_CHART_SHEET_NAME)
        force_freq_chart = wb.add_chart({"type": "scatter", "subtype": "straight"})

        create_force_frequency_relationship_charts(
            force_freq_sheet,
            force_freq_chart,
            0,
            "A1",
            dm[1][AMPLITUDE_UUID]["n"],
            num_metrics,
        )


"""
df = pd.DataFrame({
    'metrics': ['','','',''] + ['Twitch Amplitude', '', '', '', '', '', ''] * 5,
    'aggregates': ['', 'Treatment', 'n', ''] + ['mean', 'std', 'cov', 'SEM', 'Min' , 'Max', ''] * 5,
    'A1': ['A1','','242',''] + [dm[1][u]['mean'], dm[1][u]['std'], '', '', dm[1][u]['min'], dm[1][u]['max'], ''] * 5,
    'B1': ['B1','','242',''] + [dm[1][u]['mean'], dm[1][u]['std'], '', '', dm[1][u]['min'], dm[1][u]['max'], ''] * 5,
    })
"""

def append_aggregates(d):
    return [d['mean'], d['std'], d['cov'], d['sem'], d['min'], d['max'], '']

def aggregate_metrics_df(dms):
    keys = dms[0][1].keys()
    empty = [''] * 6
    cols = [['', '', '', '']] * len(dms)

    labels = [''] * 4
    aggregates = ['', 'Treatment Description', 'n (twitches)', '']


    for m in ALL_METRICS:
        if m in keys:
            if m in [WIDTH_UUID, RELAXATION_TIME_UUID, CONTRACTION_TIME_UUID]:
                for k in dms[0][1][m].keys():
                    labels += [CALCULATED_METRIC_DISPLAY_NAMES[m].format(k)] + empty
                    aggregates += ['Mean', 'StDev', 'CoV', 'SEM', 'Min', 'Max', '']

                for i, dm in enumerate(dms):
                    for k in dm[1][m].keys():
                        cols[i] += append_aggregates(dm[1][m][k])
            else:
                labels += [CALCULATED_METRIC_DISPLAY_NAMES[m]] + empty
                aggregates += ['Mean', 'StDev', 'CoV', 'SEM', 'Min', 'Max', '']

                for i, dm in enumerate(dms):
                    cols[i] += append_aggregates(dm[1][m])

    data = {'metrics': labels, 'aggregates': aggregates}
    for i,c in enumerate(cols):
        data[str(i)] = c

    return pd.DataFrame(data)


def per_twitch_df(dms):
    idx = list(dms[0][0].keys())[0]
    keys = list(dms[0][0][idx].keys())
    labels = []
    metrics = [[]]*len(dms[0][0].keys())

    labels += ['', 'Timepoint of Twitch Contraction']
    for m in ALL_METRICS:
        if m in keys:
            if m in [WIDTH_UUID, RELAXATION_TIME_UUID, CONTRACTION_TIME_UUID]:
                for k in dms[0][1][m].keys():
                    labels += [CALCULATED_METRIC_DISPLAY_NAMES[m].format(k)]
            else:
                labels += [CALCULATED_METRIC_DISPLAY_NAMES[m]]
    labels += [''] * 4
    num_per_twitch_metrics = len(labels)
    labels *= len(dms)

    for i,k in enumerate(dms[0][0].keys()): #for each twitch
        values = []
        for j,dm in enumerate(dms): #for each well
            values += [f'Twitch {i+1}', k / MICRO_TO_BASE_CONVERSION] # add time point
            for m in ALL_METRICS:
                if m in keys:
                    if m == WIDTH_UUID:
                        for q in dm[0][k][m].keys():
                            values += [dm[0][k][m][q][WIDTH_VALUE_UUID]]
                    elif m in [RELAXATION_TIME_UUID, CONTRACTION_TIME_UUID]:
                        for q in dm[0][k][m].keys():
                            values += [dm[0][k][m][q][TIME_VALUE_UUID]]
                    else:
                        values += [dm[0][k][m]]
            values += [''] * 4
        metrics[i] = values

    data = {'labels': labels}
    for i,m in enumerate(metrics):
        data[i] = m

    return (pd.DataFrame(data), num_per_twitch_metrics)

write_xlsx()
