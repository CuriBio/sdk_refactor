# -*- coding: utf-8 -*-
"""Metrics for skeletal and cardiac muscles.

If a new metric is requested, you must implement `fit`,
`add_per_twitch_metrics`, and `add_aggregate_metrics`.
"""

# for hashing dataframes
import abc
import json
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union
from uuid import UUID

from nptyping import NDArray
import numpy as np
import pandas as pd
import polars as pl
from pandas import DataFrame
from pandas import Series
from pulse3D.transforms import get_time_window_indices

from .compression_cy import interpolate_x_for_y_between_two_points
from .compression_cy import interpolate_y_for_x_between_two_points
from .constants import DEFAULT_TWITCH_WIDTH_PERCENTS
from .constants import INTERPOLATED_DATA_PERIOD_SECONDS
from .constants import MICRO_TO_BASE_CONVERSION
from .constants import PRIOR_VALLEY_INDEX_UUID
from .constants import SUBSEQUENT_PEAK_INDEX_UUID
from .constants import SUBSEQUENT_VALLEY_INDEX_UUID


class MetricCalculator:
    def __init__(
        self,
        filtered_data,
        twitch_indices,
        peak_and_valley_indices,
        twitch_width_percents,
        baseline_width_percents,
        rounded=False,
    ):
        self._peak_and_valley_indices = peak_and_valley_indices
        self._filtered_data = filtered_data
        self._twitch_indices = twitch_indices
        self._twitch_width_percents = sorted(twitch_width_percents)
        self._baseline_width_percents = sorted(baseline_width_percents)
        self._rounded = rounded

        self._metrics_data = dict()
        # self._metrics_dfs = dict()

    def __getitem__(self, metric):
        vals = self._metrics_data.get(metric)
        if vals is None:
            res = getattr(self, f"_calculate_{metric}")()
            self._metrics_data.update(res)
            vals = self._metrics_data[metric]
        return vals

    def _calculate_twitch_amplitude(self):
        # TODO this needs to be set to the baseline width
        twitch_width = 90

        amplitudes: Dict[int, float] = dict()

        {"peak_idx": int, "width": int, "data_type": str, "label": str, "val": float}

        def foo():


        # TODO need to mess around with polars DFs to figure out the best way to handle this

        for twitch_data_series in self["twitch_width_coordinates"]:
            twitch_peak_idx = int(twitch_data_series.name)
            twitch_peak_x, twitch_peak_y = self._filtered_data[:, twitch_peak_idx]

            # C10 in the metric definition diagram is the C point at 90% twitch width
            c10x = twitch_data_series["time", "contraction", twitch_width]
            c10y = twitch_data_series["force", "contraction", twitch_width]
            r90x = twitch_data_series["time", "relaxation", twitch_width]
            r90y = twitch_data_series["force", "relaxation", twitch_width]

            twitch_base_y = interpolate_y_for_x_between_two_points(twitch_peak_x, c10x, c10y, r90x, r90y)
            amplitude_y = twitch_peak_y - twitch_base_y

            amplitudes[twitch_peak_idx] = amplitude_y

        return {"twitch_amplitude": pd.Series(amplitudes)}

    def _calculate_twitch_fraction_amplitude(self):
        amplitudes = self["twitch_amplitude"]
        return {"twitch_fraction_amplitude": amplitudes / np.nanmax(amplitudes)}

    def _calculate_twitch_frequency(self):
        twitch_idx_list = list(self._twitch_indices.keys())
        # final peak is needed to get freq for final twitch
        twitch_idx_list.append(self._twitch_indices[twitch_idx_list[-1]][SUBSEQUENT_PEAK_INDEX_UUID])

        timepoints = self._filtered_data[0]
        twitch_timepoints = timepoints[np.array(twitch_idx_list)]
        twitch_periods = np.diff(twitch_timepoints)
        twitch_freqs = MICRO_TO_BASE_CONVERSION / twitch_periods

        return {"twitch_frequency": pd.Series(twitch_freqs)}

    def _calculate_twitch_width(self):
        return self.__calculate_twitch_width_and_coordinates()

    def _calculate_twitch_width_coordinates(self):
        return self.__calculate_twitch_width_and_coordinates()

    def __calculate_twitch_width_and_coordinates(self):
        width_df = pl.DataFrame({}, schema={"peak_idx": int, "width": int, "val": float})
        coordinate_df = pl.DataFrame(
            {}, schema={"peak_idx": int, "width": int, "data_type": str, "label": str, "val": float}
        )

        timepoints_arr = self._filtered_data[0]
        force_amplitudes_arr = self._filtered_data[1]

        for iter_twitch_peak_idx in self._twitch_indices:
            peak_force = force_amplitudes_arr[iter_twitch_peak_idx]

            # calculate magnitude of rise
            prior_valley_idx = self._twitch_indices[iter_twitch_peak_idx][PRIOR_VALLEY_INDEX_UUID]  # type: ignore
            prior_valley_force = force_amplitudes_arr[prior_valley_idx]
            magnitude_of_rise = peak_force - prior_valley_force

            # calculate magnitude of fall
            subsequent_valley_idx = self._twitch_indices[iter_twitch_peak_idx][SUBSEQUENT_VALLEY_INDEX_UUID]  # type: ignore
            subsequent_valley_force = force_amplitudes_arr[subsequent_valley_idx]
            magnitude_of_fall = peak_force - subsequent_valley_force

            rising_idx = iter_twitch_peak_idx - 1
            falling_idx = iter_twitch_peak_idx + 1

            for iter_percent in self._twitch_width_percents:
                rising_threshold = peak_force - (iter_percent / 100) * magnitude_of_rise
                falling_threshold = peak_force - (iter_percent / 100) * magnitude_of_fall

                # move to the left from the twitch peak until the rising threshold is reached
                while force_amplitudes_arr[rising_idx] > rising_threshold:
                    rising_idx -= 1
                # move to the right from the twitch peak until the falling threshold is reached
                while force_amplitudes_arr[falling_idx] > falling_threshold:
                    falling_idx += 1

                interpolated_rising_timepoint = interpolate_x_for_y_between_two_points(
                    rising_threshold,
                    timepoints_arr[rising_idx],
                    force_amplitudes_arr[rising_idx],
                    timepoints_arr[rising_idx + 1],
                    force_amplitudes_arr[rising_idx + 1],
                )
                interpolated_falling_timepoint = interpolate_x_for_y_between_two_points(
                    falling_threshold,
                    timepoints_arr[falling_idx],
                    force_amplitudes_arr[falling_idx],
                    timepoints_arr[falling_idx - 1],
                    force_amplitudes_arr[falling_idx - 1],
                )
                width_val = interpolated_falling_timepoint - interpolated_rising_timepoint

                if self._rounded:
                    width_val = int(round(width_val, 0))
                    interpolated_falling_timepoint = int(round(interpolated_falling_timepoint, 0))
                    interpolated_rising_timepoint = int(round(interpolated_rising_timepoint, 0))
                    rising_threshold = int(round(rising_threshold, 0))
                    falling_threshold = int(round(falling_threshold, 0))

                # fill width-value dictionary
                width_df.vstack(
                    pl.DataFrame(
                        {
                            "peak_idx": iter_twitch_peak_idx,
                            "width": iter_percent,
                            "val": width_val / MICRO_TO_BASE_CONVERSION,
                        }
                    ),
                    in_place=True,
                )

                coordinate_dicts = [
                    {"data_type": "time", "label": "contraction", "val": rising_threshold},
                    {"data_type": "time", "label": "relaxation", "val": falling_threshold},
                    {"data_type": "force", "label": "contraction", "val": rising_threshold},
                    {"data_type": "force", "label": "relaxation", "val": falling_threshold},
                ]
                coordinate_dicts = [
                    {"peak_idx": iter_twitch_peak_idx, "width": iter_percent, **d} for d in coordinate_dicts
                ]

                coordinate_df.vstack(pl.DataFrame(coordinate_dicts), in_place=True)

        return {"twitch_width": width_df, "twitch_width_coordinates": coordinate_df}

    def _calculate_twitch_contraction_velocity(self):
        return self.__calculate_twitch_velocity(True)

    def _calculate_twitch_relaxation_velocity(self):
        return self.__calculate_twitch_velocity(False)

    def __calculate_twitch_velocity(self, is_contraction):
        initial_width = min(self._twitch_width_percents)
        final_width = max(self._twitch_width_percents)

        coord_type = "contraction" if is_contraction else "relaxation"

        velocity = []

        for twitch_data in self["twitch_width_coordinates"].values():
            x_initial = twitch_data["time"][coord_type][initial_width]
            y_initial = twitch_data["force"][coord_type][initial_width]

            x_final = twitch_data["time"][coord_type][final_width]
            y_final = twitch_data["force"][coord_type][final_width]

            # change in force / change in time
            velocity.append(abs((y_initial - y_final) / (x_initial - x_final)) * MICRO_TO_BASE_CONVERSION)

        return {f"twitch_{coord_type}_velocity": pd.Series(velocity)}

    def _calculate_twitch_interval_irregularity(self):
        list_of_twitch_indices = list(self._twitch_indices.keys())
        num_twitches = len(list_of_twitch_indices)

        timepoints = self._filtered_data[0]

        estimates = {list_of_twitch_indices[0]: None}
        for twitch in range(1, num_twitches - 1):
            last_twitch_index = list_of_twitch_indices[twitch - 1]
            current_twitch_index = list_of_twitch_indices[twitch]
            next_twitch_index = list_of_twitch_indices[twitch + 1]

            last_interval = timepoints[current_twitch_index] - timepoints[last_twitch_index]
            current_interval = timepoints[next_twitch_index] - timepoints[current_twitch_index]
            interval = abs(current_interval - last_interval)

            estimates[current_twitch_index] = interval
        estimates[list_of_twitch_indices[-1]] = None

        return {"twitch_interval_irregularity": pd.Series(estimates) / MICRO_TO_BASE_CONVERSION}

    def _calculate_twitch_auc(self):
        width_percent = 90  # what percent of repolarization to use as the bottom limit for calculating AUC
        estimates_dict: Dict[int, Union[int, float]] = dict()

        timepoints, force = self._filtered_data

        for twitch_peak_idx, twitch_data in self["twitch_width_coordinates"].items():
            start_timepoint = twitch_data["time"]["contraction"][width_percent]
            stop_timepoint = twitch_data["time"]["relaxation"][width_percent]

            auc_window = (timepoints >= start_timepoint) & (timepoints <= stop_timepoint)
            auc_total = np.trapz(force[auc_window], dx=INTERPOLATED_DATA_PERIOD_SECONDS)

            if self._rounded:
                auc_total = int(round(auc_total, 0))

            estimates_dict[twitch_peak_idx] = auc_total

        return {"twitch_auc": pd.Series(estimates_dict)}

    def _calculate_twitch_contraction_time(self):
        return self.__calculate_peak_baseline_time_diff(True)

    def _calculate_twitch_relaxation_time(self):
        return self.__calculate_peak_baseline_time_diff(False)

    def __calculate_peak_baseline_time_diff(self, is_contraction):
        # TODO test
        # dictionary of time differences for each peak
        coord_label = "contraction" if is_contraction else "relaxation"

        def diff_fn(x, y):
            return x - y if is_contraction else y - x

        estimates_dict = {}  # type: ignore

        for twitch_idx, twitch_data in self["twitch_width_coordinates"].items():
            estimates_dict[twitch_idx] = {}

            for width_percent in self._twitch_width_percents:
                percent = 100 - width_percent if is_contraction else width_percent
                percent_time = twitch_data["time"][coord_label][percent]

                peak_time = self._filtered_data[0, twitch_idx]

                estimates_dict[twitch_idx][width_percent] = (
                    diff_fn(peak_time, percent_time) / MICRO_TO_BASE_CONVERSION
                )

        return estimates_dict


class BaseMetric:
    """Any new metric needs to implement three methods.

    1) estimate the per-twitch values
    2) add the per-twitch values to the per-twitch dictionary
    3) add the aggregate statistics to the aggregate dictionary.

    Most metrics will estimate a single value per twitch, but others are
    nested (twitch widths, time-to/from peak, etc.)
    """

    def __init__(self, rounded: bool = False, **kwargs: Dict[str, Any]):
        self.rounded = rounded

    @abc.abstractmethod
    def fit(
        self,
        peak_and_valley_indices: Tuple[NDArray[int], NDArray[int]],
        filtered_data: NDArray[(2, Any), int],
        twitch_indices: Dict[int, Dict[UUID, Optional[int]]],
        **kwargs: Dict[str, Any],
    ) -> Series:
        pass

    def add_per_twitch_metrics(
        self, main_df: DataFrame, metric_id: UUID, metrics: Union[NDArray[int], NDArray[float]]
    ) -> None:
        """Add estimated per-twitch metrics to per-twitch DataFrame.

        Args:
            main_df (DataFrame): DataFrame storing per-twitch metrics
            metric_id (UUID): UUID of metric to add
            metrics (Union[NDArray[int], NDArray[float]]): estimated per-twitch metrics
        """
        main_df[metric_id] = metrics

    def add_aggregate_metrics(
        self, aggregate_df: DataFrame, metric_id: UUID, metrics: Union[NDArray[int], NDArray[float]]
    ) -> None:
        """Add estimated metrics to aggregate DataFrame.

        Args:
            aggregate_df (DataFrame): DataFrame storing aggregate metrics
            metric_id (UUID): UUID of metric to add
            metrics (Union[NDArray[int], NDArray[float]]): estimated per-twitch metrics
        """
        aggregate_metrics = self.create_statistics_df(metrics, rounded=self.rounded)
        aggregate_df[metric_id] = aggregate_metrics

    @classmethod
    def create_statistics_df(cls, metric: NDArray[int], rounded: bool = False) -> DataFrame:
        """Calculate various statistics for a specific metric.

        Args:
        metric: a 1D array of integer values of a specific metric results

        Returns:
        d: of the average statistics of that metric in which the metrics are the key and
        average statistics are the value
        """
        statistics: Dict[str, Any] = {k: None for k in ["n", "Mean", "StDev", "CoV", "SEM", "Min", "Max"]}
        statistics["n"] = len(metric)

        if len(metric) > 0:
            statistics["Mean"] = np.nanmean(metric)
            statistics["StDev"] = np.nanstd(metric)
            statistics["CoV"] = statistics["StDev"] / statistics["Mean"]
            statistics["SEM"] = statistics["StDev"] / statistics["n"] ** 0.5
            statistics["Min"] = np.nanmin(metric)
            statistics["Max"] = np.nanmax(metric)

            if rounded:
                for iter_key, iter_value in statistics.items():
                    statistics[iter_key] = int(round(iter_value))

        statistics = {k: [v] for k, v in statistics.items()}
        statistics_df = pd.DataFrame.from_dict(statistics)
        return statistics_df


class TwitchAmplitude(BaseMetric):
    """Calculate the amplitude for each twitch."""

    def __init__(self, rounded: bool = False, **kwargs: Dict[str, Any]):
        super().__init__(rounded=rounded, **kwargs)

    def fit(
        self,
        peak_and_valley_indices: Tuple[NDArray[int], NDArray[int]],
        filtered_data: NDArray[(2, Any), int],
        twitch_indices: Dict[int, Dict[UUID, Optional[int]]],
        **kwargs: Dict[str, Any],
    ) -> Series:

        amplitudes = self.calculate_amplitudes(
            twitch_indices=twitch_indices, filtered_data=filtered_data, rounded=self.rounded
        )

        return amplitudes

    @staticmethod
    def calculate_amplitudes(
        twitch_indices: Dict[int, Dict[UUID, Optional[int]]],
        filtered_data: NDArray[(2, Any), int],
        rounded: bool = False,
    ) -> Series:
        """Get the amplitudes for all twitches.

        Given amplitude of current peak, and amplitude of prior/subsequent valleys, twitch amplitude
        is calculated as the mean distance from peak to both valleys.

        Args:
            twitch_indices: a dictionary in which the key is an integer representing (TODO fix this for every metric) of interest and the value is an inner dictionary with various UUID
                of prior/subsequent peaks and valleys and their index values.

            filtered_data: a 2D array of the time and value (magnetic, voltage, displacement, force)
                data after it has gone through noise filtering

        Returns:
            Pandas Series of float values representing the amplitude of each twitch
        """
        twitch_width = 90

        _, coordinates = TwitchWidth.calculate_twitch_widths(
            filtered_data=filtered_data,
            twitch_indices=twitch_indices,
            twitch_width_percents=(twitch_width, 10),
            rounded=rounded,
            as_dict=True,  # Tanner (1/20/23): using as_dict=True here to speed this up. This calculation is performed in the MA Controller, so it must be as fast as possible
        )

        estimates_dict: Dict[int, float] = dict()

        for twitch_peak_idx, twitch_data in coordinates.items():
            twitch_peak_x, twitch_peak_y = filtered_data[:, twitch_peak_idx]

            # C10 in the metric definition diagram is the C point at 90% twitch width
            c10x = twitch_data["time"]["contraction"][twitch_width]
            c10y = twitch_data["force"]["contraction"][twitch_width]
            r90x = twitch_data["time"]["relaxation"][twitch_width]
            r90y = twitch_data["force"]["relaxation"][twitch_width]

            twitch_base_y = interpolate_y_for_x_between_two_points(twitch_peak_x, c10x, c10y, r90x, r90y)
            amplitude_y = twitch_peak_y - twitch_base_y

            estimates_dict[twitch_peak_idx] = amplitude_y

        estimates = pd.Series(estimates_dict)
        return estimates


class TwitchFractionAmplitude(TwitchAmplitude):
    """Calculate the fraction of max amplitude for each twitch."""

    def __init__(self, **kwargs: Dict[str, Any]):
        super().__init__(rounded=False, **kwargs)

    def fit(
        self,
        peak_and_valley_indices: Tuple[NDArray[int], NDArray[int]],
        filtered_data: NDArray[(2, Any), int],
        twitch_indices: Dict[int, Dict[UUID, Optional[int]]],
        **kwargs: Dict[str, Any],
    ) -> Series:
        amplitudes = super().fit(
            peak_and_valley_indices=peak_and_valley_indices,
            twitch_indices=twitch_indices,
            filtered_data=filtered_data,
        )
        estimates = amplitudes / np.nanmax(amplitudes)
        return estimates


class TwitchWidth(BaseMetric):
    """Calculate the width of each twitch at fraction of twitch."""

    def __init__(
        self,
        rounded: bool = False,
        twitch_width_percents: Tuple[int, ...] = DEFAULT_TWITCH_WIDTH_PERCENTS,
        **kwargs: Dict[str, Any],
    ):
        super().__init__(rounded=rounded, **kwargs)

        self.twitch_width_percents = twitch_width_percents

    def fit(
        self,
        peak_and_valley_indices: Tuple[NDArray[int], NDArray[int]],
        filtered_data: NDArray[(2, Any), int],
        twitch_indices: Dict[int, Dict[UUID, Optional[int]]],
        **kwargs: Dict[str, Any],
    ) -> DataFrame:
        widths, _ = self.calculate_twitch_widths(
            filtered_data=filtered_data,
            twitch_indices=twitch_indices,
            twitch_width_percents=tuple(self.twitch_width_percents),
            rounded=self.rounded,
        )

        return widths

    def add_aggregate_metrics(self, aggregate_df: DataFrame, metric_id: UUID, metrics: DataFrame) -> None:
        for iter_percent in self.twitch_width_percents:
            estimates = metrics[iter_percent]
            aggregate_estimates = self.create_statistics_df(estimates, rounded=self.rounded)
            aggregate_df[metric_id, iter_percent] = aggregate_estimates

    @staticmethod
    def calculate_twitch_widths(
        filtered_data: NDArray[(2, Any), int],
        twitch_indices: Dict[int, Dict[UUID, Optional[int]]],
        twitch_width_percents: Tuple[int, ...] = DEFAULT_TWITCH_WIDTH_PERCENTS,
        rounded: bool = False,
        as_dict: bool = False,
    ) -> Tuple[DataFrame, DataFrame]:
        """Determine twitch width between 10-90% down to the nearby valleys.

        TODO
        Args:
            twitch_indices: a HashableDataFrame in which index is an integer representing the time points
                of all the peaks of interest and columns are UUIDs of prior/subsequent peaks and valleys and
                their index values.
            filtered_data: Tuple[Tuple] of time and value (magnetic, voltage, displacement, force)
                data after it has gone through noise filtering

        Returns:
            width_df: DataFrame, where each index is an integer representing the time points, and each column
            is a percent-twitch width of all the peaks of interest

            coordinate_df: MultiIndex DataFrame, where each index is an integer representing the time points,
            and each  column level corresponds to the time (X) / force(Y), contraction (rising) / relaxation
            (falling), and percent-twitch width coordinates
        """
        coordinate_dict: Dict[int, Dict[str, Dict[str, Any]]] = dict()

        width_dict: Dict[int, Dict[int, Any]] = {twitch_index: {} for twitch_index in twitch_indices}

        timepoints_arr = filtered_data[0]
        force_amplitudes_arr = filtered_data[1]

        twitch_width_percents = sorted(twitch_width_percents)  # type: ignore

        for iter_twitch_peak_idx in twitch_indices:
            peak_force = force_amplitudes_arr[iter_twitch_peak_idx]

            # calculate magnitude of rise
            prior_valley_idx = twitch_indices[iter_twitch_peak_idx][PRIOR_VALLEY_INDEX_UUID]  # type: ignore
            prior_valley_force = force_amplitudes_arr[prior_valley_idx]
            magnitude_of_rise = peak_force - prior_valley_force

            # calculate magnitude of fall
            subsequent_valley_idx = twitch_indices[iter_twitch_peak_idx][SUBSEQUENT_VALLEY_INDEX_UUID]  # type: ignore
            subsequent_valley_force = force_amplitudes_arr[subsequent_valley_idx]
            magnitude_of_fall = peak_force - subsequent_valley_force

            rising_idx = iter_twitch_peak_idx - 1
            falling_idx = iter_twitch_peak_idx + 1

            twitch_dict = {  # type: ignore
                metric_type: {contraction_type: {} for contraction_type in ("contraction", "relaxation")}
                for metric_type in ("force", "time")
            }

            for iter_percent in twitch_width_percents:
                rising_threshold = peak_force - (iter_percent / 100) * magnitude_of_rise
                falling_threshold = peak_force - (iter_percent / 100) * magnitude_of_fall

                # move to the left from the twitch peak until the rising threshold is reached
                while force_amplitudes_arr[rising_idx] > rising_threshold:
                    rising_idx -= 1
                # move to the right from the twitch peak until the falling threshold is reached
                while force_amplitudes_arr[falling_idx] > falling_threshold:
                    falling_idx += 1

                interpolated_rising_timepoint = interpolate_x_for_y_between_two_points(
                    rising_threshold,
                    timepoints_arr[rising_idx],
                    force_amplitudes_arr[rising_idx],
                    timepoints_arr[rising_idx + 1],
                    force_amplitudes_arr[rising_idx + 1],
                )
                interpolated_falling_timepoint = interpolate_x_for_y_between_two_points(
                    falling_threshold,
                    timepoints_arr[falling_idx],
                    force_amplitudes_arr[falling_idx],
                    timepoints_arr[falling_idx - 1],
                    force_amplitudes_arr[falling_idx - 1],
                )
                width_val = interpolated_falling_timepoint - interpolated_rising_timepoint

                if rounded:
                    width_val = int(round(width_val, 0))
                    interpolated_falling_timepoint = int(round(interpolated_falling_timepoint, 0))
                    interpolated_rising_timepoint = int(round(interpolated_rising_timepoint, 0))
                    rising_threshold = int(round(rising_threshold, 0))
                    falling_threshold = int(round(falling_threshold, 0))

                # fill width-value dictionary
                width_dict[iter_twitch_peak_idx][iter_percent] = width_val / MICRO_TO_BASE_CONVERSION
                twitch_dict["force"]["contraction"][iter_percent] = rising_threshold
                twitch_dict["force"]["relaxation"][iter_percent] = falling_threshold
                twitch_dict["time"]["contraction"][iter_percent] = interpolated_rising_timepoint
                twitch_dict["time"]["relaxation"][iter_percent] = interpolated_falling_timepoint

            # fill coordinate value dictionary
            coordinate_dict[iter_twitch_peak_idx] = twitch_dict

        if as_dict:
            return width_dict, coordinate_dict

        # convert coordinate dictionary to dataframe
        coordinate_df = pd.DataFrame.from_dict(
            {
                (twitch_index, metric_type, contraction_type, twitch_width): coordinate_dict[twitch_index][
                    metric_type
                ][contraction_type][twitch_width]
                for twitch_index in twitch_indices
                for metric_type in ("force", "time")
                for contraction_type in ["contraction", "relaxation"]
                for twitch_width in twitch_width_percents
            },
            orient="index",
        )

        index = pd.MultiIndex.from_tuples(list(coordinate_df.index))

        coordinate_df = pd.Series(coordinate_df.values.squeeze(), index=index)
        coordinate_df = pd.DataFrame(coordinate_df)
        coordinate_df = coordinate_df.unstack(level=0)
        coordinate_df = coordinate_df.T.droplevel(level=0, axis=0)

        # convert width dictionary to dataframe
        width_df = pd.DataFrame.from_dict(width_dict).T

        return width_df, coordinate_df


class TwitchVelocity(BaseMetric):
    """Calculate velocity of each contraction or relaxation twitch."""

    def __init__(
        self,
        rounded: bool = False,
        is_contraction: bool = True,
        twitch_width_percents: Tuple[int, ...] = DEFAULT_TWITCH_WIDTH_PERCENTS,
        **kwargs: Dict[str, Any],
    ):
        super().__init__(rounded=rounded, **kwargs)

        self.twitch_width_percents = twitch_width_percents

        velocity_start = min(self.twitch_width_percents)
        velocity_end = max(self.twitch_width_percents)
        self.velocity_index_start = self.twitch_width_percents.index(velocity_start)
        self.velocity_index_end = self.twitch_width_percents.index(velocity_end)

        self.is_contraction = is_contraction

    def fit(
        self,
        peak_and_valley_indices: Tuple[NDArray[int], NDArray[int]],
        filtered_data: NDArray[(2, Any), int],
        twitch_indices: Dict[int, Dict[UUID, Optional[int]]],
        **kwargs: Dict[str, Any],
    ) -> Series:
        _, coordinates = TwitchWidth.calculate_twitch_widths(
            filtered_data=filtered_data,
            twitch_indices=twitch_indices,
            twitch_width_percents=tuple(self.twitch_width_percents),
            rounded=self.rounded,
        )

        velocities = self.calculate_twitch_velocity(
            coordinate_df=coordinates, is_contraction=self.is_contraction
        )

        return velocities

    def calculate_twitch_velocity(self, coordinate_df: DataFrame, is_contraction: bool) -> Series:
        """Find the velocity for each twitch.

        Args:
            twitch_indices: a dictionary in which the key is an integer representing the time points
                of all the peaks of interest and the value is an inner dictionary with various UUID of
                prior/subsequent peaks and valleys and their index values.

            coordinate_df: DataFrame storing time (X) and force (Y()) values
                for each %-contraction and %-relaxation.  Stored as a MultiIndex dataframe, with
                level(0) = ['time','force']
                level(1) = ['contraction','relaxation']
                level(2) = np.arange(10,95,5)

            is_contraction: a boolean indicating if twitch velocities to be calculating are for the
                twitch contraction or relaxation

        Returns:
            DataFrame floats that are the velocities of each twitch
        """
        coord_type = "contraction" if is_contraction else "relaxation"

        twitch_base = self.twitch_width_percents[self.velocity_index_end]
        twitch_top = self.twitch_width_percents[self.velocity_index_start]

        Y_end = coordinate_df["force", coord_type, twitch_top]
        Y_start = coordinate_df["force", coord_type, twitch_base]

        X_end = coordinate_df["time", coord_type, twitch_top]
        X_start = coordinate_df["time", coord_type, twitch_base]

        # change in force / change in time
        velocity = abs((Y_end - Y_start) / (X_end - X_start))
        velocity *= MICRO_TO_BASE_CONVERSION

        return velocity


class TwitchIrregularity(BaseMetric):
    """Calculate irregularity of each twitch."""

    def __init__(self, rounded: bool = False, **kwargs: Dict[str, Any]):

        super().__init__(rounded=rounded, **kwargs)

    def fit(
        self,
        peak_and_valley_indices: Tuple[NDArray[int], NDArray[int]],
        filtered_data: NDArray[(2, Any), int],
        twitch_indices: Dict[int, Dict[UUID, Optional[int]]],
        **kwargs: Dict[str, Any],
    ) -> Series:
        irregularity = self.calculate_interval_irregularity(
            twitch_indices=twitch_indices, time_series=filtered_data[0]
        )

        return irregularity / MICRO_TO_BASE_CONVERSION

    def add_aggregate_metrics(
        self, aggregate_dict: Dict[UUID, Any], metric_id: UUID, metrics: Union[NDArray[int], NDArray[float]]
    ) -> None:
        statistics_dict = self.create_statistics_df(metric=metrics[1:-1], rounded=self.rounded)
        statistics_dict["n"] += 2
        aggregate_dict[metric_id] = statistics_dict

    @staticmethod
    def calculate_interval_irregularity(
        twitch_indices: Dict[int, Dict[UUID, Optional[int]]], time_series: NDArray[(1, Any), int]
    ) -> Series:
        """Find the interval irregularity for each twitch.

        Args:
            twitch_indices: a dictionary in which the key is an integer representing the time points
                of all the peaks of interest and the value is an inner dictionary with various UUID of
                prior/subsequent peaks and valleys and their index values.

            filtered_data: a 2D array (time vs value) of the data

        Returns:
            Pandas Series of floats that are the interval irregularities of each twitch
        """
        list_of_twitch_indices = list(twitch_indices.keys())
        num_twitches = len(list_of_twitch_indices)

        estimates = {list_of_twitch_indices[0]: None}
        for twitch in range(1, num_twitches - 1):
            last_twitch_index = list_of_twitch_indices[twitch - 1]
            current_twitch_index = list_of_twitch_indices[twitch]
            next_twitch_index = list_of_twitch_indices[twitch + 1]

            last_interval = time_series[current_twitch_index] - time_series[last_twitch_index]
            current_interval = time_series[next_twitch_index] - time_series[current_twitch_index]
            interval = abs(current_interval - last_interval)

            estimates[current_twitch_index] = interval
        estimates[list_of_twitch_indices[-1]] = None

        estimates = pd.Series(estimates)

        return estimates


class TwitchAUC(BaseMetric):
    """Calculate area under each twitch."""

    def __init__(
        self,
        rounded: bool = False,
        twitch_width_percents: Tuple[int, ...] = DEFAULT_TWITCH_WIDTH_PERCENTS,
        **kwargs: Dict[str, Any],
    ):

        super().__init__(rounded=rounded, **kwargs)

        self.twitch_width_percents = twitch_width_percents

    def fit(
        self,
        peak_and_valley_indices: Tuple[NDArray[int], NDArray[int]],
        filtered_data: NDArray[(2, Any), int],
        twitch_indices: Dict[int, Dict[UUID, Optional[int]]],
        **kwargs: Dict[str, Any],
    ) -> Series:
        _, coordinates = TwitchWidth.calculate_twitch_widths(
            filtered_data=filtered_data,
            twitch_indices=twitch_indices,
            twitch_width_percents=tuple(self.twitch_width_percents),
            rounded=self.rounded,
        )

        auc = self.calculate_area_under_curve(
            twitch_indices=twitch_indices, filtered_data=filtered_data, coordinate_df=coordinates
        )
        return auc

    def calculate_area_under_curve(
        self,
        twitch_indices: Dict[int, Dict[UUID, Optional[int]]],
        filtered_data: NDArray[(2, Any), int],
        coordinate_df: DataFrame,
    ) -> Series:
        """Calculate the area under the curve (AUC) for twitches.

        Args:
            twitch_indices: a dictionary in which the key is an integer representing the time points
                of all the peaks of interest and the value is an inner dictionary with various UUIDs
                of prior/subsequent peaks and valleys and their index values.

            filtered_data: a 2D array of the time and value (magnetic, voltage, displacement, force)
                data after it has gone through noise filtering and interpolation

            per_twitch_widths: a list of dictionaries where the first key is the percentage of the
                way down to the nearby valleys, the second key is a UUID representing either the
                value of the width, or the rising or falling coordinates. The final value is either
                an int representing the width value or a tuple of ints for the x/y coordinates

        Returns:
            Pandas Series of floats representing area under the curve for each twitch
        """
        width_percent = 90  # what percent of repolarization to use as the bottom limit for calculating AUC
        estimates_dict: Dict[int, Union[int, float]] = dict()

        rising_x_values = coordinate_df["time"]["contraction"].T.to_dict()
        falling_x_values = coordinate_df["time"]["relaxation"].T.to_dict()

        for iter_twitch_peak_idx in twitch_indices.keys():
            start_timepoint = rising_x_values[iter_twitch_peak_idx][width_percent]
            stop_timepoint = falling_x_values[iter_twitch_peak_idx][width_percent]

            auc_window_indices = get_time_window_indices(filtered_data[0], start_timepoint, stop_timepoint)
            auc_total = np.trapz(filtered_data[1, auc_window_indices], dx=INTERPOLATED_DATA_PERIOD_SECONDS)

            if self.rounded:
                auc_total = int(round(auc_total, 0))

            estimates_dict[iter_twitch_peak_idx] = auc_total

        estimates = pd.Series(estimates_dict)
        return estimates


class TwitchPeriod(BaseMetric):
    """Calculate period of each twitch."""

    def __init__(self, rounded: bool = False, **kwargs: Dict[str, Any]):
        super().__init__(rounded=rounded, **kwargs)

    def fit(
        self,
        peak_and_valley_indices: Tuple[NDArray[int], NDArray[int]],
        filtered_data: NDArray[(2, Any), int],
        twitch_indices: Dict[int, Dict[UUID, Optional[int]]],
        **kwargs: Dict[str, Any],
    ) -> Series:
        periods = self.calculate_twitch_period(
            twitch_indices=twitch_indices,
            peak_indices=peak_and_valley_indices[0],
            filtered_data=filtered_data,
        )

        return periods / MICRO_TO_BASE_CONVERSION

    @staticmethod
    def calculate_twitch_period(
        twitch_indices: Dict[int, Dict[UUID, Optional[int]]],
        peak_indices: NDArray[int],
        filtered_data: NDArray[(2, Any), int],
    ) -> Series:
        """Find the distance between each twitch at its peak.

        Args:
            twitch_indices: a dictionary in which the key is an integer representing the time points
                of all the peaks of interest and the value is an inner dictionary with various UUID
                of prior/subsequent peaks and valleys and their index values.
            all_peak_indices: a 1D array of the indices in the data array that all peaks are at
            filtered_data: a 2D array (time vs value) of the data

        Returns:
            Pandas Series of period for each twitch
        """
        list_of_twitch_indices = list(twitch_indices.keys())
        idx_of_first_twitch = np.where(peak_indices == list_of_twitch_indices[0])[0][0]
        estimates = {twitch_index: None for twitch_index in twitch_indices.keys()}

        time_series = filtered_data[0, :]
        for iter_twitch_idx in range(len(list_of_twitch_indices)):
            current_peak = peak_indices[iter_twitch_idx + idx_of_first_twitch]
            next_peak = peak_indices[iter_twitch_idx + idx_of_first_twitch + 1]

            period = time_series[next_peak] - time_series[current_peak]

            estimates[current_peak] = period

        estimates = pd.Series(estimates)
        return estimates


class TwitchFrequency(BaseMetric):
    """Calculate frequency of each twitch."""

    def __init__(self, rounded: bool = False, **kwargs: Dict[str, Any]):
        super().__init__(rounded=rounded, **kwargs)

    def fit(
        self,
        peak_and_valley_indices: Tuple[NDArray[int], NDArray[int]],
        filtered_data: NDArray[(2, Any), int],
        twitch_indices: Dict[int, Dict[UUID, Optional[int]]],
        **kwargs: Dict[str, Any],
    ) -> Series:
        period_metric = TwitchPeriod(rounded=self.rounded)
        periods = period_metric.fit(
            peak_and_valley_indices=peak_and_valley_indices,
            filtered_data=filtered_data,
            twitch_indices=twitch_indices,
        )
        estimates = 1 / periods.astype(float)
        return estimates


class TwitchPeakTime(BaseMetric):
    """Calculate time from percent twitch width to peak."""

    def __init__(
        self,
        rounded: bool = False,
        is_contraction: bool = True,
        twitch_width_percents: Tuple[int, ...] = DEFAULT_TWITCH_WIDTH_PERCENTS,
        **kwargs: Dict[str, Any],
    ):
        super().__init__(rounded=rounded, **kwargs)

        self.twitch_width_percents = twitch_width_percents
        self.is_contraction = is_contraction

    def fit(
        self,
        peak_and_valley_indices: Tuple[NDArray[int], NDArray[int]],
        filtered_data: NDArray[(2, Any), int],
        twitch_indices: Dict[int, Dict[UUID, Optional[int]]],
        **kwargs: Dict[str, Any],
    ) -> DataFrame:
        _, coordinates = TwitchWidth.calculate_twitch_widths(
            filtered_data=filtered_data,
            twitch_indices=twitch_indices,
            twitch_width_percents=tuple(self.twitch_width_percents),
            rounded=self.rounded,
        )

        time_difference = self.calculate_twitch_time_diff(
            twitch_indices, filtered_data, coordinate_df=coordinates, is_contraction=self.is_contraction
        )

        return time_difference

    def add_aggregate_metrics(
        self,
        aggregate_df: DataFrame,
        metric_id: UUID,
        metrics: DataFrame,
    ) -> None:
        for iter_percent in self.twitch_width_percents:
            estimates = metrics[iter_percent]
            aggregate_estimates = self.create_statistics_df(estimates, rounded=self.rounded)
            try:
                aggregate_df[metric_id, iter_percent] = aggregate_estimates
            except ValueError:
                # Exception occurs when used for C10 to Peak and Peak to R90 metrics due to init df shape
                aggregate_df[metric_id] = aggregate_estimates

    def add_per_twitch_metrics(
        self, main_df: DataFrame, metric_id: UUID, metrics: Union[NDArray[int], NDArray[float]]
    ) -> None:
        try:
            main_df[metric_id] = metrics
        except ValueError:
            # Exception occurs when used for C10 to Peak and Peak to R90 metrics due to init df shape
            main_df[metric_id] = metrics[self.twitch_width_percents[0]]

    def calculate_twitch_time_diff(
        self,
        twitch_indices: Dict[int, Dict[UUID, Optional[int]]],
        filtered_data: NDArray[(2, Any), int],
        coordinate_df: DataFrame,
        is_contraction: bool = True,
    ) -> DataFrame:
        """Calculate time from percent contraction / relaxation to twitch peak.

        Args:
            twitch_indices: a dictionary in which the key is an integer representing the time points
                of all the peaks of interest and the value is an inner dictionary with various UUIDs
                of prior/subsequent peaks and valleys and their index values.

            filtered_data: a 2D array of the time and value (magnetic, voltage, displacement, force)
                data after it has gone through noise filtering

            coordinate_df: DataFrame storing time (X) and force (Y()) values
                for each %-contraction and %-relaxation.  Stored as a MultiIndex dataframe, with
                level(0) = ['time','force']
                level(1) = ['contraction','relaxation']
                level(2) = np.arange(10,95,5)

            is_contraction: bool, specifies whether to compute time-to-peak for contraction or
                relaxation side of twitch
        Returns:
            time_differences: a list of dictionaries where the first key is the percentage of the way
            down to the nearby valleys, the second key is a UUID representing either the relaxation or
            contraction time.  The final value is float indicating time from relaxation/contraction to peak
        """
        # dictionary of time differences for each peak
        coord_label = "contraction" if is_contraction else "relaxation"
        coords = coordinate_df["time"][coord_label]
        coords = coords.T.to_dict()

        def diff_fn(x, y):
            return x - y if is_contraction else y - x

        estimates_dict = {twitch_index: {} for twitch_index in twitch_indices.keys()}  # type: ignore
        for iter_twitch_idx in twitch_indices.keys():

            for iter_percent in self.twitch_width_percents:
                percent = iter_percent
                if is_contraction:
                    percent = 100 - percent

                percent_time = coords[iter_twitch_idx][percent]
                peak_time = filtered_data[0, iter_twitch_idx]

                estimates_dict[iter_twitch_idx][iter_percent] = diff_fn(peak_time, percent_time)
        estimates = pd.DataFrame.from_dict(estimates_dict, orient="index")

        return estimates / MICRO_TO_BASE_CONVERSION


class WellGroupMetric(BaseMetric):
    """Calculate aggregrate group metrics."""

    def __init__(
        self,
        **kwargs: Dict[str, Any],
    ):
        super().__init__(False, **kwargs)

    def add_group_aggregate_metrics(
        self, aggregate_df: DataFrame, metric_column: Tuple, metrics: pd.Series, metric_type
    ) -> None:
        """Get aggregate metrics for entire well group.

        Args:
            aggregate_df (DataFrame): DataFrame storing aggregate metrics for well group
            metric_column (UUID, str): multi-index column in aggregate metrics dataframe
            metrics (Union[NDArray[int], NDArray[float]]): estimates from all wells in single group
            metric_type (str): scalar or by_width
        """
        aggregate_metrics = self.create_statistics_df(metrics.values, rounded=self.rounded)
        if metric_type == "scalar":
            aggregate_df[metric_column[0]] = aggregate_metrics
        else:
            aggregate_df[metric_column[0], metric_column[1]] = aggregate_metrics
