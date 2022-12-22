# -*- coding: utf-8 -*-
"""Metrics for skeletal and cardiac muscles.

If a new metric is requested, you must implement `fit`,
`add_per_twitch_metrics`, and `add_aggregate_metrics`.
"""

# for hashing dataframes
import abc
from functools import lru_cache
from functools import partial
from hashlib import sha256
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from uuid import UUID

from nptyping import Float64
from nptyping import NDArray
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Series
from pandas.util import hash_pandas_object

from .constants import DEFAULT_TWITCH_WIDTH_PERCENTS
from .constants import MICRO_TO_BASE_CONVERSION
from .constants import PRIOR_VALLEY_INDEX_UUID
from .constants import SUBSEQUENT_VALLEY_INDEX_UUID


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
    ) -> Union[NDArray[Float64], List[Dict[int, Dict[UUID, Any]]], DataFrame]:
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
    ) -> DataFrame:

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
            twitch_indices: a dictionary in which the key is an integer representing the time points
                of all the peaks of interest and the value is an inner dictionary with various UUID
                of prior/subsequent peaks and valleys and their index values.

            filtered_data: a 2D array of the time and value (magnetic, voltage, displacement, force)
                data after it has gone through noise filtering

        Returns:
            Pandas Series of float values representing the amplitude of each twitch
        """
        data_series = filtered_data[1, :]

        estimates_dict: Dict[int, float] = dict()
        for iter_twitch_idx, iter_twitch_info in twitch_indices.items():
            peak_amplitude = data_series[iter_twitch_idx]
            prior_amplitude = data_series[iter_twitch_info[PRIOR_VALLEY_INDEX_UUID]]
            subsequent_amplitude = data_series[iter_twitch_info[SUBSEQUENT_VALLEY_INDEX_UUID]]

            amplitude_value = (
                (peak_amplitude - prior_amplitude) + (peak_amplitude - subsequent_amplitude)
            ) / 2

            if rounded:
                amplitude_value = round(amplitude_value, 0)

            estimates_dict[iter_twitch_idx] = amplitude_value

        estimates = pd.Series(estimates_dict) * MICRO_TO_BASE_CONVERSION
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
    ) -> DataFrame:
        amplitudes = super().fit(
            peak_and_valley_indices=peak_and_valley_indices,
            twitch_indices=twitch_indices,
            filtered_data=filtered_data,
        )
        estimates = amplitudes / np.nanmax(amplitudes)
        return estimates


class TwitchWidthCoordinates(BaseMetric):
    """Calculate contraction (or relaxation) percent-width coordinates."""

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
        twitch_indices_hashable = HashableDataFrame(DataFrame.from_dict(twitch_indices).T)
        filtered_data_hashable = tuple(tuple(d) for d in filtered_data)

        _, coordinates = TwitchWidth.calculate_twitch_widths(
            twitch_indices=twitch_indices_hashable,
            filtered_data=filtered_data_hashable,
            rounded=self.rounded,
            twitch_width_percents=tuple(self.twitch_width_percents),
        )

        return coordinates


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
        twitch_indices_hashable = HashableDataFrame(DataFrame.from_dict(twitch_indices).T)
        filtered_data_hashable = tuple(tuple(d) for d in filtered_data)

        widths, _ = self.calculate_twitch_widths(
            twitch_indices=twitch_indices_hashable,
            filtered_data=filtered_data_hashable,
            rounded=self.rounded,
            twitch_width_percents=tuple(self.twitch_width_percents),
        )

        return widths

    def add_aggregate_metrics(self, aggregate_df: DataFrame, metric_id: UUID, metrics: DataFrame) -> None:
        for iter_percent in self.twitch_width_percents:
            estimates = metrics[iter_percent]
            aggregate_estimates = self.create_statistics_df(estimates, rounded=self.rounded)
            aggregate_df[metric_id, iter_percent] = aggregate_estimates

    @staticmethod
    @lru_cache()
    def calculate_twitch_widths(
        twitch_indices: DataFrame,
        filtered_data: Tuple[Tuple[float], Tuple[float]],
        rounded: bool = False,
        twitch_width_percents: Tuple[int, ...] = DEFAULT_TWITCH_WIDTH_PERCENTS,
    ) -> Tuple[DataFrame, DataFrame]:
        """Determine twitch width between 10-90% down to the nearby valleys.

        This is a time-consuming step, so we use a hashable DataFrame for the
        twitch indices, and a tuple of tuples for the filtered data.

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
            and each  column level corresponds to the time (X) / force(Y), contration (rising) / relaxation
            (falling), and percent-twitch width coordinates
        """
        coordinate_dict: Dict[int, Dict[str, Dict[str, Any]]] = dict()

        width_dict: Dict[int, Dict[int, Any]] = {twitch_index: {} for twitch_index in twitch_indices.index}

        time_series = filtered_data[0]
        value_series = filtered_data[1]

        for iter_twitch_peak_idx in twitch_indices.index:
            peak_value = value_series[iter_twitch_peak_idx]

            prior_valley_idx = int(twitch_indices[PRIOR_VALLEY_INDEX_UUID][iter_twitch_peak_idx])
            prior_valley_value = value_series[prior_valley_idx]

            subsequent_valley_idx = int(twitch_indices[SUBSEQUENT_VALLEY_INDEX_UUID][iter_twitch_peak_idx])
            subsequent_valley_value = value_series[subsequent_valley_idx]

            rising_amplitude = peak_value - prior_valley_value
            falling_amplitude = peak_value - subsequent_valley_value

            rising_idx = iter_twitch_peak_idx - 1
            falling_idx = iter_twitch_peak_idx + 1

            twitch_dict = {  # type: ignore
                metric_type: {contraction_type: {} for contraction_type in ("contraction", "relaxation")}
                for metric_type in ("force", "time")
            }

            for iter_percent in sorted(twitch_width_percents):
                rising_threshold = peak_value - (iter_percent / 100) * rising_amplitude
                falling_threshold = peak_value - (iter_percent / 100) * falling_amplitude

                # move to the left from the twitch peak until the threshold is reached
                while abs(value_series[rising_idx] - prior_valley_value) > abs(
                    rising_threshold - prior_valley_value
                ):
                    rising_idx -= 1
                # move to the right from the twitch peak until the falling threshold is reached
                while abs(value_series[falling_idx] - subsequent_valley_value) > abs(
                    falling_threshold - subsequent_valley_value
                ):
                    falling_idx += 1
                interpolated_rising_timepoint = interpolate_x_for_y_between_two_points(
                    rising_threshold,
                    time_series[rising_idx],
                    value_series[rising_idx],
                    time_series[rising_idx + 1],
                    value_series[rising_idx + 1],
                )
                interpolated_falling_timepoint = interpolate_x_for_y_between_two_points(
                    falling_threshold,
                    time_series[falling_idx],
                    value_series[falling_idx],
                    time_series[falling_idx - 1],
                    value_series[falling_idx - 1],
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

        # convert coordinate dictionary to dataframe
        coordinate_df = pd.DataFrame.from_dict(
            {
                (twitch_index, metric_type, contraction_type, twitch_width): coordinate_dict[twitch_index][
                    metric_type
                ][contraction_type][twitch_width]
                for twitch_index in twitch_indices.index
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
        twitch_indices_hashable = HashableDataFrame(DataFrame.from_dict(twitch_indices).T)
        filtered_data_hashable = tuple(tuple(d) for d in filtered_data)

        _, coordinates = TwitchWidth.calculate_twitch_widths(
            twitch_indices=twitch_indices_hashable,
            filtered_data=filtered_data_hashable,
            rounded=self.rounded,
            twitch_width_percents=tuple(self.twitch_width_percents),
        )

        velocities = self.calculate_twitch_velocity(
            twitch_indices=twitch_indices, coordinate_df=coordinates, is_contraction=self.is_contraction
        )

        return velocities

    def calculate_twitch_velocity(
        self,
        twitch_indices: Dict[int, Dict[UUID, Optional[int]]],
        coordinate_df: DataFrame,
        is_contraction: bool,
    ) -> Series:
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
        velocity *= MICRO_TO_BASE_CONVERSION**2

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
        twitch_indices_hashable = HashableDataFrame(DataFrame.from_dict(twitch_indices).T)
        filtered_data_hashable = tuple(tuple(d) for d in filtered_data)

        _, coordinates = TwitchWidth.calculate_twitch_widths(
            twitch_indices=twitch_indices_hashable,
            filtered_data=filtered_data_hashable,
            rounded=self.rounded,
            twitch_width_percents=tuple(self.twitch_width_percents),
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
                data after it has gone through noise filtering

            per_twitch_widths: a list of dictionaries where the first key is the percentage of the
                way down to the nearby valleys, the second key is a UUID representing either the
                value of the width, or the rising or falling coordinates. The final value is either
                an int representing the width value or a tuple of ints for the x/y coordinates

        Returns:
            Pandas Series of floats representing area under the curve for each twitch
        """
        width_percent = 90  # what percent of repolarization to use as the bottom limit for calculating AUC
        estimates_dict: Dict[int, Union[int, float]] = dict()

        value_series = filtered_data[1, :]
        time_series = filtered_data[0, :]

        rising_x_values = coordinate_df["time"]["contraction"].T.to_dict()
        rising_y_values = coordinate_df["force"]["contraction"].T.to_dict()

        falling_x_values = coordinate_df["time"]["relaxation"].T.to_dict()
        falling_y_values = coordinate_df["force"]["relaxation"].T.to_dict()

        for iter_twitch_peak_idx, iter_twitch_indices_info in twitch_indices.items():
            # iter_twitch_peak_timepoint = time_series[iter_twitch_peak_idx]
            prior_valley_value = value_series[iter_twitch_indices_info[PRIOR_VALLEY_INDEX_UUID]]
            subsequent_valley_value = value_series[iter_twitch_indices_info[SUBSEQUENT_VALLEY_INDEX_UUID]]

            rising_x = rising_x_values[iter_twitch_peak_idx][width_percent]
            rising_y = rising_y_values[iter_twitch_peak_idx][width_percent]
            rising_coords = (rising_x, rising_y)

            falling_x = falling_x_values[iter_twitch_peak_idx][width_percent]
            falling_y = falling_y_values[iter_twitch_peak_idx][width_percent]
            falling_coords = (falling_x, falling_y)

            auc_total: Union[float, int] = 0

            # calculate area of rising side
            rising_idx = iter_twitch_peak_idx
            # move to the left from the twitch peak until the threshold is reached
            while abs(value_series[rising_idx - 1] - prior_valley_value) > abs(rising_y - prior_valley_value):
                left_x = time_series[rising_idx - 1]
                right_x = time_series[rising_idx]
                left_y = value_series[rising_idx - 1]
                right_y = value_series[rising_idx]

                auc_total += self.calculate_trapezoid_area(
                    left_x,
                    right_x,
                    left_y,
                    right_y,
                    rising_coords,
                    falling_coords,
                )
                rising_idx -= 1
            # final trapezoid at the boundary of the interpolated twitch width point
            left_x = rising_x
            right_x = time_series[rising_idx]
            left_y = rising_y
            right_y = value_series[rising_idx]

            auc_total += self.calculate_trapezoid_area(
                left_x, right_x, left_y, right_y, rising_coords, falling_coords
            )

            # calculate area of falling side
            falling_idx = iter_twitch_peak_idx
            # move to the left from the twitch peak until the threshold is reached
            while abs(value_series[falling_idx + 1] - subsequent_valley_value) > abs(
                falling_y - subsequent_valley_value
            ):
                left_x = time_series[falling_idx]
                right_x = time_series[falling_idx + 1]
                left_y = value_series[falling_idx]
                right_y = value_series[falling_idx + 1]

                auc_total += self.calculate_trapezoid_area(
                    left_x, right_x, left_y, right_y, rising_coords, falling_coords
                )
                falling_idx += 1

            # final trapezoid at the boundary of the interpolated twitch width point
            left_x = time_series[falling_idx]
            right_x = falling_x
            left_y = value_series[rising_idx]
            right_y = falling_y

            auc_total += self.calculate_trapezoid_area(
                left_x, right_x, left_y, right_y, rising_coords, falling_coords
            )
            if self.rounded:
                auc_total = int(round(auc_total, 0))

            estimates_dict[iter_twitch_peak_idx] = auc_total

        estimates = pd.Series(estimates_dict)
        return estimates

    @staticmethod
    def calculate_trapezoid_area(
        left_x: int,
        right_x: int,
        left_y: int,
        right_y: int,
        rising_coords: Tuple[Union[float, int], Union[float, int]],
        falling_coords: Tuple[Union[float, int], Union[float, int]],
    ) -> Union[int, float]:
        """Calculate the area under the trapezoid.

        Returns: area of the trapezoid
        """
        rising_x, rising_y = rising_coords
        falling_x, falling_y = falling_coords

        interp_y_for_lower_bound = partial(
            interpolate_y_for_x_between_two_points, x_1=rising_x, y_1=rising_y, x_2=falling_x, y_2=falling_y
        )

        trapezoid_left_side = abs(left_y - interp_y_for_lower_bound(left_x))
        trapezoid_right_side = abs(right_y - interp_y_for_lower_bound(right_x))

        trapezoid_h = right_x - left_x
        auc_total = (trapezoid_left_side + trapezoid_right_side) / 2 * trapezoid_h

        return auc_total


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
    ) -> DataFrame:
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
        twitch_indices_hashable = HashableDataFrame(DataFrame.from_dict(twitch_indices).T)
        filtered_data_hashable = tuple(tuple(d) for d in filtered_data)

        _, coordinates = TwitchWidth.calculate_twitch_widths(
            twitch_indices=twitch_indices_hashable,
            filtered_data=filtered_data_hashable,
            rounded=self.rounded,
            twitch_width_percents=tuple(self.twitch_width_percents),
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


class TwitchPeakToBaseline(BaseMetric):
    """Calculate full contraction or full relaxation time."""

    def __init__(
        self,
        rounded: bool = False,
        is_contraction: bool = True,
        **kwargs: Dict[str, Any],
    ):
        super().__init__(rounded=rounded, **kwargs)
        self.is_contraction = is_contraction

    def fit(
        self,
        peak_and_valley_indices: Tuple[NDArray[int], NDArray[int]],
        filtered_data: NDArray[(2, Any), int],
        twitch_indices: Dict[int, Dict[UUID, Optional[int]]],
        **kwargs: Dict[str, Any],
    ) -> Series:
        time_series = filtered_data[0, :]
        valley_key = PRIOR_VALLEY_INDEX_UUID if self.is_contraction else SUBSEQUENT_VALLEY_INDEX_UUID

        def get_diff(x, y):
            diff = x - y if self.is_contraction else y - x
            return diff

        peak_times = [time_series[k] for k in twitch_indices.keys()]
        valley_times = [time_series[twitch_indices[k][valley_key]] for k in twitch_indices.keys()]

        estimates_list = [
            get_diff(peak_time, valley_time) for peak_time, valley_time in zip(peak_times, valley_times)
        ]
        estimates = pd.Series(estimates_list, index=twitch_indices.keys()) / MICRO_TO_BASE_CONVERSION
        return estimates


def interpolate_x_for_y_between_two_points(
    desired_y: Union[int, float],
    x_1: Union[int, float],
    y_1: Union[int, float],
    x_2: Union[int, float],
    y_2: Union[int, float],
) -> Union[int, float]:
    """Find a value of x between two points that matches the desired y value.

    Uses linear interpolation, based on point-slope formula.
    """
    denominator = x_2 - x_1

    if denominator == 0:
        raise ZeroDivisionError("Denominator cannot be 0.")

    slope = (y_2 - y_1) / denominator
    return (desired_y - y_1) / slope + x_1


def interpolate_y_for_x_between_two_points(
    desired_x: Union[int, float],
    x_1: Union[int, float],
    y_1: Union[int, float],
    x_2: Union[int, float],
    y_2: Union[int, float],
) -> Union[int, float]:
    """Find a value of y between two points that matches the desired x value.

    Uses linear interpolation, based on point-slope formula.
    """
    denominator = x_2 - x_1

    if denominator == 0:
        raise ZeroDivisionError("Denominator cannot be 0.")

    slope = (y_2 - y_1) / denominator
    return slope * (desired_x - x_1) + y_1


class HashableDataFrame(DataFrame):
    def __init__(self, obj):
        super().__init__(obj)

    def __hash__(self):
        hash_value = sha256(hash_pandas_object(self, index=True).values)
        hash_value = hash(hash_value.hexdigest())
        return hash_value

    def __eq__(self, other):
        return self.equals(other)
