# -*- coding: utf-8 -*-
"""Metrics for skeletal and cardiac muscles.

If a new metric is requested, you must implement `fit`,
`add_per_twitch_metrics`, and `add_aggregate_metrics`.
"""

from functools import partial
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

from .constants import MICRO_TO_BASE_CONVERSION
from .constants import PRIOR_VALLEY_INDEX_UUID
from .constants import SUBSEQUENT_VALLEY_INDEX_UUID
from .constants import TIME_VALUE_UUID
from .constants import WIDTH_FALLING_COORDS_UUID
from .constants import WIDTH_RISING_COORDS_UUID
from .constants import WIDTH_VALUE_UUID

TWITCH_WIDTH_PERCENTS = np.arange(10, 95, 5)
TWITCH_WIDTH_INDEX_OF_CONTRACTION_VELOCITY_START = np.where(TWITCH_WIDTH_PERCENTS == 10)[0]
TWITCH_WIDTH_INDEX_OF_CONTRACTION_VELOCITY_END = np.where(TWITCH_WIDTH_PERCENTS == 90)[0]


class BaseMetric:
    """Any new metric needs to implement three methods.

    1) estimate the per-twitch values
    2) add the per-twitch values to the per-twitch dictionary
    3) add the aggregate statistics to the aggregate dictionary.

    Most metrics will estimate a single value per twitch, but others are
    nested (twitch widths, time-to/from peak, etc.)
    """

    def __init__(  # pylint:disable=unused-argument # Kristian (11/1/21) need kwargs for sub-class parameters
        self, rounded: bool = False, **kwargs: Dict[str, Any]
    ):

        self.rounded = rounded

    def fit(
        self,
        peak_and_valley_indices: Tuple[NDArray[int], NDArray[int]],
        filtered_data: NDArray[(2, Any), int],
        twitch_indices: NDArray[int],
        **kwargs: Dict[str, Any],
    ) -> Union[NDArray[Float64], List[Dict[int, Dict[UUID, Any]]]]:

        pass

    @staticmethod
    def add_per_twitch_metrics(
        main_twitch_dict: Dict[Any, Any], metric_id: UUID, metrics: Union[NDArray[int], NDArray[float]]
    ) -> None:
        """Add estimated per-twitch metrics to per-twitch dictionary.

        Args:
            main_twitch_dict (Dict[Any, Any]): dictionary storing per-twitch metrics
            metric_id (UUID): UUID of metric to add
            metrics (Union[NDArray[int], NDArray[float]]): estimated per-twitch metrics
        """
        for i, twitch_dict in enumerate(main_twitch_dict.values()):
            twitch_dict[metric_id] = metrics[i]

    def add_aggregate_metrics(
        self,
        aggregate_dict: Dict[
            UUID,
            Any,
        ],
        metric_id: UUID,
        metrics: Union[NDArray[int], NDArray[float]],
    ) -> None:
        """Add estimated metrics to aggregate dictionary.

        Args:
            aggregate_dict (Dict[UUID,Any,]): dictionary storing aggregate metrics
            metric_id (UUID): UUID of metric to add
            metrics (Union[NDArray[int], NDArray[float]]): estimated per-twitch metrics
        """
        aggregate_dict[metric_id] = self.create_statistics_dict(metrics, rounded=self.rounded)

    @classmethod
    def create_statistics_dict(
        cls, metric: NDArray[int], rounded: bool = False
    ) -> Dict[str, Union[Float64, int]]:
        """Calculate various statistics for a specific metric.

        Args:
        metric: a 1D array of integer values of a specific metric results

        Returns:
        a dictionary of the average statistics of that metric in which the metrics are the key and
        average statistics are the value
        """
        d: Dict[str, Union[Float64, int]] = dict()
        d["n"] = len(metric)
        if len(metric) > 0:
            d["mean"] = np.mean(metric)
            d["std"] = np.std(metric)
            d["min"] = np.min(metric)
            d["max"] = np.max(metric)
            d["cov"] = d["std"] / d["mean"]
            d["sem"] = d["std"] / d["n"] ** 0.5

            if rounded:
                for iter_key, iter_value in d.items():
                    d[iter_key] = int(round(iter_value))

        else:
            d["mean"] = None
            d["std"] = None
            d["min"] = None
            d["max"] = None
            d["cov"] = None
            d["sem"] = None

        return d


class TwitchAmplitude(BaseMetric):
    """Calculate the amplitude for each twitch."""

    def __init__(self, rounded: bool = False, **kwargs: Dict[str, Any]):
        super().__init__(rounded=rounded, **kwargs)

    def fit(
        self,
        peak_and_valley_indices: Tuple[NDArray[int], NDArray[int]],
        filtered_data: NDArray[(2, Any), int],
        twitch_indices: NDArray[int],
        **kwargs: Dict[str, Any],
    ) -> NDArray[Float64]:

        amplitudes: NDArray[Float64] = self.calculate_amplitudes(
            twitch_indices=twitch_indices, filtered_data=filtered_data, rounded=self.rounded
        )

        return amplitudes

    @staticmethod
    def calculate_amplitudes(
        twitch_indices: Dict[int, Dict[UUID, Optional[int]]],
        filtered_data: NDArray[(2, Any), int],
        rounded: bool = False,
    ) -> NDArray[float]:
        """Get the amplitudes for all twitches.

        Given amplitude of current peak, and amplitude of prior/subsequent valleys, twitch amplitude
        is calculated as the mean distance from peak to both valleys.

        Args:
            twitch_indices: a dictionary in which the key is an integer representing the time points
                of all the peaks of interest and the value is an inner dictionary with various UUID
                of prior/subsequent peaks and valleys and their index values.

            filtered_data: a 2D array of the time and value (magnetic, voltage, displacement, force...)
                data after it has gone through noise filtering

        Returns:
            a 1D array of integers representing the amplitude of each twitch
        """
        num_twitches = len(twitch_indices)
        amplitudes: NDArray[Float64] = np.zeros(
            (num_twitches),
        )
        data_series = filtered_data[1, :]

        for i, (iter_twitch_idx, iter_twitch_info) in enumerate(twitch_indices.items()):
            peak_amplitude = data_series[iter_twitch_idx]
            prior_amplitude = data_series[iter_twitch_info[PRIOR_VALLEY_INDEX_UUID]]
            subsequent_amplitude = data_series[iter_twitch_info[SUBSEQUENT_VALLEY_INDEX_UUID]]

            amplitude_value = (
                (peak_amplitude - prior_amplitude) + (peak_amplitude - subsequent_amplitude)
            ) / 2

            if rounded:
                amplitude_value = round(amplitude_value, 0)

            amplitudes[i] = amplitude_value

        amplitudes = amplitudes * MICRO_TO_BASE_CONVERSION
        return amplitudes


class TwitchFractionAmplitude(BaseMetric):
    """Calculate the fraction of max amplitude for each twitch."""

    def __init__(self, rounded: bool = False, **kwargs: Dict[str, Any]):
        super().__init__(rounded=rounded, **kwargs)

    def fit(
        self,
        peak_and_valley_indices: Tuple[NDArray[int], NDArray[int]],
        filtered_data: NDArray[(2, Any), int],
        twitch_indices: NDArray[int],
        **kwargs: Dict[str, Any],
    ) -> NDArray[Float64]:

        amplitude_metric = TwitchAmplitude(rounded=False)
        amplitudes: NDArray[Float64] = amplitude_metric.fit(
            peak_and_valley_indices=peak_and_valley_indices,
            twitch_indices=twitch_indices,
            filtered_data=filtered_data,
        )

        fraction_amplitude: NDArray[Float64] = amplitudes / np.nanmax(amplitudes)

        return fraction_amplitude


class TwitchWidth(BaseMetric):
    """Calculate the width of each twitch at fraction of twitch."""

    def __init__(
        self,
        rounded: bool = False,
        twitch_width_percents: Optional[List[int]] = None,
        **kwargs: Dict[str, Any],
    ):
        super().__init__(rounded=rounded, **kwargs)

        if twitch_width_percents is None:
            twitch_width_percents = TWITCH_WIDTH_PERCENTS

        self.twitch_width_percents = twitch_width_percents

    def fit(
        self,
        peak_and_valley_indices: Tuple[NDArray[int], NDArray[int]],
        filtered_data: NDArray[(2, Any), int],
        twitch_indices: NDArray[int],
        **kwargs: Dict[str, Any],
    ) -> List[Dict[int, Dict[UUID, Any]]]:
        widths = self.calculate_twitch_widths(
            twitch_indices=twitch_indices,
            filtered_data=filtered_data,
            rounded=self.rounded,
            twitch_width_percents=self.twitch_width_percents,
        )

        return widths

    def add_aggregate_metrics(
        self,
        aggregate_dict: Dict[UUID, Any],
        metric_id: UUID,
        metrics: Union[NDArray[int], NDArray[float]],
    ) -> None:

        width_stats_dict: Dict[int, Dict[str, Union[float, int]]] = dict()

        for iter_percent in self.twitch_width_percents:
            iter_list_of_width_values: List[Union[float, int]] = []
            for iter_twitch in metrics:
                iter_width_value = iter_twitch[iter_percent][WIDTH_VALUE_UUID]
                if not isinstance(iter_width_value, (float, int)):  # making mypy happy
                    raise NotImplementedError(
                        f"The width value under key {WIDTH_VALUE_UUID} must be a float or an int. It was: {iter_width_value}"
                    )
                iter_list_of_width_values.append(iter_width_value)
            iter_stats_dict = self.create_statistics_dict(
                metric=iter_list_of_width_values, rounded=self.rounded
            )
            width_stats_dict[iter_percent] = iter_stats_dict

        aggregate_dict[metric_id] = width_stats_dict

    @staticmethod
    def calculate_twitch_widths(
        twitch_indices: Dict[int, Dict[UUID, Optional[int]]],
        filtered_data: NDArray[(2, Any), int],
        rounded: bool = True,
        twitch_width_percents: List[int] = np.arange(10, 95, 5),
    ) -> List[Dict[int, Dict[UUID, Any]]]:
        """Determine twitch width between 10-90% down to the nearby valleys.

        Args:
            twitch_indices: a dictionary in which the key is an integer representing the time points
            of all the peaks of interest and the value is an inner dictionary with various UUIDs of
            prior/subsequent peaks and valleys and their index values.

            filtered_data: a 2D array of the time and value (magnetic, voltage, displacement, force...)
            data after it has gone through noise filtering

        Returns:
            a list of dictionaries where the first key is the percentage of the way down to the nearby
            valleys, the second key is a UUID representing either the value of the width, or the rising
            or falling coordinates. The final value is either an int (for value) or a tuple of ints for
            the x/y coordinates
        """
        widths: List[Dict[int, Dict[UUID, Any]]] = list()

        value_series = filtered_data[1, :]
        time_series = filtered_data[0, :]
        for iter_twitch_peak_idx, iter_twitch_indices_info in twitch_indices.items():

            iter_width_dict: Dict[int, Dict[UUID, Any]] = dict()

            peak_value = value_series[iter_twitch_peak_idx]
            prior_valley_value = value_series[iter_twitch_indices_info[PRIOR_VALLEY_INDEX_UUID]]
            subsequent_valley_value = value_series[iter_twitch_indices_info[SUBSEQUENT_VALLEY_INDEX_UUID]]

            rising_amplitude = peak_value - prior_valley_value
            falling_amplitude = peak_value - subsequent_valley_value

            rising_idx = iter_twitch_peak_idx - 1
            falling_idx = iter_twitch_peak_idx + 1
            for iter_percent in twitch_width_percents:
                iter_percent_dict: Dict[
                    UUID, Union[Tuple[Union[float, int], Union[float, int]], Union[float, int]]
                ] = dict()
                rising_threshold = peak_value - iter_percent / 100 * rising_amplitude
                falling_threshold = peak_value - iter_percent / 100 * falling_amplitude

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

                iter_percent_dict[WIDTH_VALUE_UUID] = width_val / MICRO_TO_BASE_CONVERSION
                iter_percent_dict[WIDTH_RISING_COORDS_UUID] = (
                    interpolated_rising_timepoint,
                    rising_threshold,
                )
                iter_percent_dict[WIDTH_FALLING_COORDS_UUID] = (
                    interpolated_falling_timepoint,
                    falling_threshold,
                )
                iter_width_dict[iter_percent] = iter_percent_dict
            widths.append(iter_width_dict)

        return widths


class TwitchVelocity(BaseMetric):
    """Calculate velocity of each contraction or relaxation twitch."""

    def __init__(
        self,
        rounded: bool = False,
        is_contraction: bool = True,
        twitch_width_percents: Optional[List[int]] = None,
        **kwargs: Dict[str, Any],
    ):
        super().__init__(rounded=rounded, **kwargs)

        if twitch_width_percents is None:
            twitch_width_percents = TWITCH_WIDTH_PERCENTS

        velocity_start = min(twitch_width_percents)
        velocity_end = max(twitch_width_percents)

        self.twitch_width_percents = twitch_width_percents
        self.velocity_index_start = list(twitch_width_percents).index(velocity_start)
        self.velocity_index_end = list(twitch_width_percents).index(velocity_end)
        self.is_contraction = is_contraction

    def fit(
        self,
        peak_and_valley_indices: Tuple[NDArray[int], NDArray[int]],
        filtered_data: NDArray[(2, Any), int],
        twitch_indices: NDArray[int],
        **kwargs: Dict[str, Any],
    ) -> NDArray[Float64]:
        width_metric = TwitchWidth(rounded=self.rounded, twitch_width_percents=self.twitch_width_percents)
        widths = width_metric.fit(
            peak_and_valley_indices=peak_and_valley_indices,
            twitch_indices=twitch_indices,
            filtered_data=filtered_data,
        )

        velocities: NDArray[Float64] = self.calculate_twitch_velocity(
            twitch_indices=twitch_indices, per_twitch_widths=widths, is_contraction=self.is_contraction
        )

        return velocities

    def calculate_twitch_velocity(
        self,
        twitch_indices: NDArray[int],
        per_twitch_widths: List[
            Dict[
                int,
                Dict[
                    UUID,
                    Union[Tuple[Union[float, int], Union[float, int]], Union[float, int]],
                ],
            ],
        ],
        is_contraction: bool,
    ) -> NDArray[float]:
        """Find the velocity for each twitch.

        Args:
            twitch_indices: a dictionary in which the key is an integer representing the time points
                of all the peaks of interest and the value is an inner dictionary with various UUID of
                prior/subsequent peaks and valleys and their index values.

            per_twitch_widths: a list of dictionaries where the first key is the percentage of the
                way down to the nearby valleys, the second key is a UUID representing either the value
                of the width, or the rising or falling coordinates. The final value is either an
                int (for value) or a tuple of ints for the x/y coordinates

            is_contraction: a boolean indicating if twitch velocities to be calculating are for the
                twitch contraction or relaxation

        Returns:
            an array of floats that are the velocities of each twitch
        """
        list_of_twitch_indices = list(twitch_indices.keys())
        num_twitches = len(list_of_twitch_indices)
        coord_type = WIDTH_RISING_COORDS_UUID
        if not is_contraction:
            coord_type = WIDTH_FALLING_COORDS_UUID

        twitch_base = self.twitch_width_percents[self.velocity_index_end]
        twitch_top = self.twitch_width_percents[self.velocity_index_start]

        iter_list_of_velocities: List[Union[float, int]] = []
        for twitch in range(num_twitches):
            iter_coord_base = per_twitch_widths[twitch][twitch_base][coord_type]
            iter_coord_top = per_twitch_widths[twitch][twitch_top][coord_type]

            if not isinstance(iter_coord_base, tuple):  # making mypy happy
                raise NotImplementedError(
                    f"The width value under twitch {twitch} must be a Tuple. It was: {iter_coord_base}"
                )
            if not isinstance(iter_coord_top, tuple):  # making mypy happy
                raise NotImplementedError(
                    f"The width value under twitch {twitch} must be a Tuple. It was: {iter_coord_top}"
                )

            velocity = abs(
                (iter_coord_top[1] - iter_coord_base[1]) / (iter_coord_top[0] - iter_coord_base[0])
            )
            iter_list_of_velocities.append(velocity)

        values = np.asarray(iter_list_of_velocities, dtype=float)
        return values * MICRO_TO_BASE_CONVERSION ** 2


class TwitchIrregularity(BaseMetric):
    """Calculate irregularity of each twitch."""

    def __init__(self, rounded: bool = False, **kwargs: Dict[str, Any]):

        super().__init__(rounded=rounded, **kwargs)

    def fit(
        self,
        peak_and_valley_indices: Tuple[NDArray[int], NDArray[int]],
        filtered_data: NDArray[(2, Any), int],
        twitch_indices: NDArray[int],
        **kwargs: Dict[str, Any],
    ) -> NDArray[Float64]:
        irregularity: NDArray[Float64] = self.calculate_interval_irregularity(
            twitch_indices=twitch_indices, time_series=filtered_data[0]
        )

        return irregularity / MICRO_TO_BASE_CONVERSION

    def add_aggregate_metrics(
        self,
        aggregate_dict: Dict[
            UUID,
            Any,
        ],
        metric_id: UUID,
        metrics: Union[NDArray[int], NDArray[float]],
    ) -> None:
        statistics_dict = self.create_statistics_dict(metric=metrics[1:-1], rounded=self.rounded)
        statistics_dict["n"] += 2

        aggregate_dict[metric_id] = statistics_dict

    @staticmethod
    def calculate_interval_irregularity(
        twitch_indices: NDArray[int],
        time_series: NDArray[(1, Any), int],
    ) -> NDArray[float]:
        """Find the interval irregularity for each twitch.

        Args:
            twitch_indices: a dictionary in which the key is an integer representing the time points
                of all the peaks of interest and the value is an inner dictionary with various UUID of
                prior/subsequent peaks and valleys and their index values.

            filtered_data: a 2D array (time vs value) of the data

        Returns:
            an array of floats that are the interval irregularities of each twitch
        """
        list_of_twitch_indices = list(twitch_indices.keys())
        num_twitches = len(list_of_twitch_indices)

        iter_list_of_intervals: List[Union[float, int, None]] = []
        iter_list_of_intervals.append(None)

        for twitch in range(1, num_twitches - 1):
            last_twitch_index = list_of_twitch_indices[twitch - 1]
            current_twitch_index = list_of_twitch_indices[twitch]
            next_twitch_index = list_of_twitch_indices[twitch + 1]

            last_interval = time_series[current_twitch_index] - time_series[last_twitch_index]
            current_interval = time_series[next_twitch_index] - time_series[current_twitch_index]
            interval = abs(current_interval - last_interval)

            iter_list_of_intervals.append(interval)

        iter_list_of_intervals.append(None)
        return np.asarray(iter_list_of_intervals, dtype=float)


class TwitchAUC(BaseMetric):
    """Calculate area under each twitch."""

    def __init__(
        self, rounded: bool = False, twitch_width_percents: List[int] = None, **kwargs: Dict[str, Any]
    ):

        super().__init__(rounded=rounded, **kwargs)

        if twitch_width_percents is None:
            twitch_width_percents = TWITCH_WIDTH_PERCENTS

        self.twitch_width_percents = twitch_width_percents

    def fit(
        self,
        peak_and_valley_indices: Tuple[NDArray[int], NDArray[int]],
        filtered_data: NDArray[(2, Any), int],
        twitch_indices: NDArray[int],
        **kwargs: Dict[str, Any],
    ) -> NDArray[Float64]:
        width_metric = TwitchWidth(rounded=self.rounded, twitch_width_percents=self.twitch_width_percents)
        widths = width_metric.fit(
            peak_and_valley_indices=peak_and_valley_indices,
            twitch_indices=twitch_indices,
            filtered_data=filtered_data,
        )

        auc: NDArray[Float64] = self.calculate_area_under_curve(
            twitch_indices=twitch_indices, filtered_data=filtered_data, per_twitch_widths=widths
        )

        return auc

    def calculate_area_under_curve(  # pylint:disable=too-many-locals # Eli (9/1/20): may be able to refactor before pull request
        self,
        twitch_indices: Dict[int, Dict[UUID, Optional[int]]],
        filtered_data: NDArray[(2, Any), int],
        per_twitch_widths: List[
            Dict[
                int,
                Dict[
                    UUID,
                    Union[Tuple[Union[float, int], Union[float, int]], Union[float, int]],
                ],
            ],
        ],
    ) -> NDArray[float]:
        """Calculate the area under the curve (AUC) for twitches.

        Args:
            twitch_indices: a dictionary in which the key is an integer representing the time points
                of all the peaks of interest and the value is an inner dictionary with various UUIDs
                of prior/subsequent peaks and valleys and their index values.

            filtered_data: a 2D array of the time and value (magnetic, voltage, displacement, force...)
                data after it has gone through noise filtering

            per_twitch_widths: a list of dictionaries where the first key is the percentage of the
                way down to the nearby valleys, the second key is a UUID representing either the
                value of the width, or the rising or falling coordinates. The final value is either
                an int representing the width value or a tuple of ints for the x/y coordinates

        Returns:
            a 1D array of integers which represent the area under the curve for each twitch
        """
        width_percent = 90  # what percent of repolarization to use as the bottom limit for calculating AUC
        auc_per_twitch: List[float] = list()
        value_series = filtered_data[1, :]
        time_series = filtered_data[0, :]

        for iter_twitch_idx, (iter_twitch_peak_idx, iter_twitch_indices_info) in enumerate(
            twitch_indices.items()
        ):
            # iter_twitch_peak_timepoint = time_series[iter_twitch_peak_idx]
            width_info = per_twitch_widths[iter_twitch_idx]
            prior_valley_value = value_series[iter_twitch_indices_info[PRIOR_VALLEY_INDEX_UUID]]
            subsequent_valley_value = value_series[iter_twitch_indices_info[SUBSEQUENT_VALLEY_INDEX_UUID]]
            rising_coords = width_info[width_percent][WIDTH_RISING_COORDS_UUID]
            falling_coords = width_info[width_percent][WIDTH_FALLING_COORDS_UUID]

            if not isinstance(rising_coords, tuple):  # Eli (9/1/20): this appears needed to make mypy happy
                raise NotImplementedError(
                    f"Rising coordinates under the key {WIDTH_RISING_COORDS_UUID} must be a tuple."
                )

            if not isinstance(falling_coords, tuple):  # Eli (9/1/20): this appears needed to make mypy happy
                raise NotImplementedError(
                    f"Falling coordinates under the key {WIDTH_FALLING_COORDS_UUID} must be a tuple."
                )

            rising_x, rising_y = rising_coords
            falling_x, falling_y = falling_coords

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
                left_x,
                right_x,
                left_y,
                right_y,
                rising_coords,
                falling_coords,
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
                    left_x,
                    right_x,
                    left_y,
                    right_y,
                    rising_coords,
                    falling_coords,
                )
                falling_idx += 1

            # final trapezoid at the boundary of the interpolated twitch width point
            left_x = time_series[falling_idx]
            right_x = falling_x
            left_y = value_series[rising_idx]
            right_y = falling_y

            auc_total += self.calculate_trapezoid_area(
                left_x,
                right_x,
                left_y,
                right_y,
                rising_coords,
                falling_coords,
            )
            if self.rounded:
                auc_total = int(round(auc_total, 0))
            auc_per_twitch.append(auc_total)

        return np.asarray(auc_per_twitch, dtype=float)

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
            interpolate_y_for_x_between_two_points,
            x_1=rising_x,
            y_1=rising_y,
            x_2=falling_x,
            y_2=falling_y,
        )

        trapezoid_h = right_x - left_x
        trapezoid_left_side = abs(left_y - interp_y_for_lower_bound(left_x))
        trapezoid_right_side = abs(right_y - interp_y_for_lower_bound(right_x))
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
        twitch_indices: NDArray[int],
        **kwargs: Dict[str, Any],
    ) -> NDArray[Float64]:
        periods: NDArray[Float64] = self.calculate_twitch_period(
            twitch_indices=twitch_indices,
            peak_indices=peak_and_valley_indices[0],
            filtered_data=filtered_data,
        )

        return periods / MICRO_TO_BASE_CONVERSION

    @staticmethod
    def calculate_twitch_period(
        twitch_indices: NDArray[int],
        peak_indices: NDArray[int],
        filtered_data: NDArray[(2, Any), int],
    ) -> NDArray[int]:
        """Find the distance between each twitch at its peak.

        Args:
            twitch_indices:a dictionary in which the key is an integer representing the time points
                of all the peaks of interest and the value is an inner dictionary with various UUID
                of prior/subsequent peaks and valleys and their index values.

            all_peak_indices: a 1D array of the indices in teh data array that all peaks are at

            filtered_data: a 2D array (time vs value) of the data

        Returns:
            an array of integers that are the period of each twitch
        """
        list_of_twitch_indices = list(twitch_indices.keys())
        idx_of_first_twitch = np.where(peak_indices == list_of_twitch_indices[0])[0][0]
        period: List[int] = []
        time_series = filtered_data[0, :]
        for iter_twitch_idx in range(len(list_of_twitch_indices)):

            period.append(
                time_series[peak_indices[iter_twitch_idx + idx_of_first_twitch + 1]]
                - time_series[peak_indices[iter_twitch_idx + idx_of_first_twitch]]
            )

        return np.asarray(period, dtype=np.int32)


class TwitchFrequency(BaseMetric):
    """Calculate frequency of each twitch."""

    def __init__(self, rounded: bool = False, **kwargs: Dict[str, Any]):
        super().__init__(rounded=rounded, **kwargs)

    def fit(
        self,
        peak_and_valley_indices: Tuple[NDArray[int], NDArray[int]],
        filtered_data: NDArray[(2, Any), int],
        twitch_indices: NDArray[int],
        **kwargs: Dict[str, Any],
    ) -> NDArray[Float64]:
        period_metric = TwitchPeriod(rounded=self.rounded)
        periods: NDArray[Float64] = period_metric.fit(
            peak_and_valley_indices=peak_and_valley_indices,
            filtered_data=filtered_data,
            twitch_indices=twitch_indices,
        )

        frequencies: NDArray[Float64] = 1 / periods.astype(float)

        return frequencies


class TwitchPeakTime(BaseMetric):
    """Calculate time from percent twitch width to peak."""

    def __init__(
        self,
        rounded: bool = False,
        is_contraction: bool = True,
        twitch_width_percents: List[int] = None,
        **kwargs: Dict[str, Any],
    ):
        super().__init__(rounded=rounded, **kwargs)

        if twitch_width_percents is None:
            twitch_width_percents = TWITCH_WIDTH_PERCENTS

        self.is_contraction = is_contraction
        self.twitch_width_percents = twitch_width_percents

    def fit(
        self,
        peak_and_valley_indices: Tuple[NDArray[int], NDArray[int]],
        filtered_data: NDArray[(2, Any), int],
        twitch_indices: NDArray[int],
        **kwargs: Dict[str, Any],
    ) -> List[Dict[int, Dict[UUID, Any]]]:
        width_metric = TwitchWidth(rounded=self.rounded, twitch_width_percents=self.twitch_width_percents)
        widths = width_metric.fit(
            peak_and_valley_indices=peak_and_valley_indices,
            twitch_indices=twitch_indices,
            filtered_data=filtered_data,
        )

        time_difference = self.calculate_twitch_time_diff(
            twitch_indices=twitch_indices,
            filtered_data=filtered_data,
            per_twitch_widths=widths,
            is_contraction=self.is_contraction,
        )

        return time_difference

    def add_aggregate_metrics(
        self,
        aggregate_dict: Dict[
            UUID,
            Any,
        ],
        metric_id: UUID,
        metrics: Union[NDArray[int], NDArray[float]],
    ) -> None:

        width_stats_dict: Dict[int, Dict[str, Union[float, int]]] = dict()

        for iter_percent in self.twitch_width_percents:
            iter_list_of_width_values: List[Union[float, int]] = []
            for iter_twitch in metrics:
                iter_width_value = iter_twitch[iter_percent][TIME_VALUE_UUID]
                if not isinstance(iter_width_value, (float, int)):  # making mypy happy
                    raise NotImplementedError(
                        f"The width value under key {TIME_VALUE_UUID} must be a float or an int. It was: {iter_width_value}"
                    )
                iter_list_of_width_values.append(iter_width_value)
            iter_stats_dict = self.create_statistics_dict(
                metric=iter_list_of_width_values, rounded=self.rounded
            )
            width_stats_dict[iter_percent] = iter_stats_dict

        aggregate_dict[metric_id] = width_stats_dict

    def calculate_twitch_time_diff(
        self,
        twitch_indices: Dict[int, Dict[UUID, Optional[int]]],
        filtered_data: NDArray[(2, Any), int],
        per_twitch_widths: List[
            Dict[
                int,
                Dict[
                    UUID,
                    Any,
                ],
            ],
        ],
        is_contraction: bool = True,
    ) -> List[Dict[int, Dict[UUID, NDArray[float]]]]:
        """Calculate time from percent contraction / relaxation to twitch peak.

        Args:
            twitch_indices: a dictionary in which the key is an integer representing the time points
                of all the peaks of interest and the value is an inner dictionary with various UUIDs
                of prior/subsequent peaks and valleys and their index values.

            filtered_data: a 2D array of the time and value (magnetic, voltage, displacement, force...)
                data after it has gone through noise filtering

            per_twitch_widths: a list of dictionaries where the first key is the percentage of the
                way down to the nearby valleys, the second key is a UUID representing either the
                value of the width, or the rising or falling coordinates. The final value is either
                an int representing the width value or a tuple of ints for the x/y coordinates

            is_contraction: bool, specifies whether to compute time-to-peak for contraction or
                relaxation side of twitch
        Returns:
            time_differences: a list of dictionaries where the first key is the percentage of the way
            down to the nearby valleys, the second key is a UUID representing either the relaxation or
            contraction time.  The final value is float indicating time from relaxation/contraction to peak
        """
        # dictionary of time differences for each peak
        coord_type = WIDTH_RISING_COORDS_UUID
        if not is_contraction:
            coord_type = WIDTH_FALLING_COORDS_UUID

        time_differences: List[Dict[int, Dict[UUID, NDArray[float]]]] = list()
        time_series = filtered_data[0, :]

        # iterate over each twitch
        for i, (iter_twitch_idx, iter_twitch_info) in enumerate(twitch_indices.items()):

            peak_time_value = time_series[iter_twitch_idx]
            prior_valley_time_value = time_series[iter_twitch_info[PRIOR_VALLEY_INDEX_UUID]]

            # compile time differences for each peak
            iter_twich_difference_dict: Dict[int, Dict[UUID, NDArray[float]]] = dict()

            # iterate of width percentages
            for iter_percent in self.twitch_width_percents:

                iter_percent_difference_dict: Dict[UUID, NDArray[float]] = dict()

                iter_percent_coord_dict = per_twitch_widths[i][iter_percent]
                width_percent_time = iter_percent_coord_dict[coord_type][0]

                if is_contraction:
                    difference = width_percent_time - prior_valley_time_value
                else:
                    difference = width_percent_time - peak_time_value

                iter_percent_difference_dict[TIME_VALUE_UUID] = difference / MICRO_TO_BASE_CONVERSION
                iter_twich_difference_dict[iter_percent] = iter_percent_difference_dict

            time_differences.append(iter_twich_difference_dict)

        return time_differences


class TwitchPeakToBaseline(BaseMetric):
    """Calculate full contraction or full relaxation time."""

    def __init__(
        self,
        rounded: bool = False,
        is_contraction: bool = True,
        # twitch_width_percents: List[int] = None,
        **kwargs: Dict[str, Any],
    ):
        super().__init__(rounded=rounded, **kwargs)

        self.is_contraction = is_contraction

    def fit(
        self,
        peak_and_valley_indices: Tuple[NDArray[int], NDArray[int]],
        filtered_data: NDArray[(2, Any), int],
        twitch_indices: NDArray[int],
        **kwargs: Dict[str, Any],
    ) -> NDArray[Float64]:

        num_twitches = len(twitch_indices)
        full_differences: NDArray[Float64] = np.zeros(
            (num_twitches),
        )

        time_series = filtered_data[0, :]

        for i, (iter_twitch_idx, iter_twitch_info) in enumerate(twitch_indices.items()):
            peak_time = time_series[iter_twitch_idx]
            prior_valley_time = time_series[iter_twitch_info[PRIOR_VALLEY_INDEX_UUID]]
            subsequent_valley_time = time_series[iter_twitch_info[SUBSEQUENT_VALLEY_INDEX_UUID]]

            if self.is_contraction:
                time_difference = peak_time - prior_valley_time
            else:
                time_difference = subsequent_valley_time - peak_time

            if self.rounded:
                time_difference = int(np.round(time_difference))

            full_differences[i] = time_difference

        return full_differences / MICRO_TO_BASE_CONVERSION


def interpolate_x_for_y_between_two_points(  # pylint:disable=invalid-name # (Eli 9/1/20: I can't think of a shorter name to describe this concept fully)
    desired_y: Union[int, float],
    x_1: Union[int, float],
    y_1: Union[int, float],
    x_2: Union[int, float],
    y_2: Union[int, float],
) -> Union[int, float]:
    """Find a value of x between two points that matches the desired y value.

    Uses linear interpolation, based on point-slope formula.
    """
    if (x_2 - x_1) == 0:
        raise ZeroDivisionError("Denominator cannot be 0.")

    slope = (y_2 - y_1) / (x_2 - x_1)

    return (desired_y - y_1) / slope + x_1


def interpolate_y_for_x_between_two_points(  # pylint:disable=invalid-name # (Eli 9/1/20: I can't think of a shorter name to describe this concept fully)
    desired_x: Union[int, float],
    x_1: Union[int, float],
    y_1: Union[int, float],
    x_2: Union[int, float],
    y_2: Union[int, float],
) -> Union[int, float]:
    """Find a value of y between two points that matches the desired x value.

    Uses linear interpolation, based on point-slope formula.
    """
    if (x_2 - x_1) == 0:
        raise ZeroDivisionError("Denominator cannot be 0.")

    slope = (y_2 - y_1) / (x_2 - x_1)
    return slope * (desired_x - x_1) + y_1
