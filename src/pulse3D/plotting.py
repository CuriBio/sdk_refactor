# -*- coding: utf-8 -*-
from typing import Dict
from typing import Union

import numpy as np

from .constants import CHART_ALPHA
from .constants import CHART_GAMMA
from .constants import CHART_PIXELS_PER_SECOND

"""
"""


def compute_chart_width(
    N: Union[int, float],
    pixPerSec: Union[int, float] = CHART_PIXELS_PER_SECOND,
    alpha: Union[int, float] = CHART_ALPHA,
    gamma: Union[int, float] = CHART_GAMMA,
) -> float:
    """Compute full figure (chart) width (in pixels), as a function of number
    of seconds in plate recording.

    Args:
        N (int,float): number of seconds in recording
        pixPerSec (float): number of pixels per second
        alpha (int): pixels to left of plot area
        gamma (int): pixels to right of plot area

    Returns:
        chartwidth (int): width of chart (in pixels)
    """
    chartwidth = np.ceil(alpha + gamma + (pixPerSec * N))
    return chartwidth


def compute_x_coordinate(chart_width: float, alpha: Union[int, float] = CHART_ALPHA) -> float:
    """Compute the position of the upper-left plot position, as a function of
    chart width (in pixels), and number of pixels to left of plot.

    Args:
        chart_width (int): width of figure (in pixels)
        alpha (int): pixels to left of plot area   # Tanner (12/15/22): this description seems wrong

    Returns:
        x-coordinate of plot area (where origin is upper left of chart), as a percentage

    Raises:
        ValueError: alpha must be less than or equal to chart width
    """
    if chart_width <= alpha:
        raise ValueError("Chart width must be greater than left chart boundary.")

    return alpha / chart_width


def compute_plot_width(
    chart_width: float, alpha: Union[int, float] = CHART_ALPHA, gamma: Union[int, float] = CHART_GAMMA
) -> float:
    """Compute plot area width (as a percentage).

    Args:
        chart_width (float): width of chart (in pixels)
        alpha (int,float): pixels to left of plot area
        gamma (int,float): pixels to right of plot area

    Returns:
        plot width, as a percentage of chart width

    Raises:
        ValueError: chart width must be greater than sum of alpha and gamma
    """
    figure_boundaries = alpha + gamma
    if chart_width <= figure_boundaries:
        raise ValueError("Chart width must be greater than figure boundaries.")

    return (chart_width - figure_boundaries) / chart_width


def plotting_parameters(
    N: Union[int, float],
    alpha: Union[int, float] = CHART_ALPHA,
    gamma: Union[int, float] = CHART_GAMMA,
    include_y2_axis: bool = False,
) -> Dict[str, float]:
    """Estimate plotting parameters for a given number of time samples.

    Args:
        N (int,float): number of seconds in recording
        alpha (int): pixels to left of plot area
        gamma (int): pixels to right of plot area

    Raises:
        ValueError: alpha, gamma, and N must all be greater than 0
    """
    if alpha <= 0:
        raise ValueError("Left chart boundary (alpha) must be positive.")
    if gamma <= 0:
        raise ValueError("Right chart boundary (gamma) must be positive.")
    if N <= 0:
        raise ValueError("Number of time samples (N) must be positive.")

    if include_y2_axis:
        gamma += 50

    chart_width = compute_chart_width(N, alpha=alpha, gamma=gamma)
    plot_width = compute_plot_width(chart_width, alpha=alpha, gamma=gamma)
    x_coordinate = compute_x_coordinate(chart_width, alpha=alpha)

    return {"chart_width": chart_width, "plot_width": plot_width, "x": x_coordinate}
