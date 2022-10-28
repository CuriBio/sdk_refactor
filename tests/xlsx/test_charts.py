# -*- coding: utf-8 -*-
from pulse3D import plotting
from pulse3D.constants import CHART_ALPHA
from pulse3D.constants import CHART_GAMMA
from pulse3D.constants import CHART_PIXELS_PER_SECOND
import pytest


def test_plotting_params():
    # check chart size when given no data
    with pytest.raises(ValueError):
        plotting.plotting_parameters(N=0)

    with pytest.raises(ValueError):
        plotting.plotting_parameters(N=-1)

    with pytest.raises(ValueError):
        plotting.plotting_parameters(N=1, alpha=0)

    with pytest.raises(ValueError):
        plotting.plotting_parameters(N=1, gamma=0)


def test_chart_width():
    # check that chart width is computed correctly
    N = 1
    expected = (CHART_ALPHA + CHART_GAMMA) + (CHART_PIXELS_PER_SECOND * N)
    actual = plotting.compute_chart_width(N=N, alpha=CHART_ALPHA, gamma=CHART_GAMMA)
    assert expected == actual


def test_plot_width():
    # total distance from chart edge to plot edge cannot be greater than total chart width
    with pytest.raises(ValueError):
        plotting.compute_plot_width(chart_width=100, alpha=60, gamma=60)

    # check that plot width is computed correctly
    chart_width = 2 * (CHART_ALPHA + CHART_GAMMA)
    expected = 0.5
    actual = plotting.compute_plot_width(chart_width, alpha=CHART_ALPHA, gamma=CHART_GAMMA)
    assert expected == actual


def test_x_coordinate():
    # plot X position can never be greater than 1
    with pytest.raises(ValueError):
        plotting.compute_x_coordinate(chart_width=100, alpha=101)

    # check that plot X position is computed correctly
    expected = 0.5
    actual = plotting.compute_x_coordinate(chart_width=100, alpha=50)
    assert expected == actual
