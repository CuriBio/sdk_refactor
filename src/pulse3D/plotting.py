import numpy as np
from typing import Dict
from typing import Union

from .constants import CHART_ALPHA
from .constants import CHART_GAMMA
from .constants import CHART_PIXELS_PER_SECOND

"""
XlsxWriter chart and plot sizing functions, set_size() and set_plotarea(),
are a bit arcance -- hence the need for these functions.
For documentation on XlsxWriter figure sizing parameters, see here: 
https://xlsxwriter.readthedocs.io/working_with_charts.html#chart-layout
"""

def compute_chart_width(N: Union[int, float],
                        pixPerSec: Union[int,float]=CHART_PIXELS_PER_SECOND, 
                        alpha: Union[int,float]=CHART_ALPHA, 
                        gamma: Union[int,float]=CHART_GAMMA) -> float:
    """
    Compute full figure (chart) width (in pixels), as a function of number of seconds in plate recording.
    
    Args:
    N (int,float): number of seconds in recording
    pixPerSec (float): number of pixels per second
    alpha (int): pixels to left of plot area
    gamma (int): pixels to right of plot area
    
    Returns:
    chartwidth (int): width of chart (in pixels)
    """
    chartwidth = np.ceil(alpha + gamma + (pixPerSec*N))
    return chartwidth

def compute_x_coordinate(chart_width: float, 
                         alpha: Union[int,float]=CHART_ALPHA) -> float:
    """
    Compute the position of the upper-left plot position, as a function of chart width (in pixels), and number of pixels to left of plot.
    
    Parameters:
    - - - - - 
    chart_width (int): width of figure (in pixels)
    alpha (int): pixels to left of plot area
    
    Returns:
    - - - -
    x_coord (float): percentage, x-coordinate of plot area (where origin is upper left of chart)
    """
    x_coord = alpha / chart_width
    return x_coord

def compute_plot_width(chart_width: float, 
                       alpha:Union[int,float]=CHART_ALPHA, 
                       gamma:Union[int,float]=CHART_GAMMA) -> float:
    """
    Compute plot area width (as a percentage).

    Args:
    chart_width (float): width of chart (in pixels)
    alpha (int,float): pixels to left of plot area
    gamma (int,float): pixels to right of plot area

    Returns:
    plotwidth (float): plot width, as a percentage of chart width
    """
    pAlpha = alpha/chart_width
    pGamma = gamma/chart_width

    plotwidth=(1.0-(pAlpha+pGamma))
    
    return plotwidth

def plotting_parameters(N:Union[int,float]) -> Dict[str, float]:

    """
    
    """

    chart_width = compute_chart_width(N)
    plot_width = compute_plot_width(chart_width)
    x_coordinate = compute_x_coordinate(chart_width)

    return {'chart_width': chart_width, 'plot_width': plot_width, 'x': x_coordinate}