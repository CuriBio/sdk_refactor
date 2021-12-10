# -*- coding: utf-8 -*-
"""More accurate estimation of magnet positions."""
from typing import Any
from typing import Dict
from typing import List
from typing import TYPE_CHECKING

from nptyping import NDArray
from numba import njit
import numpy as np
from scipy.optimize import least_squares
import scipy.signal as signal

from .constants import TISSUE_SENSOR_READINGS
from .constants import WELL_IDX_TO_MODULE_ID


if TYPE_CHECKING:
    from .plate_recording import WellFile


# Kevin (12/1/21): Sensor locations relative to origin
SENSOR_DISTANCES_FROM_CENTER_POINT = np.asarray([[-2.15, 1.7, 0], [2.15, 1.7, 0], [0, -2.743, 0]])

ADJACENT_WELL_DISTANCE_MM = 19.5  # TODO Tanner (12/2/21): make sure this constant is named correctly
WELL_VERTICAL_SPACING = np.asarray([0, -ADJACENT_WELL_DISTANCE_MM, 0])
WELL_HORIZONTAL_SPACING = np.asarray([ADJACENT_WELL_DISTANCE_MM, 0, 0])
WELLS_PER_ROW = 6

FULL_CIRCLE_DEGREES = 360
FULL_CIRCLE_RADIANS = 2 * np.pi
RAD_PER_DEGREE = FULL_CIRCLE_RADIANS / FULL_CIRCLE_DEGREES

# Kevin (12/1/21): Used for calculating magnet's dipole moment
MAGNET_VOLUME = np.pi * (0.75 / 2.0) ** 2
# Kevin (12/1/21): This is part of the dipole model
DIPOLE_MODEL_FACTOR = 4 * np.pi

NUM_PARAMS = 6


@njit()
def meas_field(
    xpos: NDArray[(1, Any), float],
    ypos: NDArray[(1, Any), float],
    zpos: NDArray[(1, Any), float],
    theta: NDArray[(1, Any), float],
    phi: NDArray[(1, Any), float],
    remn: NDArray[(1, Any), float],
    manta: NDArray[(72, 3), float],
    num_active_module_ids: int,
) -> NDArray[(1, Any), float]:
    """Simulate fields using a magnetic dipole model."""

    fields = np.zeros((len(manta), 3))
    # Tanner (12/2/21): numba doesn't like using *= here, for some reason it thinks the dtype of phi and theta are int64 yet they are float64
    theta = theta * RAD_PER_DEGREE  # magnet pitch
    phi = phi * RAD_PER_DEGREE  # magnet yaw

    # Kevin (12/1/21): This simulates the fields for each magnet at each sensor on the device.
    # Each magnet has particular values for xpos -> remn.
    # These are iteratively optimized to match the simulated fields within a certain tolerance, at which point the algorithm terminates
    for magnet in range(
        0, num_active_module_ids
    ):  # TODO Tanner (12/2/21): If it is necessary to speed the algorithm up, this is the best place to attempt an optimization as the algorithm spends a significant amount of time performing these calculations
        # Kevin (12/1/21): compute moment vectors based on magnet strength and orientation
        moment_vectors = (
            MAGNET_VOLUME
            * remn[magnet]
            * np.asarray(
                [
                    np.sin(theta[magnet]) * np.cos(phi[magnet]),
                    np.sin(theta[magnet]) * np.sin(phi[magnet]),
                    np.cos(theta[magnet]),
                ]
            )
        )
        # Kevin (12/1/21): compute distance vector from origin to moment
        r = -np.asarray([xpos[magnet], ypos[magnet], zpos[magnet]]) + manta
        # Kevin (12/1/21): Calculate the euclidian distance from the magnet to a given sensor
        r_abs = np.sqrt(np.sum(r ** 2, axis=1))
        # Kevin (12/1/21): simulate fields at sensors using dipole model for each magnet
        for field in range(0, len(r)):
            fields[field] += (
                3 * r[field] * np.dot(moment_vectors, r[field]) / r_abs[field] ** 5
                - moment_vectors / r_abs[field] ** 3
            )
    # Kevin (12/1/21): Reshaping to match the format of the data coming off the mantarray
    return fields.reshape((1, 3 * len(r)))[0] / DIPOLE_MODEL_FACTOR


def objective_function_ls(
    pos: NDArray[(1, Any), float],
    b_meas: NDArray[(1, Any), float],
    manta: NDArray[(72, 3), float],
    num_active_module_ids: int,
) -> NDArray[(1, Any), float]:
    """Cost function to be minimized by the least squares."""
    pos = pos.reshape(NUM_PARAMS, num_active_module_ids)
    x = pos[0]
    y = pos[1]
    z = pos[2]
    theta = pos[3]
    phi = pos[4]
    remn = pos[5]

    b_calc = meas_field(x, y, z, theta, phi, remn, manta, num_active_module_ids)
    return b_calc - b_meas


def get_positions(data: NDArray[(24, 3, 3, Any), float]) -> Dict[str, NDArray[(1, Any), float]]:
    """Generate initial guess data and run least squares optimizer on
    instrument data to get magnet positions.

    Takes an array indexed as [well, sensor, axis, timepoint] Data
    should be the difference of the data with plate on the instrument
    and empty plate calibration data Assumes 3 active sensors for each
    well, that all active wells have magnets, and that all magnets have
    the well beneath them active
    """
    # Tanner (12/3/21): hardcoding these for now. Could be made constants if appropriate
    num_sensors = 3
    num_axes = 3

    # Tanner (12/2/21): Every well/sensor/axis will always be active as of now
    num_active_module_ids = 24
    active_module_ids = list(range(num_active_module_ids))

    # Kevin (12/1/21): Manta gives the locations of all active sensors on all arrays with respect to a common point
    # computing the locations of each centrally located point about which each array is to be distributed,
    # for the purpose of offsetting the values in triad by the correct well spacing
    # The values in "triad" and "manta" relate to layout of the board itself so they don't change at all so long as the board doesn't
    triad = SENSOR_DISTANCES_FROM_CENTER_POINT.copy()
    manta = np.empty((triad.shape[0] * num_active_module_ids, triad.shape[1]))
    for module_id in range(0, num_active_module_ids):
        module_slice = slice(module_id * triad.shape[0], (module_id + 1) * triad.shape[0])
        manta[module_slice, :] = (
            triad
            + module_id // WELLS_PER_ROW * WELL_VERTICAL_SPACING
            + (module_id % WELLS_PER_ROW) * WELL_HORIZONTAL_SPACING
        )

    # Kevin (12/1/21): run meas_field with some dummy values so numba compiles it. There needs to be some delay before it's called again for it to compile
    dummy = np.asarray([1])
    meas_field(dummy, dummy, dummy, dummy, dummy, dummy, manta, num_active_module_ids)

    # Kevin (12/1/21): Initial guess is dependent on where the plate sits relative to the sensors
    initial_guess_values = {"X": 0, "Y": 1, "Z": -5, "THETA": 95, "PHI": 0, "REMN": -575}
    # Kevin (12/1/21): Each magnet has its own positional coordinates and other characteristics depending on where it's located in the consumable. Every magnet
    # position is referenced with respect to the center of the array beneath well A1, so the positions need to be adjusted to account for that, e.g. the magnet in
    # A2 has the x/y coordinate (19.5, 0), so guess is processed in the below loop to produce that value. prev_guess contains the guesses for each magnet at each position
    prev_guess = [
        initial_guess_values["X"] + ADJACENT_WELL_DISTANCE_MM * (module_id % WELLS_PER_ROW)
        for module_id in active_module_ids
    ]
    prev_guess.extend(
        [
            initial_guess_values["Y"] - ADJACENT_WELL_DISTANCE_MM * (module_id // WELLS_PER_ROW)
            for module_id in active_module_ids
        ]
    )
    for param in list(initial_guess_values.keys())[2:]:
        prev_guess.extend([initial_guess_values[param]] * num_active_module_ids)

    params = tuple(initial_guess_values.keys())
    estimations = {param: np.empty((data.shape[-1], num_active_module_ids)) for param in params}

    # Tanner (12/8/21): should probably add some sort of logging eventually

    # Kevin (12/1/21): Run the algorithm on each time index. The algorithm uses its previous outputs as its initial guess for all datapoints but the first one
    for data_idx in range(0, data.shape[-1]):
        # Kevin (12/1/21): This sorts the data from processData into something that the algorithm can operate on; it shouldn't be necessary if you combine this method and processData
        b_meas = np.empty(num_active_module_ids * 9)
        for mi_idx, module_id in enumerate(active_module_ids):
            # Kevin (12/1/21): rearrange sensor readings as a 1d vector
            b_meas[mi_idx * 9 : (mi_idx + 1) * 9] = data[module_id, :, :, data_idx].reshape(
                (1, num_sensors * num_axes)
            )

        res = least_squares(
            objective_function_ls,
            prev_guess,
            args=(b_meas, manta, num_active_module_ids),
            method="trf",
            ftol=1e-2,
        )

        outputs = np.asarray(res.x).reshape(NUM_PARAMS, num_active_module_ids)
        for i, param in enumerate(params):
            estimations[param][data_idx, :] = outputs[i]

        # Tanner (12/2/21): set the start point for next loop to the result of this loop
        prev_guess = np.asarray(res.x)

    # Kevin (12/1/21): I've gotten some strange results from downsampling; I'm not sure why that is necessarily, could be aliasing,
    # could be that the guesses for successive runs need to be really close together to get good accuracy.
    # For large files, you may be able to use the 1D approximation after running the algorithm once or twice "priming"
    return estimations


def find_magnet_positions(
    fields: NDArray[(24, 3, 3, Any), float],
    baseline: NDArray[(24, 3, 3, Any), float],
    filter_outputs: bool = True,
) -> Dict[str, NDArray[(1, Any), float]]:
    output_dict = get_positions(fields - baseline)
    if filter_outputs:
        for param, output_arr in output_dict.items():
            output_dict[param] = filter_magnet_positions(output_arr)
    return output_dict


def filter_magnet_positions(magnet_positions: NDArray[(1, Any), float]) -> NDArray[(1, Any), float]:
    high_cut_hz = 30
    b, a = signal.butter(4, high_cut_hz, "low", fs=100)
    filtered_magnet_positions = signal.filtfilt(b, a, magnet_positions)
    return filtered_magnet_positions


def format_well_file_data(well_files: List["WellFile"]) -> NDArray[(24, 3, 3, Any), float]:
    # convert well data into the array format that the magnet finding alg uses
    plate_data_array = None
    for well_idx, well_file in enumerate(well_files):
        module_id = WELL_IDX_TO_MODULE_ID[well_idx]
        tissue_data = well_file[TISSUE_SENSOR_READINGS][:]
        if plate_data_array is None:
            num_samples = tissue_data.shape[-1]
            plate_data_array = np.empty((24, 3, 3, num_samples))
        reshaped_data = tissue_data.reshape((3, 3, num_samples))
        plate_data_array[module_id - 1, :, :, :] = reshaped_data
    return plate_data_array
