from matplotlib import pyplot as plt
import numpy as np
from utils import coordinates as coords
import typing
import math
import constants


def visualize_path(path_components: np.ndarray,
                   fig: typing.Optional[plt.Figure] = None,
                   ax: typing.Optional[plt.Axes] = None, show: typing.Optional[bool] = True,
                   frequency: typing.Optional[float] = None,
                   ray_type: typing.Optional[str] = None,
                   **kwargs) -> typing.Optional[typing.Tuple[plt.Figure, plt.Axes]]:
    """
    Plots the path provided
    :param path_components: Array of points to plot for the path.
        Array is of the shape (N, 2) where each row is ground_distance, height_above_earth
    :param fig: Existing pyplot figure to append new graph onto
    :param ax: Existing pyplot axes to append new graph onto
    :param show: Whether or not to show the graph. If not return the fig/axes plotted onto
    :param frequency: The frequency of the ray path. This shows up as a label.
    :param kwargs: All other kwargs are passed to plotting function
    :param ray_type: String label for the ray. Optional
    :returns: If show is True, it returns the figure and axes. Otherwise it will return nothing.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(6, 4.5), num=0)
    if frequency is None:
        _frequency = "?"
    else:
        _frequency = int(frequency / 1E6)
    ax.set_title(f"Ray Trace")

    if ray_type is None:
        _label = f'{_frequency} MHz'
    else:
        _label = f'{constants.TYPE_ABBREVIATION[ray_type]} - {_frequency} MHz'

    heights = (path_components[:, 1]) / 1000
    distances = path_components[:, 0] / 1000
    max_x = distances[-1]
    max_y = np.amax(heights)
    plt.xlim(-max_x * .05, max_x * 1.05)
    plt.ylim(-max_y * .05, max_y * 1.05)
    ax.plot(distances, heights, label=_label, **kwargs)
    ax.legend()
    ax.set_ylabel("Altitude (km)")
    ax.set_xlabel("Range (km)")
    ax.autoscale()

    if show:
        plt.show()
    else:
        return fig, ax


def visualize_atmosphere(plasma_frequency_function: typing.Callable[[np.ndarray], np.ndarray],
                         initial_point: np.ndarray, final_point: np.ndarray,
                         fig: typing.Optional[plt.Figure] = None,
                         ax: typing.Optional[plt.Axes] = None,
                         show: typing.Optional[bool] = True,
                         point_number: int = 400,
                         max_height: float = None,
                         **kwargs) -> typing.Optional[typing.Tuple[plt.Figure, plt.Axes]]:
    """
    Plots the path provided
    :param plasma_frequency_function:
        Function that maps a vector of spherical coordinates into an array of plasma frequencies
        The callable will be passed an array of size (N, 3) and must return an array of size (N,)
    :param initial_point: Initial path point in spherical coordinates to decide where to plot atmosphere
    :param final_point: Final path point in spherical coordinates to decide where to plot atmosphere
    :param fig: Existing pyplot figure to append new graph onto
    :param ax: Existing pyplot axes to append new graph onto
    :param show: Whether or not to show the graph. If not return the fig/axes plotted onto
    :param point_number: Atmosphere is plotted on a grid. This value is grid length
    :param max_height: Maximum height above earth's surface in meters of the graph
    :param kwargs: Additional arguments are passed to ax.imshow
    :returns: If show is True, it returns the figure and axes. Otherwise it will return nothing.
    """
    default_height = 400E3
    total_angle = math.acos(np.dot(coords.spherical_to_cartesian(initial_point),
                            coords.spherical_to_cartesian(final_point)) /
                            (np.linalg.norm(initial_point) * np.linalg.norm(final_point)))
    path_component_vector = np.empty((point_number, 3))
    path_component_vector[:, 0] = 1
    path_component_vector[:, 1] = np.linspace(0, 1, point_number) * total_angle * coords.EARTH_RADIUS
    path_component_vector[:, 2] = 0

    points = coords.path_component_to_spherical(path_component_vector, initial_point, final_point)

    if max_height is None:
        if ax is not None:
            max_height = ax.get_ylim()[1]
        else:
            max_height = default_height

    if ax is None or fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

    frequency_grid = np.zeros((point_number, point_number))

    for i in range(point_number):
        plotted_vecs = np.repeat(points[i].reshape(-1, 1), point_number, axis=1).T
        plotted_vecs[:, 0] = np.linspace(coords.EARTH_RADIUS, coords.EARTH_RADIUS + max_height * 1000, point_number)
        frequency_grid[:, i] = plasma_frequency_function(plotted_vecs)/1E6

    image = ax.imshow(frequency_grid, cmap='gist_rainbow', interpolation='bilinear', origin='lower',
                      alpha=1, aspect='auto', extent=[0, total_angle*coords.EARTH_RADIUS/1000, 0, max_height],
                      **kwargs)
    ax.autoscale()
    ax.yaxis.set_ticks_position('both')
    color_bar = fig.colorbar(image, ax=ax)
    color_bar.set_label("Plasma Frequency (MHz)")

    if show:
        ax.set_title("Chapman Layers Atmosphere")
        plt.show()
    else:
        return fig, ax


def visualize_ground(total_angle, **kwargs):
    return None
