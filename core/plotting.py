"""
Plotting functions
"""
from matplotlib import pyplot as plt
import numpy as np
from core import vector, coordinates as coords
import typing
import math
from scipy import integrate

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


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
        _label = f'{ray_type} - {_frequency} MHz'

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
    Plots the atmosphere given the model
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
    total_angle = vector.angle_between_vector_collections(
        coords.spherical_to_cartesian(initial_point),
        coords.spherical_to_cartesian(final_point)
    ).item()
    path_component_vector = np.empty((point_number, 3))
    path_component_vector[:, 0] = 1
    path_component_vector[:, 1] = np.linspace(0, 1, point_number) * total_angle * coords.EARTH_RADIUS
    path_component_vector[:, 2] = 0

    points = coords.path_component_to_standard(path_component_vector, initial_point, final_point)

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


def visualize_trace(
        calculated_paths: typing.Sequence[np.ndarray],
        fig: plt.Figure = None,
        ax: plt.Axes = None,
        plot_all_traces: bool = False,
        show: bool = True, **kwargs):
    """
    Plots the atmosphere given the model
    :param calculated_paths: A tuple of paths to plot.
    Each element in tuple is an iteration of the trace
    These need to be ordered by earliest iteration to newest
    :param fig: Existing pyplot figure to append new graph onto
    :param ax: Existing pyplot axes to append new graph onto
    :param plot_all_traces: If false, only plot the newest iteration.
        Otherwise plot them all.
    :param show: Whether or not to show the graph. If not return the fig/axes plotted onto
    :param kwargs: Additional arguments are passed to ax.plot
    :returns: If show is True, it returns the figure and axes. Otherwise it will return nothing.
    """
    if ax is None or fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

    ax.set_title(f"3D Ray Trace")
    ax.autoscale(False)
    if plot_all_traces and len(calculated_paths) > 1:
        custom_lines = [plt.Line2D([0], [0], color='black', lw=4),
                        plt.Line2D([0], [0], color='white', lw=4)]
        ax.legend(custom_lines, ['Best Trace', 'Earlier Traces'])
    else:
        custom_lines = [plt.Line2D([0], [0], color='black', lw=4)]
        ax.legend(custom_lines, ['Best Trace'])

    # Plot the initial calculated paths if desired
    if plot_all_traces:
        for i in range(len(calculated_paths) - 1):
            path = calculated_paths[i]
            radii = path[:, 0]
            radii = (radii - coords.EARTH_RADIUS)/1000
            km_range = radii * (path[0, 1] - path[-1, 1]) / 1000
            ax.plot(km_range, radii, color='white', **kwargs)

    # We always plot the last ones
    path = calculated_paths[-1]
    radii = path[:, 0]
    radii = (radii - coords.EARTH_RADIUS)/1000
    km_range = radii * (path[0, 1] - path[-1, 1]) / 1000
    ax.plot(km_range, radii, color='black')
    ax.set_ylabel("Altitude (km)")
    ax.set_xlabel("Range (km)")

    if show:
        plt.show()
        plt.close(fig)
    else:
        return fig, ax


def visualize_points(points: dict, fig=None, ax=None, show=True):
    """
    Plots the points in 3d space. All points need to be cartesian
    points: a dictionary of {label: point}
    """
    if ax is None or fig is None:
        fig = plt.figure(figsize=plt.figaspect(0.5) * 1.5)
        ax = fig.add_subplot(111, projection='3d')

    def axis_equal3d(old_ax):
        """
        This function was taken from stack overflow. It is supposed to set the x, y, z axis such that they are
        equal and the content is scaled properly. It is not working 100% though
        :param old_ax: The axes object to rescale
        :return: a new axis object with the axes rescaled
        """
        extents = np.array([getattr(old_ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
        sz = extents[:, 1] - extents[:, 0]
        centers = np.mean(extents, axis=1)
        maxsize = max(abs(sz))
        r = maxsize / 2
        for ctr, dim in zip(centers, 'xyz'):
            getattr(old_ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

    for label in points.keys():
        point = points[label]
        ax.scatter(*point.tolist())
        ax.text(*point.tolist(), label)

    u, v = np.mgrid[0:2 * np.pi:60j, 0:np.pi:30j]
    x = np.cos(u) * np.sin(v) * coords.EARTH_RADIUS
    y = np.sin(u) * np.sin(v) * coords.EARTH_RADIUS
    z = np.cos(v) * coords.EARTH_RADIUS
    ax.plot_wireframe(x, y, z, color="r")
    axis_equal3d(old_ax=ax)

    if show:
        plt.show()
    return fig, ax


def visualize_tracing_debug(r, r_dot):
    """
    This function is a debug tool that will help visualize the trace as it happens
    :param r: Array of cartesian points along the path
    :param r_dot: Array of vector derivatives at each point in r
    """
    step_size = 1/r.shape[0]
    rx = integrate.simps(r_dot[:, 0], dx=step_size)
    ry = integrate.simps(r_dot[:, 1], dx=step_size)
    rz = integrate.simps(r_dot[:, 2], dx=step_size)
    r_estimate = np.zeros_like(r)
    ip = r[0]
    fp = r[-1]
    r_estimate[0] = r[0]
    for i in range(1, r.shape[0]):
        r_estimate[i, 0] = integrate.simps(r_dot[:i, 0], dx=step_size) + r[0, 0]
        r_estimate[i, 1] = integrate.simps(r_dot[:i, 1], dx=step_size) + r[0, 1]
        r_estimate[i, 2] = integrate.simps(r_dot[:i, 2], dx=step_size) + r[0, 2]
    r_estimate[-1] = r[-1]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = coords.EARTH_RADIUS * np.outer(np.cos(u), np.sin(v))
    y = coords.EARTH_RADIUS * np.outer(np.sin(u), np.sin(v))
    z = coords.EARTH_RADIUS * np.outer(np.ones(u.size), np.cos(v))
    ax.plot_surface(x, y, z, color='b', alpha=0.4)
    ax.plot3D(r[:, 0], r[:, 1], r[:, 2], 'red')
    ax.plot3D(r_estimate[:, 0], r_estimate[:, 1], r_estimate[:, 2], 'green')
    plt.show()
    plt.plot(r_dot[:, 0], color='blue')
    plt.plot(r_dot[:, 1], color='red')
    plt.plot(r_dot[:, 2], color='green')
    plt.show()
    print(f"Total r-dot: {np.array([rx, ry, rz])}.")
    print(f"Total change: {fp - ip}")
