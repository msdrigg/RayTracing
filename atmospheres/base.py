
from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
import re
from typing import Optional, Tuple


class BaseAtmosphere(ABC):
    def visualize(
        self, initial_point: np.ndarray, 
        final_point: np.ndarray,
        fig: plt.Figure = None,
        ax: plt.Axes = None,
        show: bool=False, **kwargs
    ) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Given an initial and final point  (and some formatting parameters), visualize the 
    atmosphere on a matplotlib graph

    :param initial_point : np.ndarray, shape (3, )
        3-vec cooreposing to the starting coordinate of the path in 
        cartesian coordinates. Used to determine the interval in which 
        to calculate the atmosphere
    :param final_point : np.ndarray, shape (3, )
        3-vec cooreposing to the starting coordinate of the path in 
        cartesian coordinates. Used to determine the interval in which 
        to calculate the atmosphere
    :param fig : Figure, optional
        If fig and ax are both provided, 
        display the atmosphere colormap ontop of the old axes
        otherwise, create a new figure and axes to work with
    :param ax : Axes, optional
        If fig and ax are both provided, 
        display the atmosphere colormap ontop of the old axes
        otherwise, create a new figure and axes to work with
    :param show : boolean, optional
        If show is true, display the plotted atmosphere immediately and return nothing.
        Otherwise, don't display and instead return the computed figure and axes
    :param kwargs : dict, optional
        Any additional kwargs are passed to the imshow function
    :returns : (Figure, Axes), optional
        If show is False, return the computed figure and axes, otherwise return nothing.
    """
        total_angle = angle_between(initial_point, final_point)
        normal_vec = unit_vector(cross(initial_point, final_point))
        point_number = 1000

        if ax is None or fig is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
        radii = np.linspace(EARTH_RADIUS, EARTH_RADIUS + 400E3, point_number)
        alpha = np.linspace(0, total_angle, point_number)
        r_1 = Rotation.from_rotvec(normal_vec * alpha.reshape(-1, 1))
        v_1 = r_1.apply(initial_point)
        v_1 = cartesian_to_spherical(v_1)
        frequency_grid = zeros((point_number, point_number))
        for i in range(point_number):
            plotted_vecs = repeat(v_1[i].reshape(-1, 1), point_number, axis=1).T
            plotted_vecs[:, 0] = radii
            frequency_grid[:, i] = self.plasma_frequency(plotted_vecs, using_spherical=True)/1E6
        image = ax.imshow(frequency_grid, cmap='gist_rainbow', interpolation='bilinear', origin='lower',
                          alpha=1, aspect='auto', extent=[0, total_angle*EARTH_RADIUS/1000, 0, 400],
                          **kwargs)
        ax.yaxis.set_ticks_position('both')
        color_bar = fig.colorbar(image, ax=ax)
        color_bar.set_label("Plasma Frequency (MHz)")

        if show:
            # We want to name our graph using class name of actual atmosphere
            class_name = type(self).__name__  # something like BaseAtmosphere
            # This code splits a name according to camel case so BaseAtmosphere is transformed into ['Base', 'Atmosphere']
            class_name_split = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', class_name)
            # Join all items into list and remove "atmosphere" if it is already in name
            final_name = ' '.join([x.capitalize() for x in class_name_split if x.lower() != "atmosphere"])
            ax.set_title(final_name + " Atmosphere")
            fig.show()
            plt.close(fig)
        else:
            return fig, ax

    @abstractmethod
    def plasma_frequency(self, position: np.ndarray) -> np.ndarray:
        """
        This function returns the plasma frequency given a position in space

        :param position : np.ndarray, shape (N, 3) or (3, )
            3-vec or array of 3-vecs in a cartesian coordinate system
        :returns The plasma frequency cooresponding to the input position vector. Its shape should match
        the incoming position vector's shape. If position is of shape (N, 3), the output should be 
        of shape (N, ). If the position is of shape (3, ), the output should be a scalar.
        """
        raise NotImplementedError("Inheriting classes must override the plasma_frequency method")
