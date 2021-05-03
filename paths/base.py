import copy
from abc import ABC, abstractmethod

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike

from utilities import Constants
from utilities import Vector


class BasePath(ABC):
    def __init__(self):
        self._total_angle = None

    def visualize(self, fig=None, ax=None, show=True, frequency=None, color='black'):
        """
        Visualize the path on a matplotlib graph
        :param fig : Figure, optional
            If fig and ax are both provided,
            display the atmosphere colormap on top of the old axes
            otherwise, create a new figure and axes to work with
        :param ax : Axes, optional
            If fig and ax are both provided,
            display the atmosphere colormap on top of the old axes
            otherwise, create a new figure and axes to work with
        :param show : boolean, optional
            If show is true, display the plotted atmosphere immediately and return nothing.
            Otherwise, don't display and instead return the computed figure and axes
        :param frequency : string, optional
            The frequency of the wave being traced. Used in figure title
        :param color : string, optional
            The color of the path in the plot
        :returns : (Figure, Axes), optional
            If show is False, return the computed figure and axes, otherwise return nothing.
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(6, 4.5), num=0)
        if frequency is None:
            _frequency = "?"
        else:
            _frequency = int(frequency/1E6)
        ax.set_title(f"3D Ray Trace")
        ax.autoscale(False)
        class_type = self.__class__.__name__
        _label = f'{Constants.TYPE_ABBREVIATION[class_type]} - {_frequency} MHz'
        angular_distance = np.linspace(0, 1, 100)
        radii = np.linalg.norm(self(angular_distance), axis=-1)
        radii = (radii - Constants.EARTH_RADIUS) / 1000
        km_range = angular_distance * self.total_angle * Constants.EARTH_RADIUS / 1000
        max_x = km_range[-1]
        max_y = np.amax(radii)
        plt.xlim(-max_x*.05, max_x*1.05)
        plt.ylim(-max_y*.05, max_y*1.05)
        ax.plot(km_range, radii, color=color, label=_label)
        ax.legend()
        ax.set_ylabel("Altitude (km)")
        ax.set_xlabel("Range (km)")

        if show:
            plt.show()
        else:
            return fig, ax

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def transform_adjustments(self, indexes: ArrayLike, adjustments: ArrayLike) -> ArrayLike:
        """
        adjust the change vector (optional). This is used in the GCD path to
        reduce the magnitude of changes to the angular parameter. This is because we want to
        adjust on the order of radians, not meters.

        parameters are identical to those on the adjust_parameters function

        :returns : array_like
            A new adjustments vector (possibly changed in some way)
        """
        return adjustments

    def adjust_parameters(self, indexes: ArrayLike, adjustments: ArrayLike):
        """
        adjust_parameters takes vectors of indexes and adjustments and returns a new path
        with these adjustments applied

        :param indexes : array_like
            Indices of parameters to change
        :param adjustments : array_like
            Amount to change the parameters
        :returns : BasePath
            a copy of the current path, but with adjustable_parameters changed to reflect adjustments
        """
        adjusted_params = np.copy(self.adjustable_parameters)
        broadcast_indexes, broadcast_adjustments = [
            np.array(x) for x in np.broadcast_arrays(indexes, adjustments)
        ]
        indexes = np.asarray(indexes)
        adjusted_params[indexes] = adjusted_params[indexes] + \
            self.transform_adjustments(indexes, broadcast_adjustments)

        # Override the behavior of the __copy__ method to change how this works
        # __copy__ should take no parameters and return an identical version of the path
        # (but with no shared variables)
        new_path = copy.copy(self)
        new_path.adjustable_parameters = adjusted_params
        return new_path

    @property
    def total_angle(self):
        """
        The total angle in radians from start_point to end_point
        """
        if self._total_angle is None:
            self._total_angle = Vector.angle_between(self(0), self(1))
        return self._total_angle

    @property
    @abstractmethod
    def adjustable_parameters(self) -> np.ndarray:
        """
        adjustable_parameters is a numpy array of parameters that define all the tunable parameters of the path

        For the case of the GCD for example, we concatenate all variable radial and angular parameters to get this
        parameter. The output format of this function needs to match the input format for the value parameter
        in the adjustable_parameters(self, value) function.

        the parameters in this array are used to calculate the derivative
        of the phase distance over a path depending on these varied parameters.
        """
        raise NotImplementedError("Inheriting classes must override the adjustable_parameters getter")

    @adjustable_parameters.setter
    @abstractmethod
    def adjustable_parameters(self, value: np.ndarray):
        """
        adjustable_parameters is a numpy array of parameters that define all the tunable parameters of the path

        For the case of the GCD for example, we concatenate all variable radial and angular parameters to get this
        parameter. The output format of this function needs to match the input format for the value parameter
        in the adjustable_parameters(self, value) function

        the parameters in this array are used to calculate the derivative
        of the phase distance over a path depending on these varied parameters.
        """
        raise NotImplementedError(
            "Inheriting classes must override the adjustable_parameters property setter"
        )

    @abstractmethod
    def __call__(self, fraction: ArrayLike, nu: int = 0):
        """
        given a vector (or single value) of positions along the path, return the position vector (or derivative
        if nu > 0) along the path.

        :param fraction : array_like, shape (N,)
            the positions along the path. Each position is between 0 (start) and 1 (end)
        :param nu : int, optional
            the order of the derivative to return (0 for f(x), 1 for f'(x), ...)

        :returns : ArrayLike, shape (N, 3)
            the positions along the path in cartesian coordinates
        """
        raise NotImplementedError("Inheriting classes must override the __call__ method")
