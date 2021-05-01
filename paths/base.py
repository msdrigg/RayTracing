from abc import ABC, abstractmethod

from matplotlib import pyplot as plt
import numpy as np


class BasePath(ABC):
    def visualize(self, fig=None, ax=None, show=True, frequency=None, color=None):
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(6, 4.5), num=0)
        if frequency is None:
            _frequency = "?"
        else:
            _frequency = int(frequency/1E6)
        ax.set_title(f"3D Ray Trace")
        ax.autoscale(False)
        if color is None:
            _color = 'black'
        else:
            _color = color
        class_type = self.__class__.__name__
        _label = f'{TYPE_ABBREVIATION[class_type]} - {_frequency} MHz'
        angular_distance = np.linspace(0, 1, 100)
        radii = self(angular_distance, use_spherical=True)[:, 0]
        radii = (radii - EARTH_RADIUS) / 1000
        km_range = angular_distance * self.total_angle * EARTH_RADIUS / 1000
        max_x = km_range[-1]
        max_y = amax(radii)
        plt.xlim(-max_x*.05, max_x*1.05)
        plt.ylim(-max_y*.05, max_y*1.05)
        ax.plot(km_range, radii, color=_color, label=_label)
        ax.legend()
        ax.set_ylabel("Altitude (km)")
        ax.set_xlabel("Range (km)")

        if show:
            plt.show()
        else:
            return fig, ax

    @property
    @abstractmethod
    def parameters(self):
        raise NotImplementedError("Inheriting classes must override the parameters property")

    @abstractmethod
    def __call__(self, fraction, nu=0, use_spherical=False):
        raise NotImplementedError("Inheriting classes must override the __call__ method")

    @property
    def total_angle(self):
        raise NotImplementedError("Inheriting classes must override the total_angle property")

