from Vector import spherical_to_cartesian, cartesian_to_spherical, angle_between, unit_vector
from Constants import EARTH_RADIUS
from numpy import array, exp, linspace, zeros, cross, sign, repeat
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation


class ChapmanLayers:
    def __init__(self, f0, hm, ym, gradient, start_point):
        self._parameters = array([f0, hm, ym])
        # Array-like parameter whose first element is the magnitude of the gradient in a unit of
        #   MHz per degree and whose second element is the parameter for the direction
        #   -1 for north, +1 for south, +2 for east, -2 for west
        self.gradient = gradient

        # Spherical start point
        self.start_point = start_point

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, new_params, new_gradient=None, new_start=None):
        self._parameters = new_params

        if new_gradient is not None:
            self.gradient = new_gradient

        if new_start is not None:
            self.start_point = new_start

    def plasma_frequency(self, coordinate, using_spherical=False):
        if not using_spherical:
            coordinate = cartesian_to_spherical(coordinate).reshape(-1, 3)
        h = coordinate[:, 0] - EARTH_RADIUS
        z1 = (h - self._parameters[1])/self._parameters[2]

        multiplier = self._parameters[0]
        if self.gradient is not None:
            multiplier = multiplier + self.gradient[0] * sign(self.gradient[1]) * \
                      (coordinate[:, abs(self.gradient[1])] -
                       cartesian_to_spherical(self.start_point)[abs(self.gradient[1])])
        output = multiplier*exp((1 - z1 - exp(-z1)))

        if len(output) == 1:
            return output[0]
        else:
            return output

    def visualize(self, initial_point, final_point, fig=None, ax=None,
                  point_number=100, show=False, using_spherical=False):
        if using_spherical:
            initial_point, final_point = spherical_to_cartesian(initial_point), spherical_to_cartesian(final_point)
        total_angle = angle_between(initial_point, final_point)
        normal_vec = unit_vector(cross(initial_point, final_point))
        if ax is None or fig is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
        radii = linspace(EARTH_RADIUS, EARTH_RADIUS + 400E3, point_number)
        alpha = linspace(0, total_angle, point_number)
        r_1 = Rotation.from_rotvec(normal_vec * alpha.reshape(-1, 1))
        v_1 = r_1.apply(initial_point)
        v_1 = cartesian_to_spherical(v_1)
        frequency_grid = zeros((point_number, point_number))
        for i in range(point_number):
            plotted_vecs = repeat(v_1[i].reshape(-1, 1), point_number, axis=1).T
            plotted_vecs[:, 0] = radii
            frequency_grid[:, i] = self.plasma_frequency(plotted_vecs, using_spherical=True)/1E6
        image = ax.imshow(frequency_grid, cmap='gist_rainbow', interpolation='bilinear', origin='lower',
                          alpha=1, aspect='auto', extent=[0, total_angle*EARTH_RADIUS/1000, 0, 400])
        ax.yaxis.set_ticks_position('both')
        color_bar = fig.colorbar(image, ax=ax)
        color_bar.set_label("Plasma Frequency (MHz)")

        if show:
            ax.set_title("Chapman Layers Atmosphere")
            plt.show()
        else:
            return fig, ax
