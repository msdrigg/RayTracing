    _poly_fit = UnivariateSpline(distances, heights, k=degrees, s=0, ext=0)


class GreatCircleDeviation():
    def adjust_parameters(self, indexes, adjustments, mutate=False):
        # TODO: Add some parameters to control the adjustment rate to account for unresponsive parameters
        #  specifically radial params in the case of GCD path
        adjusted_params = self.parameters
        indexes = asarray(indexes).flatten()
        copied_adjustments = asarray(adjustments).flatten()
        if len(indexes) != len(copied_adjustments):
            copied_adjustments = repeat(copied_adjustments, len(indexes))
        copied_adjustments[indexes >= self.radial_param_number - 2] = \
            copied_adjustments[indexes >= self.radial_param_number - 2] / EARTH_RADIUS
        adjusted_params[indexes, 1] = adjusted_params[indexes, 1] + copied_adjustments
        if self._poly_fit_cartesian is None:
            self.interpolate_params()
        if not mutate:
            new_path = GreatCircleDeviation(
                len(self._radial_positions) - 2,
                len(self._angular_deviations) - 2,
                initial_parameters=adjusted_params[:, 1],
                initial_coordinate=self(0),
                final_coordinate=self(1),
                using_spherical=False
            )

            new_path.interpolate_params()
            return new_path
        else:
            self._radial_positions[1:-1] = adjusted_params[:self.radial_param_number]
            self._angular_deviations[1:-1] = adjusted_params[self.radial_param_number:]

            self.interpolate_params()

    def __call__(self, fraction, nu=0, use_spherical=False):
        point = np.array(list(map(lambda poly_fit: poly_fit(fraction, nu=nu), self._poly_fit_cartesian))).T


    def interpolate_params(self, radial=False, degree=3):
        self._poly_fit_angular = UnivariateSpline(self._angular_deviations[:, 0],
                                                  self._angular_deviations[:, 1],
                                                  k=min(degree, len(self._angular_deviations) - 1), s=0, ext=0)
        if radial:
            self._poly_fit_radial = UnivariateSpline(self._radial_positions[:, 0],
                                                     self._radial_positions[:, 1],
                                                     k=degree, s=0, ext=0)
        else:
            cartesian_points = zeros((len(self._radial_positions), 3))
            for index in range(len(self._radial_positions)):
                alpha = self._radial_positions[index, 0]
                r_1 = Rotation.from_rotvec(self.normal_vec * alpha * self.total_angle)
                v_1 = r_1.apply(self.initial_point)
                rotation_vec_2 = Vector.unit_vector(cross(self.normal_vec, v_1))
                rotation_vec_2 *= self._poly_fit_angular(alpha)
                r_2 = Rotation.from_rotvec(rotation_vec_2)
                v_2 = r_2.apply(v_1)
                v_2 *= self._radial_positions[index, 1]
                cartesian_points[index] = v_2
            self._poly_fit_cartesian = []
            for index in range(3):
                self._poly_fit_cartesian.append(UnivariateSpline(self._radial_positions[:, 0],
                                                                 cartesian_points[:, index], k=degree, s=0))
