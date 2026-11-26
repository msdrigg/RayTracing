"""
Point-to-point ray tracing by the direct variational method of Coleman (2011).

The phase path of a trial path r(s), s in [0, 1], is

    P = int_0^1 F(r, r') ds,     F(r, r') = mu(r, Yp) * pt * |r'|

where mu * pt is the "ray refractive index" in the direction of the path
(see tracing/coleman_equations.py). A ray is a path for which P is stationary
with respect to every path parameter a_i, so we solve dP/da_i = 0 with
Newton-Raphson iterations.

How the derivatives are calculated
----------------------------------
The original version of this code obtained the gradient and Hessian of P by
finite differencing P itself with respect to every pair of parameters, which
cost thousands of full path integrals per Newton step. Here the chain rule is
used instead. With z = (r, r') the 6 numbers that F depends on at one
quadrature node,

    dP/da_i       = sum_q  w_q  (dz_q/da_i) . grad F(z_q)
    d2P/da_i da_j = sum_q  w_q  (dz_q/da_i) . Hess F(z_q) . (dz_q/da_j)

(the second line is exact when the path is linear in its parameters, which
GreatCircleDeviation is). dz/da comes from the path, and the 6-vector grad F
and 6x6 matrix Hess F are found by finite differences of F *at each node*,
all nodes at once: 73 vectorised evaluations of F in total, independent of
the number of parameters.
"""
import itertools
import warnings
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy import linalg

from atmospheres import BaseAtmosphere
from magnetic_fields import BaseField
from paths import BasePath
from tracing.coleman_equations import ray_refractive_index
from utilities.Constants import EARTH_RADIUS


class SystemState:
    def __init__(
            self, field: BaseField,
            atmosphere: BaseAtmosphere,
            operating_frequency: float,
            is_extraordinary_ray: bool
    ):
        self.field, self.atmosphere = field, atmosphere
        self.operating_frequency = operating_frequency
        self.is_extraordinary_ray = is_extraordinary_ray


def phase_integrand(
        system_state: SystemState,
        r: np.ndarray,
        r_dot: np.ndarray,
        use_cheater_solver: bool = False
) -> np.ndarray:
    """
    F(r, r') = mu * pt * |r'| for arrays of positions and path derivatives.

    :param system_state: the field, atmosphere, frequency and ray type
    :param r: positions, shape (N, 3)
    :param r_dot: derivatives of the position with respect to the path fraction, shape (N, 3)
    :param use_cheater_solver: if True, assume an isotropic medium (p = t)
    :returns: shape (N,). NaN where the wave can't propagate (mu^2 <= 0).
    """
    r_dot_norm = linalg.norm(r_dot, axis=1)
    t = r_dot / r_dot_norm[:, np.newaxis]

    frequency = system_state.operating_frequency
    x = np.square(np.atleast_1d(system_state.atmosphere.plasma_frequency(r)) / frequency)
    y = np.atleast_1d(system_state.field.gyro_frequency(r)) / frequency
    yt = y * np.einsum('ij,ij->i', np.reshape(system_state.field.field_vec(r), (-1, 3)), t)

    sign = -1 if system_state.is_extraordinary_ray else 1
    return ray_refractive_index(x, y, yt, sign, isotropic_approximation=use_cheater_solver) * r_dot_norm


def quadrature_rule(breakpoints: np.ndarray, order: int = 6):
    """
    Composite Gauss-Legendre rule: `order` nodes in every panel between consecutive breakpoints.
    :returns: (nodes, weights)
    """
    gauss_nodes, gauss_weights = np.polynomial.legendre.leggauss(order)
    breakpoints = np.asarray(breakpoints, dtype=float)
    half_widths = 0.5 * np.diff(breakpoints)
    centers = 0.5 * (breakpoints[1:] + breakpoints[:-1])
    nodes = (centers[:, np.newaxis] + half_widths[:, np.newaxis] * gauss_nodes).ravel()
    weights = (half_widths[:, np.newaxis] * gauss_weights).ravel()
    return nodes, weights


def integrate_parameter(
        system_state: SystemState,
        path: BasePath,
        order: int = 6,
        use_cheater_solver: bool = False
) -> float:
    """The phase path P (in meters) of a path"""
    nodes, weights = quadrature_rule(path.quadrature_breakpoints, order)
    return float(np.sum(weights * phase_integrand(
        system_state, path(nodes), path(nodes, nu=1), use_cheater_solver
    )))


# Offsets (in units of the finite difference step) of the stencil used to get the gradient and
# Hessian of F with respect to its 6 arguments: the centre, +-1 along each axis, and the four
# (+-1, +-1) corners for each of the 15 pairs of axes.
_AXIS_PAIRS = list(itertools.combinations(range(6), 2))


def _build_stencil():
    offsets = [np.zeros(6)]
    for axis in range(6):
        for direction in (1, -1):
            offset = np.zeros(6)
            offset[axis] = direction
            offsets.append(offset)
    for a, b in _AXIS_PAIRS:
        for sign_a, sign_b in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            offset = np.zeros(6)
            offset[a], offset[b] = sign_a, sign_b
            offsets.append(offset)
    return np.array(offsets)


_STENCIL = _build_stencil()


def integrand_derivatives(
        system_state: SystemState,
        r: np.ndarray,
        r_dot: np.ndarray,
        position_step: float = 5.0,
        use_cheater_solver: bool = False
):
    """
    Value, gradient and Hessian of F(r, r') with respect to z = (r, r') at every node.

    :param position_step: finite difference step for the position in meters. It has to be small
        compared to the scale of the ionosphere (tens of km) but large compared to round-off
        (r is ~6.4E6 m, so a few meters is about right).
    :returns: (F of shape (N,), grad F of shape (N, 6), Hess F of shape (N, 6, 6))
    """
    node_number = len(r)
    # F is homogeneous of degree one in r', so the natural step for r' is relative to its size
    derivative_step = 1E-4 * np.median(linalg.norm(r_dot, axis=1))
    steps = np.array([position_step] * 3 + [derivative_step] * 3)

    z = np.concatenate((r, r_dot), axis=1)
    z_all = z[np.newaxis, :, :] + (_STENCIL * steps)[:, np.newaxis, :]
    f_all = phase_integrand(
        system_state,
        z_all[:, :, :3].reshape(-1, 3), z_all[:, :, 3:].reshape(-1, 3),
        use_cheater_solver
    ).reshape(len(_STENCIL), node_number)

    f_0 = f_all[0]
    f_plus, f_minus = f_all[1:13:2], f_all[2:13:2]
    gradient = ((f_plus - f_minus) / (2 * steps[:, np.newaxis])).T

    hessian = np.empty((node_number, 6, 6))
    diagonal = (f_plus - 2 * f_0 + f_minus) / np.square(steps)[:, np.newaxis]
    hessian[:, np.arange(6), np.arange(6)] = diagonal.T
    corners = f_all[13:].reshape(len(_AXIS_PAIRS), 4, node_number)
    for (a, b), (f_pp, f_pm, f_mp, f_mm) in zip(_AXIS_PAIRS, corners):
        mixed = (f_pp - f_pm - f_mp + f_mm) / (4 * steps[a] * steps[b])
        hessian[:, a, b] = mixed
        hessian[:, b, a] = mixed
    return f_0, gradient, hessian


class Tracer:
    def __init__(
            self,
            operating_frequency: float,
            atmosphere_model: BaseAtmosphere,
            magnetic_field: BaseField,
            path_initializer: BasePath,
            cores: Optional[int] = None
    ):
        """
        :param operating_frequency: wave frequency in Hz
        :param atmosphere_model: the ionosphere
        :param magnetic_field: the geomagnetic field
        :param path_initializer: the path to start iterating from
        :param cores: unused. Kept so that old scripts still run; the calculation no longer needs
            a process pool.
        """
        if None in (operating_frequency, atmosphere_model, magnetic_field, path_initializer):
            raise ValueError("Model initializer parameters cannot be Null")

        self.field, self.atmosphere, = magnetic_field, atmosphere_model
        self.calculated_paths = [path_initializer]
        self.frequency = operating_frequency
        self.cores = cores

        # Filled in by trace()
        self.converged = False
        self.gradient_norms = []
        self.phase_path = None

    def replace_path(self, new_path: BasePath):
        self.calculated_paths = [new_path]
        self.converged = False

    def get_system_state(self, is_extraordinary_ray: bool):
        return SystemState(self.field, self.atmosphere, self.frequency, is_extraordinary_ray)

    def calculate_derivatives(
            self, path: Optional[BasePath] = None,
            is_extraordinary_ray=False, use_cheater_solver=False,
            h=5.0, quadrature_order=6, need_hessian=True
    ):
        """
        Calculate the phase path of a path and its derivatives with respect to the path parameters.

        :returns: (hessian matrix d2P/da_i da_j or None, gradient vector dP/da_i, phase path P)
        """
        if path is None:
            path = self.calculated_paths[-1]
        system_state = self.get_system_state(is_extraordinary_ray)
        nodes, weights = quadrature_rule(path.quadrature_breakpoints, quadrature_order)
        r, r_dot = path(nodes), path(nodes, nu=1)
        d_position, d_derivative = path.parameter_jacobian(nodes)
        # dz/da, shape (nodes, 6, parameters)
        jacobian = np.concatenate((d_position, d_derivative), axis=1)

        if need_hessian:
            f, f_gradient, f_hessian = integrand_derivatives(
                system_state, r, r_dot, position_step=h, use_cheater_solver=use_cheater_solver
            )
        else:
            f, f_gradient = self._integrand_gradient(system_state, r, r_dot, h, use_cheater_solver)
            f_hessian = None

        phase_path = float(np.sum(weights * f))
        gradient = np.einsum('q,qk,qki->i', weights, f_gradient, jacobian)
        matrix = None
        if need_hessian:
            weighted = np.einsum('q,qkl,qlj->qkj', weights, f_hessian, jacobian)
            matrix = np.einsum('qki,qkj->ij', jacobian, weighted)
            matrix = 0.5 * (matrix + matrix.T)
        return matrix, gradient, phase_path

    @staticmethod
    def _integrand_gradient(system_state, r, r_dot, position_step, use_cheater_solver):
        """Same as integrand_derivatives but only the 13 evaluations needed for the gradient"""
        node_number = len(r)
        derivative_step = 1E-4 * np.median(linalg.norm(r_dot, axis=1))
        steps = np.array([position_step] * 3 + [derivative_step] * 3)
        z = np.concatenate((r, r_dot), axis=1)
        z_all = z[np.newaxis, :, :] + (_STENCIL[:13] * steps)[:, np.newaxis, :]
        f_all = phase_integrand(
            system_state, z_all[:, :, :3].reshape(-1, 3), z_all[:, :, 3:].reshape(-1, 3), use_cheater_solver
        ).reshape(13, node_number)
        return f_all[0], ((f_all[1:13:2] - f_all[2:13:2]) / (2 * steps[:, np.newaxis])).T

    def newton_raphson_step(
            self, h=5.0, is_extraordinary_ray=False, use_cheater_solver=False,
            max_change: float = 30E3, quadrature_order=6
    ):
        """
        Perform one damped Newton-Raphson step and append the new path to self.calculated_paths.

        The full Newton step is -H^-1 g. It is scaled down if any parameter would move by more
        than max_change meters, and halved until the gradient norm decreases. (P itself can't be
        used to judge a step because rays are stationary points of P, not necessarily minima:
        high rays are saddle points.)

        :returns: (hessian, gradient, change applied to the parameters, full Newton change)
        """
        path = self.calculated_paths[-1]
        options = dict(
            is_extraordinary_ray=is_extraordinary_ray, use_cheater_solver=use_cheater_solver,
            h=h, quadrature_order=quadrature_order
        )
        matrix, gradient, _ = self.calculate_derivatives(path, **options)
        if not np.all(np.isfinite(gradient)) or not np.all(np.isfinite(matrix)):
            raise RuntimeError(
                "The path passes through a region where this ray type can't propagate "
                "(refractive index squared <= 0), so the phase path isn't defined. "
                "Choose a different starting path."
            )
        try:
            change = -linalg.solve(matrix, gradient, assume_a='sym')
        except (linalg.LinAlgError, linalg.LinAlgWarning):
            warnings.warn(
                "Using pseudo-inverse to solve matrix equation because matrix is near-singular.\n"
                "Consider using less parameters, or having no lateral parameters for this system"
            )
            change = -np.matmul(linalg.pinvh(matrix), gradient)

        scale = min(1.0, max_change / np.max(np.abs(change)))
        gradient_norm = linalg.norm(gradient)
        indexes = np.arange(change.shape[0])
        next_path = None
        for _ in range(12):
            candidate = path.adjust_parameters(indexes, scale * change)
            _, candidate_gradient, _ = self.calculate_derivatives(candidate, need_hessian=False, **options)
            if np.all(np.isfinite(candidate_gradient)):
                next_path = candidate
                if linalg.norm(candidate_gradient) < gradient_norm:
                    break
            scale *= 0.5
        if next_path is None:
            raise RuntimeError("Newton-Raphson step failed: every trial step left the propagating region")

        self.calculated_paths.append(next_path)
        return matrix, gradient, scale * change, change

    def trace(
            self, h=5.0,
            is_extraordinary_ray=False,
            use_cheater_solver: Optional[bool] = False,
            max_steps: int = 50,
            tolerance: float = 1E-3,
            verbose: bool = True
    ):
        """
        Iterate from the last calculated path until the phase path is stationary.

        :param h: finite difference step (meters) used for the derivatives of the integrand
        :param is_extraordinary_ray: trace the extraordinary ray instead of the ordinary ray
        :param use_cheater_solver: assume an isotropic medium (p = t) instead of solving for Yp and pt
        :param max_steps: maximum number of Newton-Raphson steps
        :param tolerance: stop when no parameter changes by more than this many meters in a step
        :param verbose: print progress
        :returns: the list of paths calculated so far. The last one is the solution.
            self.converged tells you whether the tolerance was reached.
        """
        options = dict(is_extraordinary_ray=is_extraordinary_ray, use_cheater_solver=use_cheater_solver, h=h)
        self.converged = False
        self.gradient_norms = []
        for i in range(1, max_steps + 1):
            _, gradient, change, newton_change = self.newton_raphson_step(**options)
            # Judge convergence on the full Newton step: a step that was only small because
            # the line search shrank it doesn't mean that we have arrived
            max_change = np.max(np.abs(newton_change))
            self.gradient_norms.append(linalg.norm(gradient))
            if verbose:
                print(f"Newton Raphson step {i}: |gradient| = {self.gradient_norms[-1]:.3e}, "
                      f"Newton step = {max_change:.3e} m, applied {np.max(np.abs(change)):.3e} m")
            if max_change < tolerance:
                self.converged = True
                break

        _, gradient, self.phase_path = self.calculate_derivatives(need_hessian=False, **options)
        self.gradient_norms.append(linalg.norm(gradient))
        if not self.converged:
            warnings.warn(
                f"Path did not converge in {max_steps} steps. "
                "It is likely that there is no ray of this type near the starting path"
            )
        elif verbose:
            print(f"Converged. Phase path = {self.phase_path / 1000:.6f} km, "
                  f"|gradient| = {self.gradient_norms[-1]:.3e}")
        return self.calculated_paths

    def visualize(self, show_history=False, show=True, fig=None, ax=None, color='black'):
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.set_title(f"3D Ray Trace with a {int(self.frequency / 1E6)} MHz frequency")
            self.atmosphere.visualize(
                self.calculated_paths[0](0),
                self.calculated_paths[0](1),
                fig=fig,
                ax=ax,
            )
            ax.autoscale(False)
            ax.set_ylabel("Altitude (km)")
            ax.set_xlabel("Range (km)")
            if show_history and len(self.calculated_paths) > 1:
                custom_lines = [Line2D([0], [0], color='black', lw=4),
                                Line2D([0], [0], color='white', lw=4)]
                ax.legend(custom_lines, ['Best Trace', 'Earlier Traces'])
            else:
                custom_lines = [Line2D([0], [0], color='black', lw=4)]
                ax.legend(custom_lines, ['Best Trace'])

        evaluations = np.linspace(0, 1, 200)
        if show_history:
            for i in range(len(self.calculated_paths) - 1):
                path = self.calculated_paths[i]
                points = path(evaluations)
                radii = (linalg.norm(points, axis=-1) - EARTH_RADIUS) / 1000
                km_range = evaluations * path.total_angle * EARTH_RADIUS / 1000
                ax.plot(km_range, radii, color='white')

        # We always plot the current ones
        path = self.calculated_paths[-1]
        points = path(evaluations)
        radii = (linalg.norm(points, axis=-1) - EARTH_RADIUS) / 1000
        km_range = evaluations * path.total_angle * EARTH_RADIUS / 1000
        ax.plot(km_range, radii, color=color)

        if show:
            plt.show()
            plt.close(fig)
        else:
            return fig, ax

    def cleanup(self):
        """Kept for backwards compatibility. There is no process pool to shut down any more."""
        pass
