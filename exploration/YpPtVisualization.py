import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button, CheckButtons

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracing.coleman_equations import equation_13, equation_14, equation_15


def equation_13_14_15(x, y, y_dot_p_normalized, y_dot_t_normalized, sign):
    x, y_squared, yt = x, np.square(y), y_dot_t_normalized * y
    yp = y_dot_p_normalized * y
    result_13 = equation_13(yp, x, y_squared, yt, sign)
    result_14 = equation_14(yp, np.square(yp), y_squared, x, yt, sign)
    result_15 = equation_15(yp, x, y_squared, sign)
    return result_13, result_14, result_15


y_dot_p_normalized_external = np.linspace(-1, 1, 1000)

# Define initial parameters
init_y = 0.9
init_yt_normalized = 0.8
init_x = 0.2
main_sign = 1

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
yp_minimizer, pt, mu2 = equation_13_14_15(init_x, init_y, y_dot_p_normalized_external, init_yt_normalized, main_sign)
line_yp, = plt.plot(
    y_dot_p_normalized_external,
    yp_minimizer,
    color="red",
    lw=2, label="Minimizer"
)
line_pt, = plt.plot(
    y_dot_p_normalized_external,
    pt,
    color="green",
    lw=2, label="pt"
)
line_mu2, = plt.plot(
    y_dot_p_normalized_external,
    mu2,
    color="gold",
    lw=2, label="mu^2"
)

plt.hlines([0, 1, -1], -1, 1, color='blue', label='y = 1,0,-1')

ax.set_xlabel('y_dot_p_normalized')
ax.set_ylim([-1.5, 1.5])
ax.set_xlim([-1, 1])
ax.legend()
ax.autoscale(False)

slider_color = 'lightgoldenrodyellow'
ax.margins(x=0)

# adjust the main plot to make room for the sliders
plt.subplots_adjust(left=0.25, bottom=0.35)
plt.title("YP and PT Visualizer")

ax_yt = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=slider_color)
yt_slider = Slider(
    ax=ax_yt,
    label='yt_norm',
    valmin=-1,
    valmax=1,
    valinit=init_yt_normalized,
)

ax_y = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=slider_color)
y_slider = Slider(
    ax=ax_y,
    label="y",
    valmin=0,
    valmax=10,
    valinit=init_y,
)

ax_x = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=slider_color)
x_slider = Slider(
    ax=ax_x,
    label="x",
    valmin=0,
    valmax=1,
    valinit=init_x,
)


# The function to be called anytime a slider's value changes
def update(_val):
    yp_minimizer_internal, pt_internal, mu2_internal = equation_13_14_15(
        x_slider.val,
        y_slider.val,
        y_dot_p_normalized_external,
        yt_slider.val,
        main_sign
    )
    line_yp.set_ydata(
        yp_minimizer_internal
    )
    line_pt.set_ydata(
        pt_internal
    )
    line_mu2.set_ydata(
        mu2_internal
    )
    fig.canvas.draw_idle()


def reset(_event):
    yt_slider.reset()
    y_slider.reset()
    x_slider.reset()


def toggle_ray_type(_label):
    global main_sign
    main_sign = -1 * main_sign
    update(None)


# register the update function with each slider
yt_slider.on_changed(update)
y_slider.on_changed(update)
x_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
reset_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
reset_button = Button(reset_ax, 'Reset', color=slider_color, hovercolor='0.975')
toggle_ax = plt.axes([0.55, 0.025, 0.2, 0.04])
ray_button = CheckButtons(toggle_ax, ['Extraordinary'], [False])

reset_button.on_clicked(reset)
ray_button.on_clicked(toggle_ray_type)
plt.show()
