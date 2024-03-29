import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.figure import Figure
import numpy as np
import re
from typing import Literal
import win32api

def tile_all_matplotlib_figures(
    resize_figs: bool = False, show: bool = True, show_mode: Literal["basic", "enhanced"] = "enhanced"
) -> None:
    """Moves all active matplotlib figures into a rectangular grid.
    If only one figure exists, it will be moved to the center of the screen.
    Must be called after figure creation and before calling plt.show(). Only
    works in Windows, but should not be hard to make it work on other OS's.

    Parameters
    ----------
    1. resize_figs : bool, (default False)
            - If True, figures will be both moved and resized so that all
              figures fit display area.
    2. show : bool (default True)
            - If True, figures will be shown.
    3. show_mode : Literal["basic", "enhanced"] (default "basic")
            - If "enhanced", plots will be shown in a manner where pressing
              `enter` in the console closes them all. If "basic", `plt.show()`
              will simply be called.
    """

    def move_figure(
        f: Figure,
        xy: tuple[int, int] | None = None,
        size: tuple[int, int] | None = None,
    ) -> None:
        """Move figure's upper left corner to pixel (x, y). Code from
        stackexchange with modifications."""
        if xy is None and size is None:
            print("Error: Must specify `xy` or `size`.")
            return
        elif xy is not None:
            if len(xy) != 2 or type(xy[0]) is not int or type(xy[1]) is not int:
                raise ValueError("`xy` values must be a tuple of two integers.")
        elif size is not None:
            if len(size) != 2 or type(size[0]) is not int or type(size[1]) is not int:
                raise ValueError("`size` values must be a tuple of two integers.")

        backend = matplotlib.get_backend()
        geometry: str | list[int | None] = ""
        if backend == "TkAgg":
            geometry = ""
            if size is not None:
                geometry += f"{size[0]}x{size[1]}"
            if xy is not None:
                geometry += f"+{xy[0]}+{xy[1]}"
            f.canvas.manager.window.wm_geometry(geometry)  # type: ignore[union-attr]
        elif backend == "WXAgg":
            if size is not None:
                print("Can't set width and height")  # Probably possible; just don't know how
            if xy is not None:
                f.canvas.manager.window.SetPosition((xy[0], xy[1]))  # type: ignore[union-attr]
        else:
            geometry = []
            if xy is not None:
                geometry += xy
            else:
                geometry += [None, None]
            if size is not None:
                geometry += size
            else:
                geometry += [None, None]
            # This works for QT and GTK You can also use window.setGeometry
            f.canvas.manager.window.setGeometry(*geometry)  # type: ignore[union-attr]
        return

    # NOTE: This code is the reason it's Windows-exclusive. To make work on
    # multiple OS's, would need to check what the OS is; if Windows do the
    # below; and if another OS use another method to find display width and
    # height

    monitors = win32api.EnumDisplayMonitors()
    monitor_coords = []

    for monitor in monitors:
        coords = monitor[2]  # [x_start, y_start, x_end, y_end]
        monitor_coords.append(coords)

    # primary_display_width = user32.GetSystemMetrics(61) primary_display_height
    # = user32.GetSystemMetrics(62)

    VERTICAL_PADDING, HORIZONTAL_PADDING = 10, 10
    VERTICAL_FUDGE_FACTOR = 70

    fignums = plt.get_fignums()
    fig_count = len(fignums)

    if fig_count == 0:
        print(f"{tile_all_matplotlib_figures.__name__}: No figures exist.")
        # return
    elif fig_count == 1:
        fig = plt.figure(1)
        fig_bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig_width, fig_height = int(fig_bbox.width * fig.dpi), int(fig_bbox.height * fig.dpi)
        fig_x = int((monitor_coords[0][2] - fig_width) / 2)
        fig_y = int((monitor_coords[0][3] - fig_height) / 2)
        move_figure(fig, xy=(fig_x, fig_y))
        # return
    elif resize_figs:
        print("WARNING: `resize_figs` is not working as intended.")
        col_count = math.floor(math.sqrt(monitor_coords[0][2] / monitor_coords[0][3]) * math.ceil(math.sqrt(fig_count)))

        row_count = math.ceil(fig_count / col_count)
        print(f"{col_count=},{row_count=}")

        fig_width = int((monitor_coords[0][2] - (col_count + 1) * HORIZONTAL_PADDING) / col_count)
        fig_height = int(
            (monitor_coords[0][3] - (row_count + 1) * VERTICAL_PADDING) / row_count
        )  # TODO: Make pairs (height/width) lists

        for n in fignums:
            i = (n - 1) % col_count
            j = int((n - 1) / col_count)
            x_start = HORIZONTAL_PADDING * (i + 1) + i * fig_width
            y_start = VERTICAL_PADDING * (j + 1) + j * fig_height
            print(f"{x_start}-{x_start+fig_width}, {y_start}-{y_start+fig_height}")
            move_figure(plt.figure(n), (x_start, y_start), (fig_width, fig_height))
    else:
        window_x = HORIZONTAL_PADDING
        window_y = VERTICAL_PADDING
        monitor_index = 0
        for n in fignums:
            fig = plt.figure(n)
            fig_bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig_width, fig_height = int(fig_bbox.width * fig.dpi), int(fig_bbox.height * fig.dpi)
            if window_x + fig_width > monitor_coords[monitor_index][2]:
                window_x = monitor_coords[monitor_index][0] + HORIZONTAL_PADDING
                window_y += fig_height + VERTICAL_PADDING + VERTICAL_FUDGE_FACTOR
                if window_y + fig_height > monitor_coords[monitor_index][3]:
                    monitor_index += 1
                    if monitor_index >= len(monitors):
                        print("Too many figures for display area.")
                        break
                    else:
                        window_x = monitor_coords[monitor_index][0] + HORIZONTAL_PADDING
                        window_y = monitor_coords[monitor_index][1] + VERTICAL_PADDING
            move_figure(fig, xy=(window_x, window_y))
            window_x += fig_width + HORIZONTAL_PADDING
    if show:
        if len(plt.get_fignums()) > 0:
            if show_mode == "enhanced":
                plt.show(block=False)
                input("Press [enter] to close all plots.")
                plt.close("all")
            else:
                plt.show()


def plot_couette():
    MU = 0.01

    FLOAT = "[\\d|\\.|e|\\-]+"
    VECTOR = f"\\(({FLOAT}),\\s+({FLOAT}),\\s+({FLOAT})\\)"
    datafile_pattern = re.compile(f"{VECTOR},\\s+{VECTOR},\\s+({FLOAT})")
    row_values = []
    with open("./examples/couette.csv") as simulation_data:
        lines = simulation_data.readlines()
        for line in lines:
            match = datafile_pattern.match(line)
            if match:
                row_values.append([float(n) for n in match.groups()])
    x, y, _, u, v, _, p = list(zip(*row_values))

    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    triang = tri.Triangulation(x, y)
    Xi, Yi = np.meshgrid(xi, yi)

    # *** Figure 1 ***
    fig, axs = plt.subplots(nrows=2, layout="constrained", sharex=True, sharey=True)
    p_interpolator = tri.LinearTriInterpolator(triang, p)
    p_interpolated = p_interpolator(Xi, Yi)
    cm = axs[0].contourf(xi, yi, p_interpolated, levels=10)
    fig.colorbar(cm, label="Gage Pressure [Pa]")
    axs[0].quiver(x, y, u, v)
    axs[0].set_title("Velocity vectors; pressure contours")
    axs[0].set_xlabel("X [m]")
    axs[0].set_ylabel("Y [m]")

    u_interpolator = tri.LinearTriInterpolator(triang, u)
    u_interpolated = u_interpolator(Xi, Yi)
    du_dy = np.gradient(u_interpolated, axis=0)
    cm = axs[1].contourf(xi, yi, du_dy, cmap="RdBu", levels=50)
    axs[1].set_title(r"$\frac{du}{dy}$")
    axs[1].set_xlabel("X [m]")
    axs[1].set_ylabel("Y [m]")
    # p_grad = np.gradient(p_interpolated, xi, yi)
    # axs[1].quiver(xi, yi, p_grad[1]*0, p_grad[0])

    fig.colorbar(cm, label="du/dy [1/s]")
    fig.savefig("./examples/couette.png", dpi=300)

    # *** Figure 2 ***
    fig, ax = plt.subplots()
    fig.suptitle("Velocity profile vs analytical")
    dp_dx = (np.min(p) - np.max(p)) / (np.max(x) - np.min(x))
    a = np.max(y) - np.min(y)
    u_analytical = 1 / (2 * MU) * dp_dx * (yi**2 - a * yi)
    ax.plot(yi, u_analytical)
    ax.scatter(y, u)

    tile_all_matplotlib_figures()


if __name__ == "__main__":
    plot_couette()
