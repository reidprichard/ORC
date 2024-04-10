import argparse
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.figure import Figure
import numpy as np
import re
import win32api


def tile_all_matplotlib_figures() -> None:
    """Moves all active matplotlib figures into a rectangular grid.
    If only one figure exists, it will be moved to the center of the screen.
    Must be called after figure creation and before calling plt.show(). Only
    works in Windows, but should not be hard to make it work on other OS's.
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
    if len(plt.get_fignums()) > 0:
        plt.show()


def interpolate_to_grid(x_list, y_list, z_list, n_x=200, n_y=200):
    x_linear = np.linspace(min(x_list), max(x_list), n_x)
    y_linear = np.linspace(min(y_list), max(y_list), n_y)
    triang = tri.Triangulation(x_list, y_list)
    x_grid, y_grid = np.meshgrid(x_linear, y_linear)

    z_interpolator = tri.LinearTriInterpolator(triang, z_list)
    z_grid = z_interpolator(x_grid, y_grid)

    return (x_grid, y_grid, z_grid)


def plot_2d(root: str, plot_title: str | None = None, save:bool=False):
    MU = 0.1
    DP = -10
    DX = 0.002
    CHANNEL_HEIGHT = 0.001

    FLOAT = "[\\d|\\.|e|\\-]+"
    VECTOR = f"\\(({FLOAT}),\\s+({FLOAT}),\\s+({FLOAT})\\)"
    datafile_pattern = re.compile(f"{VECTOR}\\t{VECTOR}\\t({FLOAT})")
    row_values = []
    with open(f"./examples/{root}.csv") as simulation_data:
        lines = simulation_data.readlines()
        for line in lines:
            match = datafile_pattern.match(line)
            if match:
                row_values.append([float(n) for n in match.groups()])
    x, y, _, u, v, _, p = list(zip(*row_values))

    x2 = []
    y2 = []
    velocity_gradient = []
    pressure_gradient = []
    with open(f"./examples/{root}_gradients.csv") as simulation_data:
        for line in simulation_data.readlines():
            centroid, vel_grad, p_grad = [s.split(", ") for s in line.replace("(", "").replace(")", "").split("\t")]
            x2.append(float(centroid[0]))
            y2.append(float(centroid[1]))
            # There should only be 9 elements, but sometimes a trailing comma adds a 10th empty one
            velocity_gradient.append(np.reshape(np.array(vel_grad[:9]), (3, 3)))
            # There should only be 3 elements, but sometimes a trailing comma adds a 4th empty one
            pressure_gradient.append(np.array(p_grad[:3]))
    velocity_gradient = np.array(velocity_gradient)  # type: ignore[assignment]
    pressure_gradient = np.array(pressure_gradient)  # type: ignore[assignment]

    # *** Figure 1 ***
    fig, axs = plt.subplots(nrows=2, layout="constrained", sharex=True, sharey=True)
    if plot_title is not None:
        fig.suptitle(plot_title)
    x_interpolated, y_interpolated, p_interpolated = interpolate_to_grid(x, y, p)
    cm = axs[0].contourf(x_interpolated, y_interpolated, p_interpolated, levels=10)
    fig.colorbar(cm, label="Gage Pressure [Pa]")
    axs[0].quiver(x, y, u, v)
    axs[0].set_title("Velocity Vectors; Pressure Contours")
    axs[0].set_xlabel("X [m]")
    axs[0].set_ylabel("Y [m]")

    x2_interpolated, y2_interpolated, du_dy = interpolate_to_grid(x2, y2, velocity_gradient[:, 0, 1])  # type: ignore[call-overload]
    norm = matplotlib.colors.TwoSlopeNorm(0)
    cm = axs[1].contourf(x2_interpolated, y2_interpolated, du_dy, cmap="RdBu", levels=20, norm=norm)
    # dp_dx = interpolate_to_grid(x2, y2, pressure_gradient[:,0])[-1]
    # dp_dy = interpolate_to_grid(x2, y2, pressure_gradient[:,1])[-1]
    # axs[1].quiver(x2_interpolated, y2_interpolated, dp_dx, dp_dy)
    axs[1].set_title(r"$\frac{du}{dy}$")
    axs[1].set_xlabel("X [m]")
    axs[1].set_ylabel("Y [m]")
    axs[1].ticklabel_format(style="scientific", scilimits=(0, 0))

    fig.colorbar(cm, label="Velocity gradient [1/s]")
    if save:
        fig.savefig("./examples/channel_flow_contour_plots.png", dpi=300)

    # *** Figure 2 ***
    fig, ax = plt.subplots()
    if plot_title is not None:
        fig.suptitle(plot_title)
    a = CHANNEL_HEIGHT
    y_linear = np.linspace(np.min(y), np.max(y), 100)
    u_analytical = 1 / (2 * MU) * DP / DX * (y_linear**2 - a * y_linear)
    ax.plot(y_linear, u_analytical)
    ax.scatter(y, u)
    ax.set_xlabel("Y [m]")
    ax.set_ylabel("U [m/s]")
    if save:
        fig.savefig("./examples/channel_flow_velocity_profile.png", dpi=300)


def plot_face_velocities(filenames: list[str], save:bool=False):
    fig, axs = plt.subplots(nrows=len(filenames), layout="constrained", sharex=True,sharey=True)
    if len(filenames)==1:
        axs = [axs]
    x:list[list[float]] = []
    y:list[list[float]] = []
    u:list[list[float]] = []
    v:list[list[float]] = []
    for fname in filenames:
        x.append([])
        y.append([])
        u.append([])
        v.append([])
        with open(fname, "r") as f:
            lines = f.readlines()
            for line in lines:
                if len(line.strip())==0:
                    continue
                _, centroid, vel = line.split("\t")
                x_str, y_str = centroid[1:-1].split(",")[:-1]
                x[-1].append(float(x_str.strip()))
                y[-1].append(float(y_str.strip()))
                u_str, v_str = vel[1:-1].split(",")[:-1]
                u[-1].append(float(u_str.strip()))
                v[-1].append(float(v_str.strip()))

    nested_max = lambda l : max([max(i) for i in l])
    nested_min = lambda l : min([min(i) for i in l])
    
    u_min, u_max = nested_min(u), nested_max(u)
    v_min, v_max = nested_min(v), nested_max(v)

    arrow_scale = (u_max**2 + v_max**2)**(1/2) * 30

    for (ax, xi, yi, ui, vi) in zip(axs, x, y, u, v):
        cm = ax.contourf(*interpolate_to_grid(xi,yi,ui), levels=np.linspace(u_min, u_max, 10))
        ax.quiver(xi, yi, ui, vi, scale=arrow_scale, scale_units="width", width=0.002)
        ax.set_title(fname.split('/')[-1].split('\\')[-1])
    fig.colorbar(cm)
    if save:
        fig.savefig("./examples/face_velocities.png", dpi=300)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="CfdPlotter", description="Plots solution fields of CFD data")
    parser.add_argument("data_file_base", default=None, nargs="?")
    parser.add_argument("-t", "--title", default=None)
    parser.add_argument("--face-velocity-files", "-f", default=None, nargs="+")
    parser.add_argument("--save", action="store_true", default=False)
    args = parser.parse_args()
    if args.data_file_base is not None:
        plot_2d(args.data_file_base, args.title, args.save)
    if args.face_velocity_files is not None:
        plot_face_velocities(args.face_velocity_files, args.save)
    tile_all_matplotlib_figures()
