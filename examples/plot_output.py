from matplotlib import pyplot as plt
import matplotlib.tri as tri
import numpy as np
import re


def main():
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
    x, y, z, u, v, w, p = list(zip(*row_values))

    fig, axs = plt.subplots(nrows=2, layout="constrained", sharex=True, sharey=True)

    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    triang = tri.Triangulation(x, y)
    Xi, Yi = np.meshgrid(xi, yi)

    p_interpolator = tri.LinearTriInterpolator(triang, p)
    p_interpolated = p_interpolator(Xi, Yi)

    u_interpolator = tri.LinearTriInterpolator(triang, u)
    u_interpolated = u_interpolator(Xi, Yi)

    du_dy = np.gradient(u_interpolated, axis=0)

    cm = axs[0].contourf(xi, yi, p_interpolated, levels=10)
    fig.colorbar(cm, label="Gage Pressure [Pa]")

    cm = axs[1].contourf(xi, yi, du_dy, cmap="inferno", levels=50)
    fig.colorbar(cm, label="du/dy [1/s]")

    axs[0].quiver(x, y, u, v)

    axs[0].set_title("Velocity vectors; pressure contours")
    axs[1].set_title(r"$\frac{du}{dy}$")

    axs[0].set_xlabel("X [m]")
    axs[1].set_xlabel("X [m]")
    axs[0].set_ylabel("Y [m]")
    axs[1].set_ylabel("Y [m]")
    fig.savefig("./examples/couette.png", dpi=300)
    # fig.text(0.04, 0.5, "Y [m]", ha="center", rotation="vertical")

    plt.show()


if __name__ == "__main__":
    main()
