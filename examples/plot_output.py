from matplotlib import pyplot as plt
import matplotlib.tri as tri
import numpy as np
import re


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

    fig.colorbar(cm, label="du/dy [1/s]")
    fig.savefig("./examples/couette.png", dpi=300)

    # *** Figure 2 ***
    fig, ax = plt.subplots()
    fig.suptitle("Velocity profile vs analytical")
    dp_dx = (np.min(p) - np.max(p))/(np.max(x) - np.min(x))
    a = np.max(y) - np.min(y)
    u_analytical = 1/(2*MU) * dp_dx * (yi**2 - a * yi)
    ax.plot(yi, u_analytical)
    ax.scatter(y, u)

    plt.show()


if __name__ == "__main__":
    plot_couette()
