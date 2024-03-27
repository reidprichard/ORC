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

    fig, ax = plt.subplots()

    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, p)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)
    cm = ax.contourf(xi,yi,zi)

    fig.colorbar(cm, label="Gage Pressure [Pa]")

    ax.quiver(x, y, u, v)
    plt.show()


if __name__ == "__main__":
    main()
