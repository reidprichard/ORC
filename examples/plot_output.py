from matplotlib import pyplot as plt
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
    ax.quiver(x, y, u, v)
    plt.show()


if __name__ == "__main__":
    main()
