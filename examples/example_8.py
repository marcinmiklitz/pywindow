"""Example 8."""

import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import pywindow as pw


def main() -> None:
    """Run script."""
    raise SystemExit("add asserts compared to all notebook output.")
    traj = pw.trajectory.DLPOLY("./data/input/HISTORY_periodic")
    logging.info(traj.no_of_frames)

    traj.analysis(
        forcefield="opls",
        swap_atoms={"he": "H"},
        ncpus=8,
        rebuild=True,
        modular=True,
    )

    logging.info(traj.analysis_output)

    windows = []
    pore_diam_opt = []
    max_diam = []

    for key in traj.analysis_output:
        for mol in traj.analysis_output[key]:
            for i in traj.analysis_output[key][mol]["windows"]["diameters"]:
                windows.append(i)
            pore_diam_opt.append(
                traj.analysis_output[key][mol]["pore_diameter_opt"]["diameter"]
            )
            max_diam.append(
                traj.analysis_output[key][mol]["maximum_diameter"]["diameter"]
            )

    x_range_windows = np.linspace(min(windows) - 1, max(windows) + 1, 1000)

    kde_windows = stats.gaussian_kde(windows)
    dist_windows = kde_windows(x_range_windows)

    x_range_pore = np.linspace(
        min(pore_diam_opt) - 1, max(pore_diam_opt) + 1, 1000
    )

    kde_pore = stats.gaussian_kde(pore_diam_opt)
    dist_pore = kde_pore(x_range_pore)

    x_range_max = np.linspace(min(max_diam) - 1, max(max_diam) + 1, 1000)

    kde_max = stats.gaussian_kde(max_diam)
    dist_max = kde_max(x_range_max)

    fig, ax = plt.subplots(figsize=(7, 2.5))

    plt.plot(
        x_range_windows, dist_windows, label="windows diameter", linewidth=2
    )

    ax.axes.get_yaxis().set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("k")
    ax.spines["left"].set_visible(False)
    ax.grid(b=False)
    ax.set_facecolor("white")
    ax.tick_params(axis="both", which="major", labelsize=12, top="off")
    [t.set_color("k") for t in ax.xaxis.get_ticklabels()]
    [
        text.set_color("k")
        for text in ax.legend(frameon=False, fontsize=10, loc=2).get_texts()
    ]
    ax.set_xlabel(r"Diameter ($\mathregular{\AA)}$", fontsize=12)

    plt.tight_layout()
    # plt.savefig("trajectory_windows.pdf", dpi=300)
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 2.5))

    plt.plot(
        x_range_pore,
        dist_pore,
        label="pore diameter",
        linewidth=2,
        color="green",
    )

    ax.axes.get_yaxis().set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("k")
    ax.spines["left"].set_visible(False)
    ax.grid(b=False)
    ax.tick_params(axis="both", which="major", labelsize=12, top="off")
    [t.set_color("k") for t in ax.xaxis.get_ticklabels()]
    [
        text.set_color("k")
        for text in ax.legend(frameon=False, fontsize=10, loc=2).get_texts()
    ]
    ax.set_xlabel(r"Diameter ($\mathregular{\AA)}$", fontsize=12)

    plt.tight_layout()
    # plt.savefig("trajectory_pores.pdf", dpi=300)
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 2.5))

    plt.plot(
        x_range_max,
        dist_max,
        label="maximum dimension",
        linewidth=2,
        color="darkorange",
    )

    ax.axes.get_yaxis().set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("k")
    ax.spines["left"].set_visible(False)
    ax.grid(b=False)
    ax.set_facecolor("white")
    ax.tick_params(axis="both", which="major", labelsize=12, top="off")
    [t.set_color("k") for t in ax.xaxis.get_ticklabels()]
    [
        text.set_color("k")
        for text in ax.legend(frameon=False, fontsize=10, loc=2).get_texts()
    ]
    ax.set_xlabel(r"Diameter ($\mathregular{\AA)}$", fontsize=12)

    plt.tight_layout()
    # plt.savefig("trajectory_maxdim.pdf", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
