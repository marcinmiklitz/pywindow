"""Example 8."""

import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import pywindow as pw

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run script."""
    script_directory = pathlib.Path(__file__).parent.resolve()

    data_directory = script_directory / "data"
    input_directory = data_directory / "input"
    output_directory = data_directory / "output"

    traj = pw.DLPOLY(input_directory / "HISTORY_periodic")

    logger.info("there are %s frames", traj.no_of_frames)

    frame_0 = traj.get_frames(0)[0]
    frame_0.swap_atom_keys({"he": "H"})
    frame_0.decipher_atom_keys("opls")

    traj.analysis(
        forcefield="opls",
        swap_atoms={"he": "H"},
        ncpus=4,
        rebuild=True,
        modular=True,
    )

    traj.save_analysis(
        output_directory / "HISTORY_periodic_out.json",
        override=True,
    )

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

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        x_range_windows,
        dist_windows,
        label="windows diameter",
        linewidth=2,
    )
    ax.tick_params(axis="both", which="major", labelsize=12, top="off")
    ax.set_xlabel(r"Diameter ($\mathregular{\AA)}$", fontsize=12)
    fig.tight_layout()
    fig.savefig(
        output_directory / "8_trajectory_windows.pdf",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close("all")

    fig, ax = plt.subplots(figsize=(8, 5))
    plt.plot(
        x_range_pore,
        dist_pore,
        label="pore diameter",
        linewidth=2,
        color="green",
    )
    ax.tick_params(axis="both", which="major", labelsize=12, top="off")
    ax.set_xlabel(r"Diameter ($\mathregular{\AA)}$", fontsize=12)
    fig.tight_layout()
    fig.savefig(
        output_directory / "8_trajectory_pores.pdf",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close("all")

    fig, ax = plt.subplots(figsize=(8, 5))
    plt.plot(
        x_range_max,
        dist_max,
        label="maximum dimension",
        linewidth=2,
        color="darkorange",
    )
    ax.tick_params(axis="both", which="major", labelsize=12, top="off")
    ax.set_xlabel(r"Diameter ($\mathregular{\AA)}$", fontsize=12)
    fig.tight_layout()
    fig.savefig(
        output_directory / "8_trajectory_maxdim.pdf",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close("all")


if __name__ == "__main__":
    main()
