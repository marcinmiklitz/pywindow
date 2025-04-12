"""Example 5."""

import logging
import pathlib

import numpy as np

import pywindow as pw

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

known_properties = {
    "SAYGOR": {
        "centre_of_mass": np.array([21.05422233, 10.13114265, 0.90872613]),
        "no_of_atoms": 448,
        "pore_diameter_opt": {
            "diameter": np.float64(9.404969612349447),
            "atom_1": 88,
            "centre_of_mass": np.array([20.89673428, 10.30868293, 1.05310027]),
        },
        "pore_volume_opt": 435.58502104435024,
        "windows": {
            "centre_of_mass": np.array(
                [
                    [10.77105705, 10.77097707, 14.02893956],
                    [14.01544846, 14.0154126, 14.01539845],
                    [13.92965524, 10.87029766, 10.87034163],
                    [10.77542236, 14.02453217, 10.77546634],
                ]
            ),
            "diameters": np.array(
                [7.8918444, 8.29658872, 5.95684323, 6.80863739]
            ),
        },
    },
}


def main() -> None:
    """Run script."""
    script_directory = pathlib.Path(__file__).parent.resolve()

    data_directory = script_directory / "data"
    input_directory = data_directory / "input"
    output_directory = data_directory / "output"

    input_files = [input_directory / "SAYGOR.pdb"]
    for input_file in input_files:
        name = input_file.name.split(".")[0]

        molsys = pw.MolecularSystem.load_file(input_file)
        mol = molsys.system_to_molecule()

        mol.calculate_centre_of_mass()
        mol.calculate_pore_volume_opt()
        mol.calculate_windows()

        logging.info("properties for %s: %s", input_file.name, mol.properties)
        (same_dict, failed_prop) = pw.utilities.compare_properties_dict(
            dict1=mol.properties,
            dict2=known_properties[name],
        )

        if not same_dict:
            msg = (
                f"mol.properties not the same as known for "
                f"{input_file.name} in {failed_prop}"
                f"\n {mol.properties}"
            )
            raise RuntimeError(msg)

        mol.dump_molecule(
            filepath=str(output_directory / f"{name}_out.pdb"),
            include_coms=True,
            override=True,
        )


if __name__ == "__main__":
    main()
