"""Example 5."""

import logging
import pathlib

import numpy as np

import pywindow as pw

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

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
                    [23.15736791, 12.82039239, 4.78076671],
                    [20.78776097, 3.84867975, 1.78827362],
                    [16.21088915, 12.37208767, -0.03607041],
                    [22.48953934, 11.49736559, -2.50069612],
                ]
            ),
            "diameters": np.array(
                [7.89184685, 8.29659052, 5.95681339, 6.80868032]
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

        logger.info("properties for %s: %s", input_file.name, mol.properties)
        (same_dict, failed_prop) = pw.compare_properties_dict(
            dict1=mol.properties,  # type:ignore[arg-type]
            dict2=known_properties[name],  # type:ignore[arg-type]
        )

        if not same_dict:
            msg = (
                f"mol.properties not the same as known for "
                f"{input_file.name} in property: {failed_prop}"
                f"\n {mol.properties}"
            )
            raise RuntimeError(msg)

        mol.dump_molecule(
            filepath=output_directory / f"{name}_out.pdb",
            include_coms=True,
            override=True,
        )


if __name__ == "__main__":
    main()
