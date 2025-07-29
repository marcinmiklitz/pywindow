"""Example 4."""

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
    "MIBQAR": {
        "centre_of_mass": np.array([12.9459922, 12.9459922, 12.9459922]),
        "no_of_atoms": 424,
        "pore_diameter_opt": {
            "diameter": np.float64(12.277215763347375),
            "atom_1": 93,
            "centre_of_mass": np.array([12.946, 12.946, 12.946]),
        },
        "pore_volume_opt": 968.94312796544568,
        "windows": {
            "centre_of_mass": np.array(
                [
                    [12.94597477, 12.94601827, 18.83787765],
                    [7.21715443, 12.94597689, 12.94602313],
                    [12.94596701, 18.77462263, 12.94596938],
                    [18.85993235, 12.94601447, 12.94598322],
                    [12.9460142, 6.92697083, 12.94597315],
                    [12.94600128, 12.94597812, 7.06434291],
                ]
            ),
            "diameters": np.array(
                [
                    7.94805059,
                    7.9373968,
                    7.94278598,
                    7.95022874,
                    7.96284952,
                    7.94711847,
                ]
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

    input_files = [input_directory / "MIBQAR.pdb"]
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
