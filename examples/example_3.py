"""Example 3."""

import logging
import pathlib

import pywindow as pw

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

known_properties = {
    "PUDXES": {"no_of_atoms": 168, "average_diameter": 13.832017514255472},
}


def main() -> None:
    """Run script."""
    script_directory = pathlib.Path(__file__).parent.resolve()

    data_directory = script_directory / "data"
    input_directory = data_directory / "input"

    input_files = [input_directory / "PUDXES.xyz"]
    for input_file in input_files:
        name = input_file.name.split(".")[0]

        molsys = pw.MolecularSystem.load_file(input_file)

        mol = molsys.system_to_molecule()
        # To calculate average diameter of a molecule we use
        # calculate_average_diameter method of Molecule class. The output is
        # in the unit of distance from the input file - in this case Angstroms.
        mol.calculate_average_diameter()

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


if __name__ == "__main__":
    main()
