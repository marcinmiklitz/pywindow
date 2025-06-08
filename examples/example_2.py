"""Example 2."""

import logging
import pathlib

import numpy as np
from rdkit import Chem

import pywindow as pw

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

known_properties = {
    "PUDXES": {
        "centre_of_mass": np.array([12.4, 12.4, 12.4]),
        "maximum_diameter": {
            "atom_1": 6,
            "atom_2": 69,
            "diameter": 21.77602100564755,
        },
        "no_of_atoms": 84,
        "pore_diameter": {"atom": 1, "diameter": 5.3970201773100097},
        "pore_diameter_opt": {
            "atom_1": 1,
            "centre_of_mass": np.array([12.4, 12.4, 12.4]),
            "diameter": 5.397020177310047,
        },
        "pore_volume": 82.311543851543604,
        "pore_volume_opt": 82.311543851543604,
        "average_diameter": 13.599974908590866,
        "windows": {
            "centre_of_mass": np.array(
                [
                    [10.77301184, 10.7730221, 14.02703257],
                    [14.02096764, 14.02099401, 14.02099029],
                    [13.94105869, 10.85893743, 10.85890308],
                    [10.79398951, 14.00595376, 10.79403717],
                ]
            ),
            "diameters": np.array(
                [3.63748192, 3.63649472, 3.62912867, 3.63426077]
            ),
        },
    },
}


def main() -> None:
    """Run script."""
    script_directory = pathlib.Path(__file__).parent.resolve()

    data_directory = script_directory / "data"
    input_directory = data_directory / "input"

    input_files = [input_directory / "PUDXES.mol2"]
    for input_file in input_files:
        name = input_file.name.split(".")[0]

        rdkit_mol = Chem.MolFromMol2File(str(input_file))
        molsys = pw.MolecularSystem.load_rdkit_mol(rdkit_mol)
        mol = molsys.system_to_molecule()

        mol.full_analysis()
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
