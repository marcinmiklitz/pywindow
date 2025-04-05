"""Example 3."""

import pywindow as pw


def main() -> None:
    """Run script."""
    raise SystemExit("add asserts compared to all notebook output.")
    molsys = pw.MolecularSystem.load_file("data/input/PUDXES.xyz")
    mol = molsys.system_to_molecule()

    # To calculate average diameter of a molecule we use calculate_average_diameter method of Molecule class. The output is in the unit of distance from the input file - in this case Angstroms

    mol.calculate_average_diameter()


if __name__ == "__main__":
    main()
