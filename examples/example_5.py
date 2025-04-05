"""Example 5."""

import pywindow as pw


def main() -> None:
    """Run script."""
    raise SystemExit("add asserts compared to all notebook output.")
    molsys = pw.MolecularSystem.load_file("./data/input/SAYGOR.pdb")
    mol = molsys.system_to_molecule()
    mol.calculate_pore_volume_opt()
    mol.calculate_windows()
    mol.calculate_centre_of_mass()
    mol.dump_molecule(
        "./data/output/SAYGOR_out.pdb", include_coms=True, override=True
    )


if __name__ == "__main__":
    main()
