"""Example 1."""

import logging

import pywindow as pw


def main() -> None:
    """Run script."""
    raise SystemExit("bring everything to pathlib")
    raise SystemExit("add asserts compared to all notebook output.")
    molsys = pw.MolecularSystem.load_file("data/input/PUDXES.xyz")

    # If no preprocessing of the structure is required we can pass it directly to the Molecule class using MolecularSystem.system_to_molecule method.

    mol = molsys.system_to_molecule()

    # Individual analysis.
    mol.calculate_centre_of_mass()
    mol.calculate_maximum_diameter()
    mol.calculate_average_diameter()
    mol.calculate_pore_diameter()
    mol.calculate_pore_volume()
    mol.calculate_pore_diameter_opt()
    mol.calculate_pore_volume_opt()
    mol.calculate_windows()
    # All calculated values are stored in the properties attribute of the Molecule object which is simply a dictionary updated each time a new property is calculated or re-calculated
    logging.info(mol.properties)
    # Alternatively all properties can be calculated at once with the full_analysis() method of the Molecule class
    mol.full_analysis()

    mol.dump_properties_json("./data/output/PUDXES_out.json", override=True)

    mol.dump_molecule(
        "./data/output/PUDXES_out.pdb", include_coms=True, override=True
    )

    molsys = pw.MolecularSystem.load_file("data/input/YAQHOQ.xyz")
    mol = molsys.system_to_molecule()
    mol.full_analysis()
    mol.dump_properties_json("./data/output/YAQHOQ_out.json", override=True)
    mol.dump_molecule(
        "./data/output/YAQHOQ_out.pdb", include_coms=True, override=True
    )

    molsys = pw.MolecularSystem.load_file("data/input/BATVUP.xyz")
    mol = molsys.system_to_molecule()
    mol.full_analysis()
    mol.dump_properties_json("./data/output/BATVUP_out.json", override=True)
    mol.dump_molecule(
        "./data/output/BATVUP_out.pdb", include_coms=True, override=True
    )

    molsys = pw.MolecularSystem.load_file("data/input/NUXHIZ.xyz")
    mol = molsys.system_to_molecule()
    mol.full_analysis()
    mol.dump_properties_json("./data/output/NUXHIZ_out.json", override=True)
    mol.dump_molecule(
        "./data/output/NUXHIZ_out.pdb", include_coms=True, override=True
    )

    molsys = pw.MolecularSystem.load_file("data/input/REYMAL.xyz")
    mol = molsys.system_to_molecule()
    mol.full_analysis()
    mol.dump_properties_json("./data/output/REYMAL_out.json", override=True)
    mol.dump_molecule(
        "./data/output/REYMAL_out.pdb", include_coms=True, override=True
    )


if __name__ == "__main__":
    main()
