"""Example 6."""

import pywindow as pw


def main() -> None:
    """Run script."""
    raise SystemExit("add asserts compared to all notebook output.")
    molsys = pw.MolecularSystem.load_file("./data/input/PUDXES_no_solvent.pdb")
    rebuild_molsys = molsys.rebuild_system()
    rebuild_molsys.dump_system(
        "./data/output/PUDXES_no_solvent_rebuild.pdb", override=True
    )
    rebuild_molsys.make_modular()
    for molecule in rebuild_molsys.molecules:
        print(
            f"Analysing molecule {molecule + 1} out of {len(rebuild_molsys.molecules)}"
        )
        mol = rebuild_molsys.molecules[molecule]
        print(mol.full_analysis(), "\n")
        # Each molecule can be saved separately
        mol.dump_molecule(
            f"./data/output/PUDXES_no_solvent_rebuild_mol_{molecule}.pdb",
            include_coms=True,
            override=True,
        )

    molsys = pw.MolecularSystem.load_file("./data/input/EPIRUR_no_solvent.pdb")
    rebuild_molsys = molsys.rebuild_system()
    rebuild_molsys.dump_system(
        "./data/output/EPIRUR_no_solvent_rebuild.pdb", override=True
    )
    rebuild_molsys.make_modular()

    for molecule in rebuild_molsys.molecules:
        print(
            f"Analysing molecule {molecule + 1} out of {len(rebuild_molsys.molecules)}"
        )
        mol = rebuild_molsys.molecules[molecule]
        print(mol.full_analysis(), "\n")
        # Each molecule can be saved separately
        mol.dump_molecule(
            f"./data/output/EPIRUR_no_solvent_rebuild_mol_{molecule}.pdb",
            include_coms=True,
            override=True,
        )

    molsys = pw.MolecularSystem.load_file("./data/input/TATVER_no_solvent.pdb")
    rebuild_molsys = molsys.rebuild_system()
    rebuild_molsys.dump_system(
        "./data/output/TATVER_no_solvent_rebuild.pdb", override=True
    )
    rebuild_molsys.make_modular()

    for molecule in rebuild_molsys.molecules:
        print(
            f"Analysing molecule {molecule + 1} out of {len(rebuild_molsys.molecules)}"
        )
        mol = rebuild_molsys.molecules[molecule]
        print(mol.full_analysis(), "\n")
        # Each molecule can be saved separately
        mol.dump_molecule(
            f"./data/output/TATVER_no_solvent_rebuild_mol_{molecule}.pdb",
            include_coms=True,
            override=True,
        )


if __name__ == "__main__":
    main()
