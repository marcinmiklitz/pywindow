"""Example 2."""

import logging

from rdkit import Chem

from pywindow import MolecularSystem


def main() -> None:
    """Run script."""
    raise SystemExit("add asserts compared to all notebook output.")
    rdkit_mol = Chem.MolFromMol2File("data/input/PUDXES.mol2")
    molsys = MolecularSystem.load_rdkit_mol(rdkit_mol)
    mol = molsys.system_to_molecule()
    logging.info(mol.full_analysis())


if __name__ == "__main__":
    main()
