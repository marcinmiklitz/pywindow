Example of loading a structure file with multiple molecules
===========================================================

We can load a structure that contains multiple molecules, separate them into
distinct molecules and analyse.

.. testsetup:: analysing-multi

    import pathlib

    path1 = pathlib.Path('_static/')
    path2 = pathlib.Path('source/_static/')

    if path1.exists():
        path = path1
    else:
        path = path2

We can load the file, but now we rebuild the structure:

.. testcode:: analysing-multi

    import pywindow as pw

    # From xyz, or pdb file:
    molsys = pw.MolecularSystem.load_file(path / "EPIRUR_no_solvent.pdb")

    # But now, we rebuild the system (capturing periodic effects).
    rebuild_molsys = molsys.rebuild_system()
    # And find all distinct molecules - as an attribute to `rebuild_molsys`.
    # We can dump the cleaned up system - see below.
    rebuild_molsys.dump_system(
        str(path / "EPIRUR_no_solvent_rebuild.pdb"),
        override=True,
    )
    rebuild_molsys.make_modular()


.. moldoc::

    import moldoc.molecule as molecule
    import pywindow as pw
    import pathlib

    path1 = pathlib.Path('_static/')
    path2 = pathlib.Path('source/_static/')

    if path1.exists():
        path = path1
    else:
        path = path2

    molsys = pw.MolecularSystem.load_file(path / "EPIRUR_no_solvent_rebuild.pdb")
    mol = molsys.system_to_molecule()

    moldoc_display_molecule = molecule.Molecule(
        atoms=(
            molecule.Atom(
                atomic_number=pw.periodic_table[ele],
                position=position,
            ) for ele, position in zip(
                mol.elements,
                mol.coordinates,
            )
        ),
        bonds=(),
        config=molecule.MoleculeConfig(
            atom_scale=1.5,
        ),
    )

Now we iterate through the molecules, analyse and save:

.. testcode:: analysing-multi

    pore_diameters = []
    for molecule_id in rebuild_molsys.molecules:
        mol = rebuild_molsys.molecules[molecule_id]
        mol.full_analysis()
        pore_diameters.append(mol.properties["pore_diameter"]["diameter"])

.. moldoc::

    import moldoc.molecule as molecule
    import pywindow as pw
    import pathlib

    path1 = pathlib.Path('_static/')
    path2 = pathlib.Path('source/_static/')

    if path1.exists():
        path = path1
    else:
        path = path2

    molsys = pw.MolecularSystem.load_file(path / "EPIRUR_no_solvent_rebuild_mol_0.pdb")
    mol = molsys.system_to_molecule()

    moldoc_display_molecule = molecule.Molecule(
        atoms=(
            molecule.Atom(
                atomic_number=pw.periodic_table[ele],
                position=position,
            ) for ele, position in zip(
                mol.elements,
                mol.coordinates,
            )
        ),
        bonds=(),
        config=molecule.MoleculeConfig(
            atom_scale=1.5,
        ),
    )

.. moldoc::

    import moldoc.molecule as molecule
    import pywindow as pw
    import pathlib

    path1 = pathlib.Path('_static/')
    path2 = pathlib.Path('source/_static/')

    if path1.exists():
        path = path1
    else:
        path = path2

    molsys = pw.MolecularSystem.load_file(path / "EPIRUR_no_solvent_rebuild_mol_1.pdb")
    mol = molsys.system_to_molecule()

    moldoc_display_molecule = molecule.Molecule(
        atoms=(
            molecule.Atom(
                atomic_number=pw.periodic_table[ele],
                position=position,
            ) for ele, position in zip(
                mol.elements,
                mol.coordinates,
            )
        ),
        bonds=(),
        config=molecule.MoleculeConfig(
            atom_scale=1.5,
        ),
    )


.. testcode:: analysing-multi
    :hide:

    import numpy as np

    known_diameters = [5.2999265295219633, 5.2993422655565112, 5.3002853308997366]
    for i,j in zip(pore_diameters, known_diameters):
        assert np.isclose(i, j)
