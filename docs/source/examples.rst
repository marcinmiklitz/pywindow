Examples
========

Here, we highlight some useful simple examples for :mod:`pywindow`.
These are copied from the `examples` path in the github repository.

.. testsetup:: analysing-cage

    import pathlib

    path1 = pathlib.Path('_static/')
    path2 = pathlib.Path('source/_static/')

    if path1.exists():
        path = path1
    else:
        path = path2

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

    molsys = pw.MolecularSystem.load_file(path / "PUDXES.xyz")
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

We can load molecules in multiple ways:

.. testcode:: analysing-cage

    import pywindow as pw

    # From xyz, or pdb file:
    molsys = pw.MolecularSystem.load_file(path / "PUDXES.xyz")

    # From an rdkit molecule:
    # rdkit_mol = Chem.MolFromMol2File(input_file)
    # molsys = pw.MolecularSystem.load_rdkit_mol(rdkit_mol)

    # Then convert to molecule.
    mol = molsys.system_to_molecule()


There are specific methods available for analysis only some aspects of a
molecule, but you can also just run the full analysis:

.. testcode:: analysing-cage

    # Separate:
    # mol.calculate_centre_of_mass()
    # mol.calculate_maximum_diameter()
    # mol.calculate_average_diameter()
    # mol.calculate_pore_diameter()
    # mol.calculate_pore_volume()
    # mol.calculate_pore_diameter_opt()
    # mol.calculate_pore_volume_opt()
    # mol.calculate_windows()

    # All together:
    mol.full_analysis()

.. warning::
    The full analysis may fail on certain steps, so if you only need specific
    things, do that.

We can output to `.json` for the data, and `.pdb` for the structure, with added
information about pore and window (We can now visualise the molecule, its
centre of pore and its window centroids):

.. testcode:: analysing-cage

    mol.dump_properties_json(
        filepath=str(path / "PUDXES_out.json"),
        override=True,
    )

    mol.dump_molecule(
        filepath=str(path / "PUDXES_out.pdb"),
        include_coms=True,
        override=True,
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

    molsys = pw.MolecularSystem.load_file(path / "PUDXES_out.pdb")
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


.. testcode:: analysing-cage
    :hide:

    import numpy as np

    known_properties = {
        "PUDXES": {
            "centre_of_mass": np.array([12.4, 12.4, 12.4]),
            "maximum_diameter": {
                "atom_1": 12,
                "atom_2": 54,
                "diameter": 22.179369990077188,
            },
            "no_of_atoms": 168,
            "pore_diameter": {"atom": 29, "diameter": 5.3970201773100221},
            "pore_diameter_opt": {
                "atom_1": 29,
                "centre_of_mass": np.array([12.4, 12.4, 12.4]),
                "diameter": 5.3970201773100221,
            },
            "pore_volume": 82.311543851544172,
            "pore_volume_opt": 82.311543851544172,
            "windows": {
                "centre_of_mass": np.array(
                    [
                        [10.77105705, 10.77097707, 14.02893956],
                        [14.01544846, 14.0154126, 14.01539845],
                        [13.92965524, 10.87029766, 10.87034163],
                        [10.77542236, 14.02453217, 10.77546634],
                    ]
                ),
                "diameters": np.array(
                    [3.63778746, 3.63562103, 3.62896512, 3.63707237]
                ),
            },
            "average_diameter": 13.83201751425547,
        },
    }

    (same_dict, failed_prop) = pw.compare_properties_dict(
        dict1=mol.properties,
        dict2=known_properties["PUDXES"],
    )

    assert same_dict
