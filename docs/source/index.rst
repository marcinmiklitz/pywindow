.. toctree::
   :hidden:
   :caption: pywindow
   :maxdepth: 2

   Molecule <_autosummary/pywindow.Molecule>
   MolecularSystem <_autosummary/pywindow.MolecularSystem>
   Single molecule examples <examples>
   Multiple molecules examples <multi_examples>

.. toctree::
  :hidden:
  :maxdepth: 2
  :caption: Modules:

  Modules <modules>

============
Introduction
============

| GitHub: https://www.github.com/marcinmiklitz/pywindow

Overview
--------

``pywindow`` is a Python 3 library for the structural analysis of molecular
pores.

For quick start head to :ref:`modindex` and see
:class:`pywindow.MolecularSystem`.


Installation
============

:mod:`pywindow` can be installed with pip:

.. code-block:: bash

  pip install pywindowx

Developer Setup
---------------

To develop with ``pywindow``, you can clone the repo and use
`just <https://github.com/casey/just>`_ and `uv <https://docs.astral.sh>`_
to setup the dev environment:

.. code-block:: bash

  just setup



Examples
========

For the specific examples of ``pywindow`` usage see
`Examples/` directory in the ``pywindow`` Github repository.

Loading input
-------------

1. Using file as an input:

.. code-block:: python

    import pywindow as pw

    molsys = pw.MolecularSystem.load_file("data/input/PUDXES.xyz")

2. Using RDKit molecule object as an input:

.. code-block:: python

    import pywindow as pw
    from rdkit import Chem

    rdkit_mol = Chem.MolFromMol2File("data/input/PUDXES.mol2")

    molsys = pw.MolecularSystem.load_rdkit_mol(rdkit_mol)

3. Using a dictionary (or another :attr:`MoleculeSystem.system`) as input:

.. code-block:: python

    import pywindow as pw

    molsys = pw.MolecularSystem.load_file("data/input/PUDXES.xyz")

    molsys2 = pw.MolecularSystem.load_system(molsys.system)

Pre-processing
--------------

If our input requires pre-processing (rebuilding molecules through periodic
boundary and/or force field atom ids deciphering) the following methods allow
that:

1. Rebuilding a periodic system

.. code-block:: python

    rebuild_molsys = molsys.rebuild_system()

2. Deciphering force field atom ids

.. code-block:: python

    molsys.decipher_atom_keys('OPLS')

If the force field is not supported by ``pywindow`` we can use
:func:`pywindow.MolecularSystem.decipher_atom_keys()` to generate a custom
force field deciphering tool.

.. code-block:: python

    some_forcefield = {
        'ca': 'C',
        'ni': 'N',
        'hc': 'H',
        'ha': 'H'
    }

    molsys.swap_atom_keys(some_forcefield)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
