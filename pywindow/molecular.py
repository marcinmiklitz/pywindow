"""
Defines MolecularSystem and Molecule class.

This module is the most important part of the ``pywindow`` package, as it is
at the frontfront of the interaction with the user. The two main classes
defined here: :class:`MolecularSystem` and :class:`Molecule` are used to
store and analyse single molecules or assemblies of single molecules. The
:class:`MolecularSystem` contains hooks to the :module:io_tools that allow
to load molecules from various input types.

It is designed to handle
input in form of single molecules or assamblies of molecules (the molecular
system) with the two main classes defined here: :class:`MolecularSystem` and
:class:`Molecule`. These are used to load and refine input data in form of a
molecular system, extract individual molecules and perform the analysis.

The :class:`MolecularSystem` is used as a first step to the analysis. It allows
to load data, refine it (rebuild molecule in periodic systems with
:func:`rebuild()``, extract individual molecules with
:func:`make_modular()` and decipher force field dependent atom keys with
:func:`decipher_atom_keys()`) and extract single molecules for analysis.

The :class:`Molecule` is designed to contain a single molecule. Here the
analysis is performed.

"""

import os
import numpy as np
from copy import deepcopy
from scipy.spatial import ConvexHull

from .io_tools import Input, Output
from .utilities import (discrete_molecules,
                        decipher_atom_key,
                        molecular_weight,
                        center_of_mass,
                        max_dim,
                        pore_diameter,
                        opt_pore_diameter,
                        sphere_volume,
                        find_windows,
                        shift_com,
                        create_supercell,
                        is_inside_polyhedron,
                        find_average_diameter,
                        calculate_pore_shape,
                        circumcircle,
                        to_list,
                        align_principal_ax,
                        get_inertia_tensor,
                        get_gyration_tensor,
                        calc_asphericity,
                        calc_acylidricity,
                        calc_relative_shape_anisotropy,
                        find_windows_new,
                        calculate_window_diameter,
                        get_window_com,
                        window_shape)


class _MolecularSystemError(Exception):
    def __init__(self, message):
        self.message = message


class _NotAModularSystem(Exception):
    def __init__(self, message):
        self.message = message


class _Shape:
    """
    Class containing shape descriptors.

    This class allows other classes, such as :class:`Pore` and
    :class:`Molecule`, inherit shape descriptors (applicable to any set of
    points in a 3D Cartesian space, let it be a shape of an intrinsic pore or
    shape of a molecule) such as asphericity, acylidricity and relative shape
    anisotropy. This class should not be used by itself.

    """

    @property
    def _asphericity(self):
        """
        Return asphericity of a shape. NOT READY!!!

        The asphericity of a shape is weighted by the mass assigned to each
        coordinate (associated with the element). In case if `elements` is
        `None`, mass of each element = 1 and this returns a non-weighted value.

        Returns
        -------
        :class:`float`
           The asphericity of a shape.

        """
        return calc_asphericity(self.elements, self.coordinates)

    @property
    def _acylidricity(self):
        """
        Return acylidricity of a shape. NOT READY!!!

        The acylidricity of a shape is weighted by the mass assigned to each
        coordinate (associated with the element). In case if `elements` is
        `None`, mass of each element = 1 and this returns a non-weighted value.

        Returns
        -------
        :class:`float`
           The acylidricity of a shape.

        """
        return calc_acylidricity(self.elements, self.coordinates)

    @property
    def _relative_shape_anisotropy(self):
        """
        Return relative shape anisotropy of a shape. NOT READY!!!

        The relative shape anisotropy of a shape is weighted by the mass
        assigned to each coordinate (associated with the element). In case if
        `elements` is `None`, mass of each element = 1 and this returns a
        non-weighted value.

        Returns
        -------
        :class:`float`
           The relative shape anisotropy of a shape.

        """
        return calc_relative_shape_anisotropy(
            self.elements, self.coordinates
            )

    @property
    def inertia_tensor(self):
        """
        Return inertia tensor of a shape.

        The inertia tensor of a shape is weighted by the mass assigned to each
        coordinate (associated with the element). In case if `elements` is
        `None`, mass of each element = 1 and this returns a non-weighted value.

        Returns
        -------
        :class:`numpy.array`
           The inertia tensor of a shape.

        """
        return get_inertia_tensor(
            self.elements, self.coordinates
            )

    @property
    def gyration_tensor(self):
        """
        Return gyration tensor of a shape.

        The gyration tensor of a shape is weighted by the mass assigned to each
        coordinate (associated with the element). In case if `elements` is
        `None`, mass of each element = 1 and this returns a non-weighted value.

        Returns
        -------
        :class:`numpy.array`
           The gyration tensor of a shape.

        """
        return get_gyration_tensor(self.elements, self.coordinates)


class _Pore(_Shape):
    """Under development."""

    def __init__(self, elements, coordinates, shape=False, **kwargs):
        self._elements, self._coordinates = elements, coordinates
        self.diameter, self.closest_atom = pore_diameter(
            elements, coordinates, **kwargs)
        self.spherical_volume = sphere_volume(self.diameter / 2)
        if 'com' in kwargs.keys():
            self.centre_coordinates = kwargs['com']
        else:
            self.centre_coordinates = center_of_mass(elements, coordinates)
        self.optimised = False

    def optimise(self, **kwargs):
        (self.diameter, self.closest_atom,
         self.centre_coordinates) = opt_pore_diameter(
             self._elements,
             self._coordinates,
             com=self.centre_coordinates,
             **kwargs)
        self.spherical_volume = sphere_volume(self.diameter / 2)
        self.optimised = True

    def get_shape(self):
        super().__init__(calculate_pore_shape(self._coordinates))

    def reset(self):
        self.__init__(self._elements, self._coordinates)


class _Window:
    """Under development."""

    def __init__(self, window, key, elements, coordinates, com_adjust):
        """
        Return.
        """
        self.raw_data = window
        self.index = key
        self.mol_coordinates = coordinates
        self.mol_elements = elements
        self.com_correction = com_adjust
        self.shape = None
        self.convexhull = None

    def calculate_diameter(self, **kwargs):
        diameter = calculate_window_diameter(
            self.raw_data, self.mol_elements, self.mol_coordinates, **kwargs
        )
        return diameter

    def calculate_centre_of_mass(self, **kwargs):
        com = get_window_com(
            self.raw_data, self.mol_elements, self.mol_coordinates,
            self.com_correction, **kwargs
        )
        return com

    def get_shape(self, **kwargs):
        self.shape = window_shape(
            self.raw_data, self.mol_elements, self.mol_coordinates
        )
        return self.shape

    def get_convexhull(self):
        hull = ConvexHull(self.shape)
        verticesx = np.append(
            self.shape[hull.vertices, 0], self.shape[hull.vertices, 0][0]
        )
        verticesy = np.append(
            self.shape[hull.vertices, 1], self.shape[hull.vertices, 1][0]
        )
        self.convexhull = verticesx, verticesy
        return self.convexhull


class Molecule(_Shape):
    """
    Container for a single molecule.

    This class is ment for the analysis of single molecules, molecular pores
    especially. The object passed to this class should therefore be a finite
    and interconnected individuum. This class should not be initialised
    directly, but through the :method:system_to_molecule
    of :class:MolecularSystem, if the former condition is met, or result from
    the :method:make_modular of :class:MolecularSystem that returns a list of
    :class:Molecule objects.

    The methods in :class:Molecule allow the analysis of structural features
    associated with molecular pores, such as intrinsic pore dimensions or
    window diameters, but some general to any molecule, such as molecular
    weight, maximum dimension of a molecule or it's average diameterself.

    Short description of molecular features (for details see documentation):


    Attributes
    ----------
    mol : : dict
        The :attribute:system of :class:MolecularSystem passed to the
        :class:Molecule which is esentially a container of the information
        that compose a molecular entity, such as coordinates and
        atom ids and/or elements.

    no_of_atoms : : int
        The number of atoms in the molecule.

    elements : : numpy.array
        An array containing the elements, as strings, composing the molecule

    atom_ids : : numpy.array (conditional)
        If the :attribute:mol contains 'atom_ids' keyword, the force field
        ids of the elements

    coordinates : : numpy.array
        The x, y and z atomic Cartesian coordinates of all elements

    parent_system : : str
        The :attribute:name of :class:MolecularSystem passed to :class:Molecule

    molecule_id : : any
        The molecule id passed when initialising :class:Molecule

    properties : : dict
        A dictionary that is populated by the output of :class:Molecule methods

    windows : : None
        Attribute that is populated with the :class:Windows objects when
        :method:get_windows of :class:Molecule is used.

    """
    def __init__(self, mol, system_name, mol_id):
        self._Output = Output()
        self.mol = mol
        self.no_of_atoms = len(mol['elements'])
        self.elements = mol['elements']
        if 'atom_ids' in mol.keys():
            self.atom_ids = mol['atom_ids']
        self.coordinates = mol['coordinates']
        self.parent_system = system_name
        self.molecule_id = mol_id
        self.properties = {'no_of_atoms': self.no_of_atoms}
        self.windows = None

    @classmethod
    def load_rdkit_mol(cls, mol, system_name='rdkit', mol_id=0):
        return cls(Input().load_rdkit_mol(mol), system_name, mol_id)

    def full_analysis(self, ncpus=1, **kwargs):
        self.molecular_weight()
        self.calculate_centre_of_mass()
        self.calculate_maximum_diameter()
        self.calculate_pore_diameter()
        self.calculate_pore_volume()
        self.calculate_pore_diameter_opt(**kwargs)
        self.calculate_pore_volume_opt(**kwargs)
        self.calculate_windows(ncpus=ncpus, **kwargs)
        # self._circumcircle(**kwargs)
        return self.properties

    def align_to_principal_axes(self, align_molsys=False):
        if align_molsys:
            self.coordinates[0] = align_principal_ax_all(
                self.elements, self.coordinates
                )
        else:
            self.coordinates[0] = align_principal_ax(
                self.elements, self.coordinates
                )
        self.aligned_to_principal_axes = True

    def _get_pore(self):
        return Pore(self.elements, self.coordinates)

    def _get_shape(self, **kwargs):
        super().__init__(self.coordinates, elements=self.elements)

    def _get_windows(self, **kwargs):
        windows = find_windows_new(self.elements, self.coordinates, **kwargs)
        if windows:
            self.windows = [
                Window(np.array(windows[0][window]), window, windows[1],
                       windows[2], windows[3])
                for window in windows[0] if window != -1
            ]
            return self.windows
        else:
            return None

    def calculate_centre_of_mass(self):
        self.centre_of_mass = center_of_mass(self.elements, self.coordinates)
        self.properties['centre_of_mass'] = self.centre_of_mass
        return self.centre_of_mass

    def calculate_maximum_diameter(self):
        self.maxd_atom_1, self.maxd_atom_2, self.maximum_diameter = max_dim(
            self.elements, self.coordinates)
        self.properties['maximum_diameter'] = {
            'diameter': self.maximum_diameter,
            'atom_1': int(self.maxd_atom_1),
            'atom_2': int(self.maxd_atom_2),
        }
        return self.maximum_diameter

    def calculate_average_diameter(self, **kwargs):
        self.average_diameter = find_average_diameter(
            self.elements, self.coordinates, **kwargs)
        return self.average_diameter

    def calculate_pore_diameter(self):
        self.pore_diameter, self.pore_closest_atom = pore_diameter(
            self.elements, self.coordinates)
        self.properties['pore_diameter'] = {
            'diameter': self.pore_diameter,
            'atom': int(self.pore_closest_atom),
        }
        return self.pore_diameter

    def calculate_pore_volume(self):
        self.pore_volume = sphere_volume(self.calculate_pore_diameter() / 2)
        self.properties['pore_volume'] = self.pore_volume
        return self.pore_volume

    def calculate_pore_diameter_opt(self, **kwargs):
        (self.pore_diameter_opt, self.pore_opt_closest_atom,
         self.pore_opt_COM) = opt_pore_diameter(self.elements,
                                                self.coordinates, **kwargs)
        self.properties['pore_diameter_opt'] = {
            'diameter': self.pore_diameter_opt,
            'atom_1': int(self.pore_opt_closest_atom),
            'centre_of_mass': self.pore_opt_COM,
        }
        return self.pore_diameter_opt

    def calculate_pore_volume_opt(self, **kwargs):
        self.pore_volume_opt = sphere_volume(
            self.calculate_pore_diameter_opt(**kwargs) / 2)
        self.properties['pore_volume_opt'] = self.pore_volume_opt
        return self.pore_volume_opt

    def calculate_pore_shape(self, filepath='shape.xyz', **kwargs):
        shape = calculate_pore_shape(self.elements, self.coordinates, **kwargs)
        shape_obj = {'elements': shape[0], 'coordinates': shape[1]}
        Output()._save_xyz(shape_obj, filepath)
        return 1

    def calculate_windows(self, **kwargs):
        windows = find_windows(self.elements, self.coordinates, **kwargs)
        if windows:
            self.properties.update(
                {
                    'windows': {
                        'diameters': windows[0], 'centre_of_mass': windows[1],
                    }
                }
            )
            return windows[0]
        else:
            self.properties.update(
                {'windows': {'diameters': None,  'centre_of_mass': None, }}
            )
        return None

    def shift_to_origin(self, **kwargs):
        self.coordinates = shift_com(self.elements, self.coordinates, **kwargs)
        self._update()

    def molecular_weight(self):
        self.MW = molecular_weight(self.elements)
        return self.MW

    def dump_properties_json(self, filepath=None, molecular=False, **kwargs):
        # We pass a copy of the properties dictionary.
        dict_obj = deepcopy(self.properties)
        # If molecular data is also required we update the dictionary.
        if molecular is True:
            dict_obj.update(self.mol)
        # If no filepath is provided we create one.
        if filepath is None:
            filepath = "_".join(
                (str(self.parent_system), str(self.molecule_id))
            )
            filepath = '/'.join((os.getcwd(), filepath))
        # Dump the dictionary to json file.
        self._Output.dump2json(dict_obj, filepath, default=to_list, **kwargs)

    def dump_molecule(self, filepath=None, include_coms=False, **kwargs):
        # If no filepath is provided we create one.
        if filepath is None:
            filepath = "_".join(
                (str(self.parent_system), str(self.molecule_id)))
            filepath = '/'.join((os.getcwd(), filepath))
            filepath = '.'.join((filepath, 'pdb'))
        # Check if there is an 'atom_ids' keyword in the self.mol dict.
        # Otherwise pass to the dump2file atom_ids='elements'.
        if 'atom_ids' not in self.mol.keys():
            atom_ids = 'elements'
        else:
            atom_ids = 'atom_ids'
        # Dump molecule into a file.
        # If coms are to be included additional steps are required.
        # First deepcopy the molecule
        if include_coms is True:
            mmol = deepcopy(self.mol)
            # add centre of mass (centre of not optimised pore) as 'He'.
            mmol['elements'] = np.concatenate(
                (mmol['elements'], np.array(['He'])))
            if 'atom_ids' not in self.mol.keys():
                pass
            else:
                mmol['atom_ids'] = np.concatenate(
                    (mmol['atom_ids'], np.array(['He'])))
            mmol['coordinates'] = np.concatenate(
                (mmol['coordinates'],
                 np.array([self.properties['centre_of_mass']])))
            # add centre of pore optimised as 'Ne'.
            mmol['elements'] = np.concatenate(
                (mmol['elements'], np.array(['Ne'])))
            if 'atom_ids' not in self.mol.keys():
                pass
            else:
                mmol['atom_ids'] = np.concatenate(
                    (mmol['atom_ids'], np.array(['Ne'])))
            mmol['coordinates'] = np.concatenate(
                (mmol['coordinates'], np.array(
                    [self.properties['pore_diameter_opt']['centre_of_mass']])))
            # add centre of windows as 'Ar'.
            if self.properties['windows']['centre_of_mass'] is not None:
                range_ = range(
                    len(self.properties['windows']['centre_of_mass']))
                for com in range_:
                    mmol['elements'] = np.concatenate(
                        (mmol['elements'], np.array(['Ar'])))
                    if 'atom_ids' not in self.mol.keys():
                        pass
                    else:
                        mmol['atom_ids'] = np.concatenate(
                            (mmol['atom_ids'],
                             np.array(['Ar{0}'.format(com + 1)])))
                    mmol['coordinates'] = np.concatenate(
                        (mmol['coordinates'], np.array([
                            self.properties['windows']['centre_of_mass'][com]
                        ])))
            self._Output.dump2file(mmol, filepath, atom_ids=atom_ids, **kwargs)

        else:
            self._Output.dump2file(
                self.mol, filepath, atom_ids=atom_ids, **kwargs)

    def _update(self):
        self.mol['coordinates'] = self.coordinates
        self.calculate_centre_of_mass()
        self.calculate_pore_diameter_opt()

    def _circumcircle(self, **kwargs):
        windows = circumcircle(self.coordinates, kwargs['atom_sets'])
        if 'output' in kwargs:
            if kwargs['output'] == 'windows':
                self.properties['circumcircle'] = {'diameter': windows, }
        else:
            if windows is not None:
                self.properties['circumcircle'] = {
                    'diameter': windows[0],
                    'centre_of_mass': windows[1],
                }
            else:
                self.properties['circumcircle'] = {
                    'diameter': None,
                    'centre_of_mass': None,
                }
        return windows


class MolecularSystem:
    """
    Container for the molecular system.

    To load input and initialise :class:`MolecularSystem`, one of the
    :class:`MolecularSystem` classmethods (:func:`load_file()`,
    :func:`load_rdkit_mol()` or :func:`load_system()`) should be used.
    :class:`MolecularSystem` **should not be initialised by itself.**

    Examples
    --------
    1. Using file as an input:

    .. code-block:: python

        pywindow.MolecularSystem.load_file(`filepath`)

    2. Using RDKit molecule object as an input:

    .. code-block:: python

        pywindow.MolecularSystem.load_rdkit_mol(rdkit.Chem.rdchem.Mol)

    2. Using a dictionary (or another :attr:`MoleculeSystem.system`) as input:

    .. code-block:: python

        pywindow.MolecularSystem.load_system({...})

    Attributes
    ----------
    system_id : :class:`str` or :class:`int`
        The input filename or user defined.

    system : :class:`dict`
        A dictionary containing all the information extracted from input.

    molecules : :class:`list`
        A list containing all the returned :class:`Molecule` s after using
        :func:`make_modular()`.

    """

    def __init__(self):
        self._Input = Input()
        self._Output = Output()
        self.system_id = 0

    @classmethod
    def load_file(cls, filepath):
        """
        Create a :class:`MolecularSystem` from an input file.

        Recognized input file formats: XYZ, PDB and MOL (V3000).

        Parameters
        ----------
        filepath : :class:`str`
           The input's filepath.

        Returns
        -------
        :class:`pywindow.molecular.MolecularSystem`
            :class:`MolecularSystem`

        """
        obj = cls()
        obj.system = obj._Input.load_file(filepath)
        obj.filename = os.path.basename(filepath)
        obj.system_id = obj.filename.split(".")[0]
        obj.name, ext = os.path.splitext(obj.filename)
        return obj

    @classmethod
    def load_rdkit_mol(cls, mol):
        """
        Create a :class:`MolecularSystem` from :class:`rdkit.Chem.rdchem.Mol`.

        Parameters
        ----------
        mol : :class:`rdkit.Chem.rdchem.Mol`
           An RDKit molecule object.

        Returns
        -------
        :class:`pywindow.molecular.MolecularSystem`
            :class:`MolecularSystem`

        """
        obj = cls()
        obj.system = obj._Input.load_rdkit_mol(mol)
        return obj

    @classmethod
    def load_system(cls, dict_, system_id='system'):
        """
        Create a :class:`MolecularSystem` from a python :class:`dict`.

        As the loaded :class:`MolecularSystem` is storred as a :class:`dict` in
        the :class:`MolecularSystem.system` it can also be loaded directly from
        a :class:`dict` input. This feature is used by :mod:`trajectory` that
        extracts trajectory frames as dictionaries and returns them
        as :class:`MolecularSystem` objects through this classmethod.

        Parameters
        ----------
        dict_ : :class:`dict`
           A python dictionary.

        system_id : :class:`str` or :class:'int', optional
           Inherited or user defined system id. (default='system')

        Returns
        -------
        :class:`pywindow.molecular.MolecularSystem`
            :class:`MolecularSystem`

        """
        obj = cls()
        obj.system = dict_
        obj.system_id = system_id
        return obj

    def rebuild_system(self, override=False, **kwargs):
        """
        Rebuild molecules in molecular system.

        Parameters
        ----------
        override : :class:`bool`, optional (default=False)
            If False the rebuild molecular system is returned as a new
            :class:`MolecularSystem`, if True, the current
            :class:`MolecularSystem` is modified.

        """
        # First we create a 3x3x3 supercell with the initial unit cell in the
        # centre and the 26 unit cell translations around to provide all the
        # atom positions necessary for the molecules passing through periodic
        # boundary reconstruction step.
        supercell_333 = create_supercell(self.system, **kwargs)
        # smolsys = self.load_system(supercell_333, self.system_id + '_311')
        # smolsys.dump_system(override=True)
        discrete = discrete_molecules(self.system, rebuild=supercell_333)
        # This function overrides the initial data for 'coordinates',
        # 'atom_ids', and 'elements' instances in the 'system' dictionary.
        coordinates = np.array([], dtype=np.float64).reshape(0, 3)
        atom_ids = np.array([])
        elements = np.array([])
        for i in discrete:
            coordinates = np.concatenate(
                [coordinates, i['coordinates']], axis=0
                )
            atom_ids = np.concatenate([atom_ids, i['atom_ids']], axis=0)
            elements = np.concatenate([elements, i['elements']], axis=0)
        rebuild_system = {
            'coordinates': coordinates,
            'atom_ids': atom_ids,
            'elements': elements
            }
        if override is True:
            self.system.update(rebuild_system)
            return None
        else:
            return self.load_system(rebuild_system)

    def swap_atom_keys(self, swap_dict, dict_key='atom_ids'):
        """
        Swap a force field atom id for another user-defined value.

        This modified all values in :attr:`MolecularSystem.system['atom_ids']`
        that match criteria.

        This function can be used to decipher a whole forcefield if an
        appropriate dictionary is passed to the function.

        Example
        -------
        In this example all atom ids 'he' will be exchanged to 'H'.

        .. code-block:: python

            pywindow.MolecularSystem.swap_atom_keys({'he': 'H'})

        Parameters
        ----------
        swap_dict: :class:`dict`
            A dictionary containg force field atom ids (keys) to be swapped
            with corresponding values (keys' arguments).

        dict_key: :class:`str`
            A key in :attr:`MolecularSystem.system` dictionary to perform the
            atom keys swapping operation on. (default='atom_ids')

        Returns
        -------
        None : :class:`NoneType`

        """
        # Similar situation to the one from decipher_atom_keys function.
        if 'atom_ids' not in self.system.keys():
            dict_key = 'elements'
        for atom_key in range(len(self.system[dict_key])):
            for key in swap_dict.keys():
                if self.system[dict_key][atom_key] == key:
                    self.system[dict_key][atom_key] = swap_dict[key]

    def decipher_atom_keys(self, forcefield='DLF', dict_key='atom_ids'):
        """
        Decipher force field atom ids.

        This takes all values in :attr:`MolecularSystem.system['atom_ids']`
        that match force field type criteria and creates
        :attr:`MolecularSystem.system['elements']` with the corresponding
        periodic table of elements equivalents.

        If a forcefield is not supported by this method, the
        :func:`MolecularSystem.swap_atom_keys()` can be used instead.

        DLF stands for DL_F notation.

        See: C. W. Yong, Descriptions and Implementations of DL_F Notation: A
        Natural Chemical Expression System of Atom Types for Molecular
        Simulations, J. Chem. Inf. Model., 2016, 56, 1405–1409.

        Parameters
        ----------
        forcefield : :class:`str`
            The forcefield used to decipher atom ids. Allowed (not case
            sensitive): 'OPLS', 'OPLS2005', 'OPLSAA', 'OPLS3', 'DLF', 'DL_F'.
            (default='DLF')

        dict_key : :class:`str`
            The :attr:`MolecularSystem.system` dictionary key to the array
            containing the force field atom ids. (default='atom_ids')

        Returns
        -------
        None : :class:`NoneType`

        """
        # In case there is no 'atom_ids' key we try 'elements'. This is for
        # XYZ and MOL files mostly. But, we keep the dict_key keyword for
        # someone who would want to decipher 'elements' even if 'atom_ids' key
        # is present in the system's dictionary.
        if 'atom_ids' not in self.system.keys():
            dict_key = 'elements'
        # I do it on temporary object so that it only finishes when successful
        temp = deepcopy(self.system[dict_key])
        for element in range(len(temp)):
            temp[element] = "{0}".format(
                decipher_atom_key(
                    temp[element], forcefield=forcefield))
        self.system['elements'] = temp

    def make_modular(self, rebuild=False):
        """
        Find and return all :class:`Molecule` s in :class:`MolecularSystem`.

        This function populates :attr:`MolecularSystem.molecules` type
        :class:`list` with :class:`Molecule` s.

        Parameters
        ----------
        rebuild : :class:`Bool`
            If True, run first the :func:`rebuild_system()`. (default=False)

        Returns
        -------
        None : :class:`NoneType`

        """
        if rebuild is True:
            supercell_333 = create_supercell(self.system)
        else:
            supercell_333 = None
        dis = discrete_molecules(self.system, rebuild=supercell_333)
        self.no_of_discrete_molecules = len(dis)
        self.molecules = {}
        for i in range(len(dis)):
            self.molecules[i] = Molecule(dis[i], self.system_id, i)

    def system_to_molecule(self):
        """
        Return :class:`MolecularSystem` as a :class:`Molecule` directly.

        Only to be used conditionally, when the :class:`MolecularSystem` is a
        discrete molecule and no input pre-processing is required.

        Returns
        -------
        :class:`pywindow.molecular.Molecule`
            :class:`Molecule`
        """
        return Molecule(self.system, self.system_id, 0)

    def _get_pores(self, sampling_points):
        """ Under development."""
        pores = []
        for point in sampling_points:
            pores.append(
                Pore(
                    self.system['elements'],
                    self.system['coordinates'],
                    com=point))
        return pores

    def dump_system(self, filepath=None, modular=False, **kwargs):
        """
        Dump a :class:`MolecularSystem` to a file (PDB or XYZ).

        Kwargs are passed to :func:`pywindow.io_tools.Output.dump2file()`.

        Parameters
        ----------
        filepath : :class:`str`
           The filepath for the dumped file. If :class:`None`, the file is
           dumped localy with :attr:`system_id` as filename.
           (defualt=None)

        modular : :class:`Bool`
            If False, dump the :class:`MolecularSystem` as in
            :attr:`MolecularSystem.system`, if True, dump the
            :class:`MolecularSystem` as catenated :class:Molecule objects
            from :attr:`MolecularSystem.molecules`

        Returns
        -------
        None : :class:`NoneType`

        """
        # If no filepath is provided we create one.
        if filepath is None:
            filepath = '/'.join((os.getcwd(), str(self.system_id)))
            filepath = '.'.join((filepath, 'pdb'))
        # If modular is True substitute the molecular data for modular one.
        system_dict = deepcopy(self.system)
        if modular is True:
            elements = np.array([])
            atom_ids = np.array([])
            coor = np.array([]).reshape(0, 3)
            for mol_ in self.molecules:
                mol = self.molecules[mol_]
                elements = np.concatenate((elements, mol.mol['elements']))
                atom_ids = np.concatenate((atom_ids, mol.mol['atom_ids']))
                coor = np.concatenate((coor, mol.mol['coordinates']), axis=0)
            system_dict['elements'] = elements
            system_dict['atom_ids'] = atom_ids
            system_dict['coordinates'] = coor
        # Check if there is an 'atom_ids' keyword in the self.mol dict.
        # Otherwise pass to the dump2file atom_ids='elements'.
        # This is mostly for XYZ files and not deciphered trajectories.
        if 'atom_ids' not in system_dict.keys():
            atom_ids = 'elements'
        else:
            atom_ids = 'atom_ids'
        # Dump system into a file.
        self._Output.dump2file(
            system_dict, filepath, atom_ids=atom_ids, **kwargs)

    def dump_system_json(self, filepath=None, modular=False, **kwargs):
        """
        Dump a :class:`MolecularSystem` to a JSON dictionary.

        The dumped JSON dictionary, with :class:`MolecularSystem`, can then be
        loaded through a JSON loader and then through :func:`load_system()`
        to retrieve a :class:`MolecularSystem`.

        Kwargs are passed to :func:`pywindow.io_tools.Output.dump2json()`.

        Parameters
        ----------
        filepath : :class:`str`
           The filepath for the dumped file. If :class:`None`, the file is
           dumped localy with :attr:`system_id` as filename.
           (defualt=None)

        modular : :class:`Bool`
            If False, dump the :class:`MolecularSystem` as in
            :attr:`MolecularSystem.system`, if True, dump the
            :class:`MolecularSystem` as catenated :class:Molecule objects
            from :attr:`MolecularSystem.molecules`

        Returns
        -------
        None : :class:`NoneType`

        """
        # We pass a copy of the properties dictionary.
        dict_obj = deepcopy(self.system)
        # In case we want a modular system.
        if modular is True:
            try:
                if self.molecules:
                    pass
            except AttributeError:
                raise _NotAModularSystem(
                    "This system is not modular. Please, run first the "
                    "make_modular() function of this class.")
            dict_obj = {}
            for molecule in self.molecules:
                mol_ = self.molecules[molecule]
                dict_obj[molecule] = mol_.mol
        # If no filepath is provided we create one.
        if filepath is None:
            filepath = '/'.join((os.getcwd(), str(self.system_id)))
        # Dump the dictionary to json file.
        self._Output.dump2json(dict_obj, filepath, default=to_list, **kwargs)
