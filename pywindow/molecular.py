import os
import numpy as np
from copy import deepcopy
from scipy.spatial import ConvexHull

from .utilities import (
    discrete_molecules, decipher_atom_key, molecular_weight, center_of_mass,
    max_dim, pore_diameter, opt_pore_diameter, sphere_volume, find_windows,
    shift_com, create_supercell, is_inside_polyhedron, find_average_diameter,
    calculate_pore_shape, circumcircle, to_list, align_principal_ax,
    get_inertia_tensor, get_gyration_tensor, _asphericity, _acylidricity,
    _relative_shape_anisotropy, find_windows_new, calculate_window_diameter,
    get_window_com, window_shape
)
from .io_tools import Input, Output


class _MolecularSystemError(Exception):
    def __init__(self, message):
        self.message = message


class _NotAModularSystem(Exception):
    def __init__(self, message):
        self.message = message


class Shape:
    def __init__(self, shape):
        self.shape_elem = shape[0]
        self.shape_coor = shape[1]

    @property
    def asphericity(self):
        return _asphericity(self.shape_elem, self.shape_coor)

    @property
    def acylidricity(self):
        return _acylidricity(self.shape_elem, self.shape_coor)

    @property
    def relative_shape_anisotropy(self):
        return _relative_shape_anisotropy(self.shape_elem, self.shape_coor)

    @property
    def inertia_tensor(self):
        return get_inertia_tensor(self.shape_elem, self.shape_coor)

    @property
    def gyration_tensor(self):
        return get_gyration_tensor(self.shape_elem, self.shape_coor)

    #def plot3Dscatter(self):
    #    fig = pyplot.figure()#
    #    ax = Axes3D(fig)
    #    ax.scatter(
    #        self.shape_coor[:, 0], self.shape_coor[:, 1], self.shape_coor[:, 2]
    #    )
    #    pyplot.show()
    #    return fig


class Pore(Shape):
    def __init__(self, elements, coordinates, shape=False, **kwargs):
        self._elements, self._coordinates = elements, coordinates
        self.diameter, self.closest_atom = pore_diameter(elements, coordinates)
        self.spherical_volume = sphere_volume(self.diameter / 2)
        self.centre_coordinates = center_of_mass(elements, coordinates)
        self.optimised = False

    def optimise(self, **kwargs):
        (self.diameter, self.closest_atom,
         self.centre_coordinates) = opt_pore_diameter(self._elements,
                                                      self._coordinates,
                                                      **kwargs)
        self.spherical_volume = sphere_volume(self.diameter / 2)
        self.optimised = True

    def get_shape(self):
        super().__init__(
            calculate_pore_shape(self._elements, self._coordinates)
        )

    def reset(self):
        self.__init__(self._elements, self._coordinates)


class Window:
    def __init__(self, window, key, elements, coordinates, com_adjust):
        self.raw_data = window
        self.index = key
        self.mol_coordinates = coordinates
        self.mol_elements = elements
        self.com_correction = com_adjust
        self.shape = None

    def calculate_diameter(self, **kwargs):
        diameter = calculate_window_diameter(
            self.raw_data, self.mol_elements, self.mol_coordinates, **kwargs
        )
        return diameter

    def get_centre_of_mass(self, **kwargs):
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

    def shape_convexhull(self):
        hull = ConvexHull(self.shape)
        verticesx = np.append(
            self.shape[hull.vertices, 0], self.shape[hull.vertices, 0][0]
        )
        verticesy = np.append(
            self.shape[hull.vertices, 1], self.shape[hull.vertices, 1][0]
        )
        return verticesx, verticesy


class Molecule(Shape):
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
        self._circumcircle(**kwargs)
        return self.properties

    def align_to_principal_axes(self):
        self.coordinates = align_principal_ax(self.elements, self.coordinates)
        self.aligned_to_principal_axes = True

    def get_pore(self):
        return Pore(self.elements, self.coordinates)

    def get_shape(self, **kwargs):
        super().__init__(self, elements, coordinates)

    def get_windows(self, **kwargs):
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
        else:
            self.properties.update(
                {'windows': {'diameters': None,  'centre_of_mass': None, }}
            )
        return windows

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


class MolecularSystem(object):
    def __init__(self):
        self._Input = Input()
        self._Output = Output()
        self.system_id = 0

    @classmethod
    def load_file(cls, filepath):
        obj = cls()
        obj.system = obj._Input.load_file(filepath)
        obj.filename = os.path.basename(filepath)
        obj.system_id = obj.filename.split(".")[0]
        obj.name, ext = os.path.splitext(obj.filename)
        return obj

    @classmethod
    def load_rdkit_mol(cls, mol):
        obj = cls()
        obj.system = obj._Input.load_rdkit_mol(mol)
        return obj

    @classmethod
    def load_system(cls, dict_, system_id='system'):
        obj = cls()
        obj.system = dict_
        obj.system_id = system_id
        return obj

    def rebuild_system(self, override=False, **kwargs):
        # First we create a 3x3x3 supercell with the initial unit cell in the
        # centre and the 26 unit cell translations around to provide all the
        # atom positions necessary for the molecules passing through periodic
        # boundary reconstruction step.
        supercell_333 = create_supercell(self.system, **kwargs)
        #smolsys = self.load_system(supercell_333, self.system_id + '_311')
        #smolsys.dump_system(override=True)
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
            return rebuild_system

    def swap_atom_keys(self, swap_dict, dict_key='atom_ids'):
        """
        Swap atom_key for atom_key in system's 'elements' array.

        Parameters
        ----------
        swap_dict: dict
            A dictionary containg atom keys (dictionary's keys) to be swapped
            with corresponding atom keys (dictionary's keys arguments).

        dict_key: str (default='elements')
            A key in MolecularSystem().system dictionary to perform the
            atom keys swapping operation on.

        Modifies
        --------
        system['elements']: array
            Replaces every occurance of dictionary key with dictionary key's
            argument in the 'elements' array of the MolecularSystem().system's
            dictionary.

        Returns
        -------
        None: NoneType

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
        Decipheres forcefield's keys for their periodic elements equivalents.

        This function runs decipher_atom_key() function on every atom key in
        the system['atom_keys'] array and substitutes the return of this
        function at the corresponding position of system['elements'] array.

        The supported forcfields is OPLS and also the DL_F notation
        (see User's Guide) with keywords allowed:
        'OPLS', 'OPLS2005', 'OPLSAA', 'OPLS3' and 'DLF', 'DL_F'.

        Parameters
        ----------
        forcefield: str
            The forcefield used to decipher the atom keys. This parameter is
            not case sensitive.

        Modifies
        --------
        system['elements']
            It substitutes the string objects in this array for the return
            string of the decipher_atom_key() for each atom key in
            system['atom_keys'] array equvalent.

        Returns
        -------
        None: NoneType

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
        return Molecule(self.system, self.system_id, 0)

    def dump_system(self, filepath=None, modular=False, **kwargs):
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
