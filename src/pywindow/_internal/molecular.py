"""Defines :class:`MolecularSystem` and :class:`Molecule` classes.

This module is the most important part of the ``pywindow`` package, as it is
at the frontfront of the interaction with the user. The two main classes
defined here: :class:`MolecularSystem` and :class:`Molecule` are used to
store and analyse single molecules or assemblies of single molecules.

The :class:`MolecularSystem` is used as a first step to the analysis. It allows
to load data, to refine it (rebuild molecules in a periodic system, decipher
force field atom ids) and to extract single molecules for analysis as
:class:`Molecule` instances.

To get started see :class:`MolecularSystem`.

To get started with the analysis of Molecular Dynamic trajectories go to
:mod:`pywindow.trajectory`.

"""

from __future__ import annotations

import pathlib
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np

from pywindow._internal.io_tools import Input, Output
from pywindow._internal.utilities import (
    align_principal_ax,
    center_of_mass,
    create_supercell,
    decipher_atom_key,
    discrete_molecules,
    find_average_diameter,
    find_windows,
    max_dim,
    molecular_weight,
    opt_pore_diameter,
    pore_diameter,
    shift_com,
    sphere_volume,
    to_list,
)

if TYPE_CHECKING:
    import rdkit


class _MolecularSystemError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message


class _NotAModularSystemError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message


class Molecule:
    """Container for a single molecule.

    This class is meant for the analysis of single molecules, molecular pores
    especially. The object passed to this class should therefore be a finite
    and interconnected individuum.

    This class should not be initialised directly, but result from
    :func:`MolecularSystem.system_to_molecule()` or
    :func:`MolecularSystem.make_modular()`.

    Methods in :class:`Molecule` allow to calculate:

        1. The maximum diameter of a molecule.

        2. The average diameter of a molecule.

        3. The intrinsic void diameter of a molecule.

        4. The intrinsic void volume of a molecule.

        5. The optimised intrinsic void diameter of a molecule.

        6. The optimised intrinsic void volume of a molecule.

        7. The circular diameter of a window of a molecule.

    Attributes:
        mol : :class:`dict`
            The :attr:`Molecular.System.system` dictionary passed to the
            :class:`Molecule` which is esentially a container of the
            information that compose a molecular entity, such as the
            coordinates and atom ids and/or elements.

        no_of_atoms : :class:`int`
            The number of atoms in the molecule.

        elements : :class:`numpy.array`
            An array containing the elements, as strings, composing the
            molecule.

        atom_ids : :class:`numpy.array` (conditional)
            If the :attr:`Molecule.mol` contains 'atom_ids' keyword, the force
            field ids of the elements.

        coordinates : :class:`numpy.array`
            The x, y and z atomic Cartesian coordinates of all elements.

        parent_system : :class:`str`
            The :attr:`name` of :class:`MolecularSystem` passed to
            :class:`Molecule`.

        molecule_id : :class:`any`
            The molecule id passed when initialising :class:`Molecule`.

        properties : :class:`dict`
            A dictionary that is populated by the output of
            :class:`Molecule` methods.

    """

    def __init__(self, mol: dict, system_name: str, mol_id: int) -> None:  # type:ignore[type-arg]
        self._Output = Output()
        self.mol = mol
        self.no_of_atoms = len(mol["elements"])
        self.elements = mol["elements"]
        if "atom_ids" in mol:
            self.atom_ids = mol["atom_ids"]
        self.coordinates = mol["coordinates"]
        self.parent_system = system_name
        self.molecule_id = mol_id
        self.properties = {"no_of_atoms": self.no_of_atoms}
        self._windows = None

    @classmethod
    def load_rdkit_mol(
        cls,
        mol: rdkit.Chem.rdchem.Mol,
        system_name: str = "rdkit",
        mol_id: int = 0,
    ) -> Molecule:
        """Create a :class:`Molecule` from :class:`rdkit.Chem.rdchem.Mol`.

        To be used only by expert users.

        Parameters:
            mol : :class:`rdkit.Chem.rdchem.Mol`
                An RDKit molecule object.

        Returns:
            :class:`pywindow.Molecule`


        """
        return cls(Input().load_rdkit_mol(mol), system_name, mol_id)

    def full_analysis(self, ncpus: int = 1) -> dict:  # type:ignore[type-arg]
        """Perform a full structural analysis of a molecule.

        This invokes other methods:

            1. :attr:`molecular_weight()`

            2. :attr:`calculate_centre_of_mass()`

            3. :attr:`calculate_maximum_diameter()`

            4. :attr:`calculate_average_diameter()`

            5. :attr:`calculate_pore_diameter()`

            6. :attr:`calculate_pore_volume()`

            7. :attr:`calculate_pore_diameter_opt()`

            8. :attr:`calculate_pore_volume_opt()`

            9. :attr:`calculate_pore_diameter_opt()`

            10. :attr:`calculate_windows()`

        Parameters:
            ncpus : :class:`int`
                Number of CPUs used for the parallelised parts of
                :func:`pywindow.utilities.find_windows()`. (default=1=serial)

        Returns:
            :attr:`Molecule.properties`
                The updated :attr:`Molecule.properties` with returns of all
                used methods.

        """
        self.molecular_weight()
        self.calculate_centre_of_mass()
        self.calculate_maximum_diameter()
        self.calculate_average_diameter()
        self.calculate_pore_diameter()
        self.calculate_pore_volume()
        self.calculate_pore_diameter_opt()
        self.calculate_pore_volume_opt()
        self.calculate_windows(ncpus=ncpus)

        return self.properties

    def _align_to_principal_axes(self, align_molsys: bool = False) -> None:  # noqa: FBT001, FBT002
        if align_molsys:
            raise NotImplementedError
            # self.coordinates[0] = align_principal_ax_all(
            #     self.elements, self.coordinates

        self.coordinates[0] = align_principal_ax(
            self.elements, self.coordinates
        )
        self.aligned_to_principal_axes = True

    def calculate_centre_of_mass(self) -> np.ndarray:  # type:ignore[type-arg]
        """Return the xyz coordinates of the centre of mass of a molecule.

        Returns:
            The centre of mass of the molecule.

        """
        self.centre_of_mass = center_of_mass(self.elements, self.coordinates)
        self.properties["centre_of_mass"] = self.centre_of_mass  # type:ignore[assignment]
        return self.centre_of_mass

    def calculate_maximum_diameter(self) -> float:
        """Return the maximum diamension of a molecule.

        Returns:
            The maximum dimension of the molecule.

        """
        self.maxd_atom_1, self.maxd_atom_2, self.maximum_diameter = max_dim(
            self.elements, self.coordinates
        )
        self.properties["maximum_diameter"] = {  # type:ignore[assignment]
            "diameter": self.maximum_diameter,
            "atom_1": int(self.maxd_atom_1),
            "atom_2": int(self.maxd_atom_2),
        }
        return self.maximum_diameter

    def calculate_average_diameter(self) -> float:
        """Return the average diamension of a molecule.

        Returns:
            The average dimension of the molecule.

        """
        self.average_diameter = find_average_diameter(
            self.elements, self.coordinates
        )
        self.properties["average_diameter"] = self.average_diameter  # type:ignore[assignment]
        return self.average_diameter

    def calculate_pore_diameter(self) -> float:
        """Return the intrinsic pore diameter.

        Returns:
            The intrinsic pore diameter.

        """
        self.pore_diameter, self.pore_closest_atom = pore_diameter(
            self.elements, self.coordinates
        )
        self.properties["pore_diameter"] = {  # type:ignore[assignment]
            "diameter": self.pore_diameter,
            "atom": int(self.pore_closest_atom),
        }
        return self.pore_diameter

    def calculate_pore_volume(self) -> float:
        """Return the intrinsic pore volume.

        Returns:
            The intrinsic pore volume.

        """
        self.pore_volume = sphere_volume(self.calculate_pore_diameter() / 2)
        self.properties["pore_volume"] = self.pore_volume  # type:ignore[assignment]
        return self.pore_volume

    def calculate_pore_diameter_opt(self) -> float:
        """Return the intrinsic pore diameter (for the optimised pore centre).

        Similarly to :func:`calculate_pore_diameter` this method returns the
        the intrinsic pore diameter, however, first a better approximation
        of the pore centre is found with optimisation.

        Returns:
            The intrinsic pore diameter.

        """
        (
            self.pore_diameter_opt,
            self.pore_opt_closest_atom,
            self.pore_opt_COM,
        ) = opt_pore_diameter(self.elements, self.coordinates)
        self.properties["pore_diameter_opt"] = {  # type:ignore[assignment]
            "diameter": self.pore_diameter_opt,
            "atom_1": int(self.pore_opt_closest_atom),
            "centre_of_mass": self.pore_opt_COM,
        }
        return self.pore_diameter_opt

    def calculate_pore_volume_opt(self) -> float:
        """Return the intrinsic pore volume (for the optimised pore centre).

        Similarly to :func:`calculate_pore_volume` this method returns the
        the volume intrinsic pore diameter, however, for the
        :func:`calculate_pore_diameter_opt` returned value.

        Returns:
            The intrinsic pore volume.

        """
        self.pore_volume_opt = sphere_volume(
            self.calculate_pore_diameter_opt() / 2
        )
        self.properties["pore_volume_opt"] = self.pore_volume_opt  # type:ignore[assignment]
        return self.pore_volume_opt

    def calculate_windows(self, ncpus: int = 1) -> np.ndarray | None:  # type:ignore[type-arg]
        """Return the diameters of all windows in a molecule.

        This function first finds and then measures the diameters of all the
        window in the molecule.

        Returns:
            An array of windows' diameters. Or, None, ff no windows were found.

        """
        windows = find_windows(
            self.elements,
            self.coordinates,
            processes=ncpus,
        )
        if windows is not None:
            self.properties.update(
                {
                    "windows": {  # type:ignore[dict-item]
                        "diameters": windows[0],
                        "centre_of_mass": windows[1],
                    }
                }
            )
            return windows[0]

        self.properties.update(
            {"windows": {"diameters": None, "centre_of_mass": None}}  # type:ignore[dict-item]
        )
        return None

    def shift_to_origin(self) -> None:
        """Shift a molecule to Origin.

        This function takes the molecule's coordinates and adjust them so that
        the centre of mass of the molecule coincides with the origin of the
        coordinate system.

        Returns:
            None : :class:`NoneType`

        """
        self.coordinates = shift_com(self.elements, self.coordinates)
        self._update()

    def molecular_weight(self) -> float:
        """Return the molecular weight of a molecule.

        Returns:
            :class:`float`
                The molecular weight of the molecule.

        """
        self.MW = molecular_weight(self.elements)
        return float(self.MW)

    def dump_properties_json(
        self,
        filepath: pathlib.Path | str | None = None,
        molecular: bool = False,  # noqa: FBT001, FBT002
        override: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Dump content of :attr:`Molecule.properties` to a JSON dictionary.

        Parameters:
            filepath:
                The filepath for the dumped file. If :class:`None`, the file is
                dumped localy with :attr:`molecule_id` as filename.
                (defualt=None)

            molecular:
                If False, dump only the content of :attr:`Molecule.properties`,
                if True, dump all the information about :class:`Molecule`.

            override:
                If True, any file in the filepath will be override.
                (default=False)
        """
        # We pass a copy of the properties dictionary.
        dict_obj = deepcopy(self.properties)
        # If molecular data is also required we update the dictionary.
        if molecular is True:
            dict_obj.update(self.mol)
        # If no filepath is provided we create one.
        if filepath is None:
            filepath = pathlib.Path(f"{self.parent_system}_{self.molecule_id}")
            filepath = pathlib.Path.cwd() / filepath
        filepath = pathlib.Path(filepath)
        # Dump the dictionary to json file.
        self._Output.dump2json(
            dict_obj,
            filepath,
            default=to_list,
            override=override,
        )

    def dump_molecule(
        self,
        filepath: pathlib.Path | str | None = None,
        include_coms: bool = False,  # noqa: FBT001, FBT002
        override: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Dump a :class:`Molecule` to a file (PDB or XYZ).

        For validation purposes an overlay of window centres and COMs can also
        be dumped as:

        He - for the centre of mass

        Ne - for the centre of the optimised cavity

        Ar - for the centres of each found window

        Parameters:
            filepath:
                The filepath for the dumped file. If :class:`None`, the file is
                dumped locally with :attr:`molecule_id` as filename.
                (default=None)

            include_coms:
                If True, dump also with an overlay of window centres and COMs.
                (default=False)

            override:
                If True, any file in the filepath will be override.
                (default=False)

        """
        # If no filepath is provided we create one.
        if filepath is None:
            filepath = pathlib.Path(
                f"{self.parent_system}_{self.molecule_id}.pdb"
            )
            filepath = pathlib.Path.cwd() / filepath
            filepath = f"{filepath}.pdb"

        filepath = pathlib.Path(filepath)
        # Check if there is an 'atom_ids' keyword in the self.mol dict.
        # Otherwise pass to the dump2file atom_ids='elements'.
        atom_ids_key = "elements" if "atom_ids" not in self.mol else "atom_ids"
        # Dump molecule into a file.
        # If coms are to be included additional steps are required.
        # First deepcopy the molecule
        if include_coms is True:
            mmol = deepcopy(self.mol)
            # add centre of mass (centre of not optimised pore) as 'He'.
            mmol["elements"] = np.concatenate(
                (mmol["elements"], np.array(["He"]))
            )
            if "atom_ids" not in self.mol:
                pass
            else:
                mmol["atom_ids"] = np.concatenate(
                    (mmol["atom_ids"], np.array(["He"]))
                )
            mmol["coordinates"] = np.concatenate(
                (
                    mmol["coordinates"],
                    np.array([self.properties["centre_of_mass"]]),
                )
            )
            # add centre of pore optimised as 'Ne'.
            mmol["elements"] = np.concatenate(
                (mmol["elements"], np.array(["Ne"]))
            )
            if "atom_ids" not in self.mol:
                pass
            else:
                mmol["atom_ids"] = np.concatenate(
                    (mmol["atom_ids"], np.array(["Ne"]))
                )
            mmol["coordinates"] = np.concatenate(
                (
                    mmol["coordinates"],
                    np.array(
                        [
                            self.properties["pore_diameter_opt"][  # type:ignore[index]
                                "centre_of_mass"
                            ]
                        ]
                    ),
                )
            )
            # add centre of windows as 'Ar'.
            if self.properties["windows"]["centre_of_mass"] is not None:  # type:ignore[index]
                range_ = range(
                    len(self.properties["windows"]["centre_of_mass"])  # type:ignore[index]
                )
                for com in range_:
                    mmol["elements"] = np.concatenate(
                        (mmol["elements"], np.array(["Ar"]))
                    )
                    if "atom_ids" not in self.mol:
                        pass
                    else:
                        mmol["atom_ids"] = np.concatenate(
                            (mmol["atom_ids"], np.array([f"Ar{com + 1}"]))
                        )
                    mmol["coordinates"] = np.concatenate(
                        (
                            mmol["coordinates"],
                            np.array(
                                [
                                    self.properties["windows"][  # type:ignore[index]
                                        "centre_of_mass"
                                    ][com]
                                ]
                            ),
                        )
                    )
            self._Output.dump2file(
                mmol,
                filepath,
                atom_ids_key=atom_ids_key,
                override=override,
            )

        else:
            self._Output.dump2file(
                self.mol,
                filepath,
                atom_ids_key=atom_ids_key,
                override=override,
            )

    def _update(self) -> None:
        self.mol["coordinates"] = self.coordinates
        self.calculate_centre_of_mass()
        self.calculate_pore_diameter_opt()


class MolecularSystem:
    """Container for the molecular system.

    To load input and initialise :class:`MolecularSystem`, one of the
    :class:`MolecularSystem` classmethods (:func:`load_file()`,
    :func:`load_rdkit_mol()` or :func:`load_system()`) should be used.
    :class:`MolecularSystem` **should not be initialised by itself.**

    Examples:
        1. Using file as an input:

        .. code-block:: python

            pywindow.MolecularSystem.load_file("filepath")

        2. Using RDKit molecule object as an input:

        .. code-block:: python

            pywindow.MolecularSystem.load_rdkit_mol(rdkit.Chem.rdchem.Mol)

        3. Using a dictionary (or another :attr:`MoleculeSystem.system`) as
        input:

        .. code-block:: python

            pywindow.MolecularSystem.load_system({...})

    Attributes:
        system_id : :class:`str` or :class:`int`
            The input filename or user defined.

        system : :class:`dict`
            A dictionary containing all the information extracted from input.

        molecules:
            A dictionary containing all the returned :class:`Molecule` s after
            using :func:`make_modular()`.

    """

    def __init__(self) -> None:
        self._Input = Input()
        self._Output = Output()
        self.system_id: str | int = 0
        self.system: dict = {}  # type: ignore[type-arg]
        self.molecules: dict[int | str, Molecule] = {}

    @classmethod
    def load_file(cls, filepath: pathlib.Path | str) -> MolecularSystem:
        """Create a :class:`MolecularSystem` from an input file.

        Recognized input file formats: XYZ, PDB and MOL (V3000).

        Parameters:
            filepath:
                The input's filepath.

        Returns:
            :class:`pywindow.MolecularSystem`


        """
        filepath = pathlib.Path(filepath)
        obj = cls()
        obj.system = obj._Input.load_file(filepath)
        obj.filename = filepath.name  # type: ignore[attr-defined]
        obj.system_id = obj.filename.split(".")[0]  # type: ignore[attr-defined]
        obj.name, ext = obj.filename.split(".")  # type: ignore[attr-defined]
        return obj

    @classmethod
    def load_rdkit_mol(cls, mol: rdkit.Chem.Mol) -> MolecularSystem:
        """Create a :class:`MolecularSystem` from :class:`rdkit.Chem.Mol`.

        Parameters:
            mol:
                An RDKit molecule object.

        Returns:
            :class:`pywindow.MolecularSystem`

        """
        obj = cls()
        obj.system = obj._Input.load_rdkit_mol(mol)
        return obj

    @classmethod
    def load_system(
        cls,
        dict_: dict,  # type: ignore[type-arg]
        system_id: str | int = "system",
    ) -> MolecularSystem:
        """Create a :class:`MolecularSystem` from a python :class:`dict`.

        As the loaded :class:`MolecularSystem` is storred as a :class:`dict` in
        the :class:`MolecularSystem.system` it can also be loaded directly from
        a :class:`dict` input. This feature is used by :mod:`trajectory` that
        extracts trajectory frames as dictionaries and returns them
        as :class:`MolecularSystem` objects through this classmethod.

        Parameters:
            dict_:
                A python dictionary.

            system_id:
                Inherited or user defined system id. (default='system')

        Returns:
            :class:`pywindow.MolecularSystem`


        """
        obj = cls()
        obj.system = dict_
        obj.system_id = system_id
        return obj

    def rebuild_system(self, override: bool = False) -> MolecularSystem:  # noqa: FBT001, FBT002
        """Rebuild molecules in molecular system.

        Parameters:
            override : :class:`bool`, optional (default=False)
                If False the rebuild molecular system is returned as a new
                :class:`MolecularSystem`, if True, the current
                :class:`MolecularSystem` is modified.

        """
        # First we create a 3x3x3 supercell with the initial unit cell in the
        # centre and the 26 unit cell translations around to provide all the
        # atom positions necessary for the molecules passing through periodic
        # boundary reconstruction step.
        supercell_333 = create_supercell(self.system)

        discrete = discrete_molecules(self.system, rebuild=supercell_333)
        # This function overrides the initial data for 'coordinates',
        # 'atom_ids', and 'elements' instances in the 'system' dictionary.
        coordinates = np.array([], dtype=np.float64).reshape(0, 3)
        atom_ids = np.array([])
        elements = np.array([])
        for i in discrete:
            coordinates = np.concatenate(
                [coordinates, i["coordinates"]], axis=0
            )
            atom_ids = np.concatenate([atom_ids, i["atom_ids"]], axis=0)
            elements = np.concatenate([elements, i["elements"]], axis=0)
        rebuild_system = {
            "coordinates": coordinates,
            "atom_ids": atom_ids,
            "elements": elements,
        }
        if override is True:
            self.system.update(rebuild_system)

        return self.load_system(rebuild_system)

    def swap_atom_keys(
        self,
        swap_dict: dict,  # type: ignore[type-arg]
        dict_key: str = "atom_ids",
    ) -> None:
        """Swap a force field atom id for another user-defined value.

        This modified all values in :attr:`MolecularSystem.system['atom_ids']`
        that match criteria.

        This function can be used to decipher a whole forcefield if an
        appropriate dictionary is passed to the function.

        Example:
            In this example all atom ids 'he' will be exchanged to 'H'.

            .. code-block:: python

                pywindow.MolecularSystem.swap_atom_keys({'he': 'H'})

        Parameters:
            swap_dict:
                A dictionary containg force field atom ids (keys) to be swapped
                with corresponding values (keys' arguments).

            dict_key:
                A key in :attr:`MolecularSystem.system` dictionary to perform
                the atom keys swapping operation on. (default='atom_ids')

        Returns:
            None : :class:`NoneType`

        """
        # Similar situation to the one from decipher_atom_keys function.
        if "atom_ids" not in self.system:
            dict_key = "elements"
        for atom_key in range(len(self.system[dict_key])):
            for key, value in swap_dict.items():
                if self.system[dict_key][atom_key] == key:
                    self.system[dict_key][atom_key] = value

    def decipher_atom_keys(
        self, forcefield: str = "DLF", dict_key: str = "atom_ids"
    ) -> None:
        """Decipher force field atom ids.

        This takes all values in :attr:`MolecularSystem.system['atom_ids']`
        that match force field type criteria and creates
        :attr:`MolecularSystem.system['elements']` with the corresponding
        periodic table of elements equivalents.

        If a forcefield is not supported by this method, the
        :func:`MolecularSystem.swap_atom_keys()` can be used instead.

        DLF stands for DL_F notation.

        See: C. W. Yong, Descriptions and Implementations of DL_F Notation: A
        Natural Chemical Expression System of Atom Types for Molecular
        Simulations, J. Chem. Inf. Model., 2016, 56, 1405-1409.

        Parameters:
            forcefield:
                The forcefield used to decipher atom ids. Allowed (not case
                sensitive): 'OPLS', 'OPLS2005', 'OPLSAA', 'OPLS3', 'DLF',
                'DL_F'. (default='DLF')

            dict_key:
                The :attr:`MolecularSystem.system` dictionary key to the array
                containing the force field atom ids. (default='atom_ids')

        Returns:
            None : :class:`NoneType`

        """
        # In case there is no 'atom_ids' key we try 'elements'. This is for
        # XYZ and MOL files mostly. But, we keep the dict_key keyword for
        # someone who would want to decipher 'elements' even if 'atom_ids' key
        # is present in the system's dictionary.
        if "atom_ids" not in self.system:
            dict_key = "elements"
        # I do it on temporary object so that it only finishes when successful
        temp = deepcopy(self.system[dict_key])
        for element in range(len(temp)):
            temp[element] = (
                f"{decipher_atom_key(temp[element], forcefield=forcefield)}"
            )
        self.system["elements"] = temp

    def make_modular(self, rebuild: bool = False) -> None:  # noqa: FBT001, FBT002
        """Find and return all :class:`Molecule` s in :class:`MolecularSystem`.

        This function populates :attr:`MolecularSystem.molecules` with
        :class:`Molecule` s.

        Parameters:
            rebuild:
                If True, run first the :func:`rebuild_system()`.

        Returns:
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
            self.molecules[i] = Molecule(
                mol=dis[i],
                system_name=str(self.system_id),
                mol_id=i,
            )

    def system_to_molecule(self) -> Molecule:
        """Return :class:`MolecularSystem` as a :class:`Molecule` directly.

        Only to be used conditionally, when the :class:`MolecularSystem` is a
        discrete molecule and no input pre-processing is required.

        Returns:
            :class:`pywindow.Molecule`

        """
        return Molecule(
            mol=self.system, system_name=str(self.system_id), mol_id=0
        )

    def dump_system(
        self,
        filepath: pathlib.Path | str | None = None,
        modular: bool = False,  # noqa: FBT001, FBT002
        override: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Dump a :class:`MolecularSystem` to a file (PDB or XYZ).

        Parameters:
            filepath:
                The filepath for the dumped file. If :class:`None`, the file is
                dumped localy with :attr:`system_id` as filename.
                (defualt=None)

            modular:
                If False, dump the :class:`MolecularSystem` as in
                :attr:`MolecularSystem.system`, if True, dump the
                :class:`MolecularSystem` as catenated :class:Molecule objects
                from :attr:`MolecularSystem.molecules`

            override:
                If True, any file in the filepath will be override.
                (default=False)

        """
        # If no filepath is provided we create one.
        if filepath is None:
            filepath = pathlib.Path.cwd() / f"{self.system_id}.pdb"

        filepath = pathlib.Path(filepath)
        # If modular is True substitute the molecular data for modular one.
        system_dict = deepcopy(self.system)
        if modular is True:
            elements = np.array([])
            atom_ids = np.array([])
            coor = np.array([]).reshape(0, 3)
            for mol in self.molecules.values():
                elements = np.concatenate((elements, mol.mol["elements"]))
                atom_ids = np.concatenate((atom_ids, mol.mol["atom_ids"]))
                coor = np.concatenate((coor, mol.mol["coordinates"]), axis=0)
            system_dict["elements"] = elements
            system_dict["atom_ids"] = atom_ids
            system_dict["coordinates"] = coor
        # Check if there is an 'atom_ids' keyword in the self.mol dict.
        # Otherwise pass to the dump2file atom_ids='elements'.
        # This is mostly for XYZ files and not deciphered trajectories.
        atom_ids_key = (
            "elements" if "atom_ids" not in system_dict else "atom_ids"
        )
        # Dump system into a file.
        self._Output.dump2file(
            system_dict,
            filepath,
            atom_ids_key=atom_ids_key,
            override=override,
        )

    def dump_system_json(
        self,
        filepath: pathlib.Path | str | None = None,
        modular: bool = False,  # noqa: FBT001, FBT002
        override: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Dump a :class:`MolecularSystem` to a JSON dictionary.

        The dumped JSON dictionary, with :class:`MolecularSystem`, can then be
        loaded through a JSON loader and then through :func:`load_system()`
        to retrieve a :class:`MolecularSystem`.

        Kwargs are passed to :func:`pywindow.io_tools.Output.dump2json()`.

        Parameters:
            filepath:
                The filepath for the dumped file. If :class:`None`, the file is
                dumped localy with :attr:`system_id` as filename.
                (defualt=None)

            modular:
                If False, dump the :class:`MolecularSystem` as in
                :attr:`MolecularSystem.system`, if True, dump the
                :class:`MolecularSystem` as catenated :class:Molecule objects
                from :attr:`MolecularSystem.molecules`

            override:
                If True, any file in the filepath will be override.
                (default=False)

        """
        # We pass a copy of the properties dictionary.
        dict_obj = deepcopy(self.system)
        # In case we want a modular system.
        if modular is True:
            try:
                if self.molecules:
                    pass
            except AttributeError:
                msg = (
                    "This system is not modular. Please, run first the "
                    "make_modular() function of this class."
                )
                raise _NotAModularSystemError(msg) from None
            dict_obj = {}
            for molecule, mol_ in self.molecules.items():
                dict_obj[molecule] = mol_.mol
        # If no filepath is provided we create one.
        if filepath is None:
            filepath = pathlib.Path.cwd() / f"{self.system_id}"
        filepath = pathlib.Path(filepath)

        # Dump the dictionary to json file.
        self._Output.dump2json(
            dict_obj,
            filepath,
            default=to_list,
            override=override,
        )
