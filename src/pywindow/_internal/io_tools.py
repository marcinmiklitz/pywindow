"""Module contains classes for input/output processing."""

from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING

import numpy as np

from pywindow._internal.utilities import (
    decipher_atom_key,
    unit_cell_to_lattice_array,
)

if TYPE_CHECKING:
    import rdkit


class _CorruptedPDBFileError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message


class _CorruptedXYZFileError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message


class _NotADictionaryError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message


class _FileTypeError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message


class Input:
    """Class used to load and process input files."""

    def __init__(self) -> None:
        self._load_funcs = {
            ".xyz": self._read_xyz,
            ".pdb": self._read_pdb,
            ".mol": self._read_mol,
        }

    def load_file(self, filepath: pathlib.Path | str) -> dict:
        """This function opens any type of a readable file.

        It decomposes the file object into a list, for each line, of lists
        containing splitted line strings using space as a spacer.

        Parameters:
            filepath:
                The full path or a relative path to any type of file.

        Returns:
            :class:`dict`
                Returns a dictionary containing the molecular information
                extracted from the input files. This information will
                vary with file type and information stored in it.
                The data is sorted into lists that contain one feature
                for example key atom_id: [atom_id_1, atom_id_2]
                Over the process of analysis this dictionary will be updated
                with new data.
        """
        self.file_path = pathlib.Path(filepath)
        _, self.file_type = filepath.suffix()
        _, self.file_name = filepath.name()
        with filepath.open("r") as ffile:
            self.file_content = ffile.readlines()

        return self._load_funcs[self.file_type]()

    def load_rdkit_mol(self, mol: rdkit.Chem.Mol) -> dict:
        """Return molecular data from :class:`rdkit.Chem.Mol` object.

        Parameters:
            mol:
                A molecule object from RDKit.

        Returns:
            :class:`dict`
                A dictionary with ``elements`` and ``coordinates`` as keys
                containing molecular data extracted from
                :class:`rdkit.Chem.Mol` object.

        """
        self.system = {
            "elements": np.empty(mol.GetNumAtoms(), dtype=str),
            "coordinates": np.empty((mol.GetNumAtoms(), 3)),
        }
        for atom in mol.GetAtoms():
            atom_id = atom.GetIdx()
            atom_sym = atom.GetSymbol()
            x, y, z = mol.GetConformer().GetAtomPosition(atom_id)
            self.system["elements"][atom_id] = atom_sym
            self.system["coordinates"][atom_id] = x, y, z
        return self.system

    def _read_xyz(self) -> dict:
        try:
            self.system = {}
            self.file_remarks = self.file_content[1]
            self.system["elements"] = np.array(
                [i.split()[0] for i in self.file_content[2:]]
            )
            self.system["coordinates"] = np.array(
                [
                    [float(j[0]), float(j[1]), float(j[2])]
                    for j in [i.split()[1:] for i in self.file_content[2:]]
                ]
            )

        except IndexError:
            msg = (
                "The XYZ file is corrupted in some way. For example, an empty "
                "line at the end etc. or it is a trajectory. If the latter is "
                "the case, please use `trajectory` module, otherwise fix it."
            )
            raise _CorruptedXYZFileError(msg) from None
        return self.system

    def _read_pdb(self) -> dict:
        if sum([i.count("END ") for i in self.file_content]) > 1:
            msg = (
                "Multiple 'END' statements were found in this PDB file."
                "If this is a trajectory, use a trajectory module, "
                "Otherwise, fix it."
            )
            raise _CorruptedPDBFileError(msg)
        self.system = {}
        self.system["remarks"] = [
            i for i in self.file_content if i[:6] == "REMARK"
        ]
        self.system["unit_cell"] = np.array(
            [
                float(x)
                for i in self.file_content
                for x in [
                    i[6:15],
                    i[15:24],
                    i[24:33],
                    i[33:40],
                    i[40:47],
                    i[47:54],
                ]
                if i[:6] == "CRYST1"
            ]
        )
        if self.system["unit_cell"].any():
            self.system["lattice"] = unit_cell_to_lattice_array(
                self.system["unit_cell"]
            )
        self.system["atom_ids"] = np.array(
            [
                i[12:16].strip()
                for i in self.file_content
                if i[:6] == "HETATM" or i[:6] == "ATOM  "
            ],
            dtype="<U8",
        )
        self.system["elements"] = np.array(
            [
                i[76:78].strip()
                for i in self.file_content
                if i[:6] == "HETATM" or i[:6] == "ATOM  "
            ],
            dtype="<U8",
        )
        self.system["coordinates"] = np.array(
            [
                [float(i[30:38]), float(i[38:46]), float(i[46:54])]
                for i in self.file_content
                if i[:6] == "HETATM" or i[:6] == "ATOM  "
            ]
        )
        return self.system

    def _read_mol(self) -> dict:
        """Read V3000 mol file."""
        self.system = {}
        if self.file_content[2] != "\n":
            self.system["remarks"] = self.file_content[2]
        file_body = [i.split() for i in self.file_content]
        elements = []
        coordinates = []
        atom_data = False
        for line in file_body:
            if len(line) > 2:  # noqa: PLR2004
                if line[2] == "END" and line[3] == "ATOM":
                    atom_data = False
                if atom_data is True:
                    elements.append(line[3])
                    coordinates.append(line[4:7])
                if line[2] == "BEGIN" and line[3] == "ATOM":
                    atom_data = True
        self.system["elements"] = np.array(elements)
        self.system["coordinates"] = np.array(coordinates, dtype=float)
        return self.system


class Output:
    """Class used to process and save output files."""

    def __init__(self) -> None:
        self.cwd = pathlib.Path.cwd()
        self._save_funcs = {
            "xyz": self._save_xyz,
            "pdb": self._save_pdb,
        }

    def dump2json(
        self,
        obj: dict,
        filepath: str | pathlib.Path,
        override: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Dump a dictionary into a JSON dictionary.

        Uses the json.dump() function.

        Parameters:
            obj:
                A dictionary to be dumpped as JSON file.

            filepath:
                The filepath for the dumped file.

            override:
                If True, any file in the filepath will be override.
                (default=False)
        """
        filepath = pathlib.Path(filepath)
        # We make sure that the object passed by the user is a dictionary.
        if isinstance(obj, dict):
            pass
        else:
            msg = "This function only accepts dictionaries as input"
            raise _NotADictionaryError(msg)
        # We check if the filepath has a json extenstion, if not we add it.
        if str(filepath[-4:]) == "json":
            pass
        else:
            filepath = ".".join((str(filepath), "json"))
        # First we check if the file already exists. If yes and the override
        # keyword is False (default), we will raise an exception. Otherwise
        # the file will be overwritten.
        if override is False:  # noqa: SIM102
            if filepath.isfile():
                msg = (
                    f"The file {filepath} already exists. Use a different "
                    "filepath, or set the 'override' kwarg to True."
                )
                raise FileExistsError(msg)
        # We dump the object to the json file. Additional kwargs can be passed.
        with filepath.open("w+") as json_file:
            json.dump(obj, json_file)

    def dump2file(
        self,
        obj: dict,
        filepath: str | pathlib.Path,
        override: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Dump a dictionary into a file. (Extensions: XYZ or PDB).

        Parameters:
            obj:
                A dictionary containing molecular information.

            filepath:
                The filepath for the dumped file.

            override:
                If True, any file in the filepath will be override.
                (default=False)
        """
        filepath = pathlib.Path(filepath)
        # First we check if the file already exists. If yes and the override
        # keyword is False (default), we will raise an exception. Otherwise
        # the file will be overwritten.
        if override is False and filepath.isfile():
            msg = (
                f"The file {filepath} already exists. "
                "Use a different filepath, "
                "or set the 'override' kwarg to True."
            )
            raise FileExistsError(msg)
        if str(filepath[-3:]) not in self._save_funcs:
            msg = (
                f"The {filepath[-3:]!s} file extension is "
                "not supported for dumping a MolecularSystem or a Molecule. "
                "Please use XYZ or PDB."
            )
            raise _FileTypeError(msg)
        self._save_funcs[str(filepath[-3:])](obj, filepath)

    def _save_xyz(self, system: dict, filepath: str | pathlib.Path) -> None:
        filepath = pathlib.Path(filepath)

        # Initial settings.
        settings = {
            "elements": "elements",
            "coordinates": "coordinates",
            "remark": " ",
            "decipher": False,
            "forcefield": None,
        }

        # Extract neccessary data.
        elements = system["elements"]
        coordinates = system["coordinates"]
        if settings["decipher"] is True:
            elements = np.array(
                [
                    decipher_atom_key(key, forcefield=settings["forcefield"])
                    for key in elements
                ]
            )
        string = "{:0d}\n{}\n".format(len(elements), str(settings["remark"]))
        for i, j in zip(elements, coordinates):
            string += "{} {:.2f} {:.2f} {:.2f}\n".format(i, *j)
        with filepath.open("w+") as file_:
            file_.write(string)

    def _save_pdb(self, system: dict, filepath: str | pathlib.Path) -> None:  # noqa: C901
        filepath = pathlib.Path(filepath)
        settings = {
            "atom_ids": "atom_ids",
            "elements": "elements",
            "coordinates": "coordinates",
            "cryst": "unit_cell",
            "connect": None,
            "remarks": None,
            "space_group": None,
            "resName": "MOL",
            "chainID": "A",
            "resSeq": 1,
            "decipher": False,
            "forcefield": None,
        }

        # We create initial string that we will gradually extend while we
        # process the data and in the end it will be written into a pdb file.
        string = "REMARK File generated using pyWINDOW."
        # Number of items (atoms) in the provided system.
        len_ = system[settings["atom_ids"]].shape[0]
        # We process the remarks, if any, given by the user (optional).
        if isinstance(settings["remarks"], (list, tuple)):
            # If a list or tuple of remarks each is written at a new line
            # with the REMARK prefix not to have to long remark line.
            for remark in settings["remarks"]:
                string = "\n".join([string, f"REMARK {remark}"])
        # Otherwise if it's a single string or an int/float we just write
        # it under single remark line, otherwise nothing happens.
        elif isinstance(settings["remarks"], (str, int, float)):
            remark = settings["remarks"]
            string = "\n".join([string, f"REMARK {remark}"])
        # If there is a unit cell (crystal data) provided we need to add it.
        if settings["cryst"] in system and system[settings["cryst"]].any():
            cryst_line = "CRYST1"
            cryst = system[settings["cryst"]]
            # The user have to provide the crystal data as a list/array
            # of six items containing unit cell edges lengths a, b and c
            # in x, y and z directions and three angles, or it can be.
            # Other options are not allowed for simplicity. It can convert
            # from the lattice array using function from utilities.
            for i in cryst[:3]:
                cryst_line = "".join([cryst_line, f"{i:9.3f}"])
            for i in cryst[3:]:
                cryst_line = "".join([cryst_line, f"{i:7.2f}"])
            # This is kind of messy, by default the data written in PDB
            # file should be P1 symmetry group therefore containing all
            # atom coordinates and not considering symmetry operations.
            # But, user can still define a space group if he wishes to.
            if settings["space_group"] is not None:
                space_group = settings["space_group"]
            else:
                space_group = "{}".format("P1")
            cryst_line = f"{cryst_line} {space_group}"
            # We add the unit cell parameters to the main string.
            string = f"{string}\n{cryst_line}"
        # For the sake of code readability we extract interesting data from the
        # system. Atom_ids are the atom ids written at the third column of a
        # PDB file and the user has here the freedom to use the forcefield
        # assigned ones. However, they have to specify it directly using the
        # atom_ids key. Otherwise, the 'elements' array from system object
        # will be used, that is also used for elements in the last column of
        # a PDB file. Other parameters like residue name (resName), chain id
        # (chainID) and residue sequence (resSeq) can be controlled by
        # appropriate parameter keyword passed to this function, Otherwise
        # the default values from settings dictionary are used.
        atom_ids = system[settings["atom_ids"]]
        elements = system[settings["elements"]]
        # If the 'elements' array of the system need deciphering atom keys this
        # is done if the user sets decipher to True. They can also provided
        # forcefield, otherwise it's None which equals to DLF.
        if settings["decipher"] is True:
            elements = np.array(
                [
                    decipher_atom_key(key, forcefield=settings["forcefield"])
                    for key in elements
                ]
            )
        coordinates = system[settings["coordinates"]]
        for i in range(len_):
            atom_line = f"ATOM  {i + 1:5d}"
            atom_id = f"{atom_ids[i].center(4):4}"
            resname = "{:3}".format(settings["resName"])
            chainid = settings["chainID"]
            atom_line = f"{atom_line} {atom_id} {resname} {chainid}"
            resseq = str(settings["resSeq"]).rjust(4)
            atom_line = f"{atom_line}{resseq}"
            coor = (
                f"{coordinates[i][0]:8.3f}{coordinates[i][1]:8.3f}"
                f"{coordinates[i][2]:8.3f}"
            )
            atom_line = f"{atom_line}    {coor}"
            big_space = "{}".format(" ".center(22))
            element = f"{elements[i].rjust(2):2}  "
            atom_line = f"{atom_line}{big_space}{element}"
            string = f"{string}\n{atom_line}"
        # The connectivity part is to be written after a function calculating
        # connectivity is finished
        # "Everything that has a beginning has an end" by Neo. :)
        string = f"{string}\nEND"
        # Check if .pdb extension is missing from filepath.
        if filepath[-4:].lower() != ".pdb":
            filepath = f"{filepath}.pdb"
        # Write the string to a a PDB file.
        with filepath.open("w+") as file:
            file.write(string)
