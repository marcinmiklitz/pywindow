"""Module intended for the analysis of molecular dynamics trajectories.

The trajectory file (DL_POLY_C:HISTORY, PDB or XYZ) should be loaded with
the one of the corresponding classes (DLPOLY, PDB or XYZ, respectively).

Example:
-------
In this example a DL_POLY_C HISTORY trajectory file is loaded.

.. code-block:: python

    pywindow.trajectory.DLPOLY('path/to/HISTORY')

Then, each of the trajectory frames can be extracted and returned as a
:class:`pywindow.MolecularSystem` object for analysis. See
:mod:`pywindow.molecular` docstring for more information.

Alternatively, the analysis can be performed on a whole or a chunk of
the trajectory with the :func:`analysis()` function. The benefit is
that the analysis can be performed in parallel and the results stored as a
single JSON dictionary in a straightforward way. Also, the deciphering of the
force field atom ids and the rebuilding of molecules can be applied to each
frame in a consitent and automated manner. The downfall is that at the
moment it is not possible to choose the set of parameters that are being
calculated in the :class:`pywindow.Molecule` as the
:func:`pywindow.Molecule.full_analysis()` is invoked by default.
However, the computational cost of calculating majority of the structural
properties is miniscule and it is usually the
:func:`pywindow.MolecularSystem.rebuild_system()` step that is the
bottleneck.

"""

from __future__ import annotations

import pathlib
from contextlib import closing
from copy import deepcopy
from mmap import ACCESS_READ, mmap
from multiprocessing import Pool

import numpy as np

from pywindow._internal.io_tools import Output
from pywindow._internal.molecular import MolecularSystem
from pywindow._internal.utilities import (
    create_supercell,
    is_number,
    lattice_array_to_unit_cell,
    to_list,
)


class _ParallelAnalysisError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message


class _TrajectoryError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message


class _FormatError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message


class _FunctionError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message


def make_supercell(
    system: dict,
    matrix: np.ndarray,
    supercell: list[float] | None = None,
) -> MolecularSystem:
    """Return a supercell.

    This functions takes the input unitcell and creates a supercell of it that
    is returned as a new :class:`pywindow.MolecularSystem`.

    Parameters:
        system:
            The unit cell for creation of the supercell

        matrix:
            The unit cell parameters in form of a lattice.

        supercell:
            A list that specifies the size of the supercell in the a, b and c
            direction. (default=[1, 1, 1])

    Returns:
        Returns the created supercell as a new :class:`MolecularSystem`.

    """
    if supercell is None:
        supercell = [1, 1, 1]
    user_supercell = [[1, supercell[0]], [1, supercell[1]], [1, supercell[1]]]
    system = create_supercell(system, matrix, supercell=user_supercell)
    return MolecularSystem.load_system(system)


class DLPOLY:
    """A container for a DL_POLY_C type trajectory (HISTORY).

    This function takes a DL_POLY_C trajectory file and maps it for the
    binary points in the file where each frame starts/ends. This way the
    process is fast, as it not require loading the trajectory into computer
    memory. When a frame is being extracted, it is only this frame that gets
    loaded to the memory.

    Frames can be accessed individually and loaded as an unmodified string,
    returned as a :class:`pywindow.MolecularSystem` (and analysed),
    dumped as PDB or XYZ or JSON (if dumped as a
    :attr:`pywindow.MolecularSystem.system`)

    Attributes:
        filepath : :class:`str`
            The filepath.

        system_id : :class:`str`
            The system id inherited from the filename.

        frames : :class:`dict`
            A dictionary that is populated, on the fly, with the extracted
            frames.

        analysis_output : :class:`dict`
            A dictionary that is populated, on the fly, with the analysis
            output.

    """

    def __init__(self, filepath: pathlib.Path | str) -> None:
        # Image conventions - periodic boundary key.
        self._imcon = {
            0: "nonperiodic",
            1: "cubic",
            2: "orthorhombic",
            3: "parallelepiped",
            4: "truncated octahedral",
            5: "rhombic dodecahedral",
            6: "x-y parallelogram",
            7: "hexagonal prism",
        }
        # Trajectory key - content type.
        self._keytrj = {
            0: "coordinates",
            1: "coordinates and velocities",
            2: "coordinates, velocities and forces",
        }
        self.filepath = pathlib.Path(filepath)
        self.system_id = filepath.name()
        self.frames = {}
        self.analysis_output = {}
        # Check the history file at init, if no errors, proceed to mapping.
        self._check_HISTORY()
        # Map the trajectory file at init.
        self._map_HISTORY()

    def _map_history(self) -> None:
        """Map history."""
        self.trajectory_map = {}
        with self.filepath.open() as trajectory_file:
            with closing(
                mmap(trajectory_file.fileno(), 0, access=ACCESS_READ)
            ) as mapped_file:
                progress = 0
                line = 0
                frame = 0

                # We need to first process trajectory file's header.
                header_flag = True
                while progress <= len(mapped_file):
                    line = line + 1
                    # We read a binary data from a mapped file.
                    bline = mapped_file.readline()
                    # If the bline length equals zero we terminate.
                    # We reached end of the file but still add the last frame!
                    if len(bline) == 0:
                        self.trajectory_map[frame] = [frame_start, progress]  # noqa: F821
                        frame = frame + 1
                        break
                    # We need to decode byte line into an utf-8 string.
                    sline = bline.decode("utf-8").strip("\n").split()
                    # We extract map's byte coordinates for each frame
                    if header_flag is False and sline[0] == "timestep":
                        self.trajectory_map[frame] = [
                            frame_start,  # noqa: F821
                            progress,
                        ]
                        frame_start = progress
                        frame = frame + 1
                    # Here we extract the map's byte coordinates for the header
                    # And also the periodic system type needed for later.
                    if header_flag is True and sline[0] == "timestep":
                        self.trajectory_map["header"] = self._decode_head(
                            [0, progress]
                        )
                        frame_start = progress  # noqa: F841
                        header_flag = False
                    progress = progress + len(bline)
            self.no_of_frames = frame

    def _decode_head(self, header_coordinates: list[int]) -> list[int | str]:
        start, end = header_coordinates
        with (
            self.filepath.open() as trajectory_file,
            closing(
                mmap(trajectory_file.fileno(), 0, access=ACCESS_READ)
            ) as mapped_file,
        ):
            header = [
                i.split()
                for i in mapped_file[start:end].decode("utf-8").split("\n")
            ]
            header = [int(i) for i in header[1]]
        self.periodic_boundary = self._imcon[header[1]]
        self.content_type = self._keytrj[header[0]]
        self.no_of_atoms = header[2]
        return header

    def get_frames(  # noqa: C901
        self,
        frames: int | str | list[int] = "all",
        override: bool = False,  # noqa: FBT001, FBT002
    ) -> MolecularSystem | None:
        """Extract frames from the trajectory file.

        Depending on the passed parameters a frame, a list of particular
        frames, a range of frames (from, to), or all frames can be extracted
        with this function.

        Parameters:
            frames : :class:`int` or :class:`list` or :class:`tuple` or
                :class:`str`
                Specified frame (:class:`int`), or frames (:class:`list`), or
                range (:class:`tuple`), or `all`/`everything` (:class:`str`).
                (default=`all`)

            override : :class:`bool`
                If True, a frame already storred in :attr:`frames` can be
                override. (default=False)

            extract_data : :class:`bool`, optional
                If False, a frame is returned as a :class:`str` block as in the
                trajectory file. Ohterwise, it is extracted and returned as
                :class:`pywindow.MolecularSystem`. (default=True)

            swap_atoms : :class:`dict`, optional
                If this kwarg is passed with an appropriate dictionary a
                :func:`pywindow.MolecularSystem.swap_atom_keys()`
                will be applied to the extracted frame.

            forcefield : :class:`str`, optional
                If this kwarg is passed with appropriate forcefield keyword a
                :func:`pywindow.MolecularSystem.decipher_atom_keys()`
                will be applied to the extracted frame.

        Returns:
            If a single frame is extracted. None : :class:`NoneType` If more
            than one frame is extracted, the frames are returned to
            :attr:`frames`

        """
        if override is True:
            self.frames = {}
        if isinstance(frames, int):
            frame = self._get_frame(
                self.trajectory_map[frames],
                frames,
            )
            if frames not in self.frames:
                self.frames[frames] = frame
            return frame
        if isinstance(frames, list):
            for frame in frames:
                if frame not in self.frames:
                    self.frames[frame] = self._get_frame(
                        self.trajectory_map[frame],
                        frame,
                    )
        if isinstance(frames, tuple):
            for frame in range(frames[0], frames[1]):
                if frame not in self.frames:
                    self.frames[frame] = self._get_frame(
                        self.trajectory_map[frame],
                        frame,
                    )
        if isinstance(frames, str) and frames in ["all", "everything"]:
            for frame in range(self.no_of_frames):
                if frame not in self.frames:
                    self.frames[frame] = self._get_frame(
                        self.trajectory_map[frame],
                        frame,
                    )

        return None

    def _get_frame(
        self,
        frame_coordinates: list[int],
        frame_no: int,
    ) -> MolecularSystem:
        kwargs_ = {"extract_data": True}

        start, end = frame_coordinates
        with (
            self.filepath.open() as trajectory_file,
            closing(
                mmap(trajectory_file.fileno(), 0, access=ACCESS_READ)
            ) as mapped_file,
        ):
            if kwargs_["extract_data"] is False:
                return mapped_file[start:end].decode("utf-8")
            # [:-1] because the split results in last list empty.
            frame = [
                i.split()
                for i in mapped_file[start:end].decode("utf-8").split("\n")
            ][:-1]
            decoded_frame = self._decode_frame(frame)
            molsys = MolecularSystem.load_system(
                decoded_frame,
                "_".join([self.system_id, str(frame_no)]),
            )
            if "swap_atoms" in kwargs:
                molsys.swap_atom_keys(kwargs["swap_atoms"])
            if "forcefield" in kwargs:
                molsys.decipher_atom_keys(kwargs["forcefield"])
            return molsys

    def _decode_frame(self, frame: list) -> dict:  # noqa: C901, PLR0912
        frame_data = {
            "frame_info": {
                "nstep": int(frame[0][1]),
                "natms": int(frame[0][2]),
                "keytrj": int(frame[0][3]),
                "imcon": int(frame[0][4]),
                "tstep": float(frame[0][5]),
            }
        }
        start_line = 1
        if frame_data["frame_info"]["imcon"] in [1, 2, 3]:
            frame_data["lattice"] = np.array(frame[1:4], dtype=float).T
            frame_data["unit_cell"] = lattice_array_to_unit_cell(
                frame_data["lattice"]
            )
            start_line = 4
        # Depending on what the trajectory key is (see __init__) we need
        # to extract every second/ third/ fourth line for elements and coor.
        elements = []
        coordinates = []
        velocities = []
        forces = []
        for i in range(len(frame[start_line:])):
            i_ = i + start_line
            if frame_data["frame_info"]["keytrj"] == 0:
                if i % 2 == 0:
                    elements.append(frame[i_][0])
                if i % 2 == 1:
                    coordinates.append(frame[i_])
            if frame_data["frame_info"]["keytrj"] == 1:
                if i % 3 == 0:
                    elements.append(frame[i_][0])
                if i % 3 == 1:
                    coordinates.append(frame[i_])
                if i % 3 == 2:  # noqa: PLR2004
                    velocities.append(frame[i_])
            if frame_data["frame_info"]["keytrj"] == 2:  # noqa: PLR2004
                if i % 4 == 0:
                    elements.append(frame[i_][0])
                if i % 4 == 1:
                    coordinates.append(frame[i_])
                if i % 4 == 2:  # noqa: PLR2004
                    velocities.append(frame[i_])
                if i % 4 == 3:  # noqa: PLR2004
                    forces.append(frame[i_])
        frame_data["atom_ids"] = np.array(elements)
        frame_data["coordinates"] = np.array(coordinates, dtype=float)
        if velocities:
            frame_data["velocities"] = np.array(velocities, dtype=float)
        if forces:
            frame_data["forces"] = np.array(forces, dtype=float)
        return frame_data

    def analysis(  # noqa: C901, PLR0912, PLR0913
        self,
        frames: int | list | tuple | str = "all",
        ncpus: int = 1,
        override: bool = False,  # noqa: FBT001, FBT002
        swap_atoms: dict | None = None,  # noqa: ARG002
        forcefield: str | None = None,  # noqa: ARG002
        modular: bool | None = None,  # noqa: ARG002
        rebuild: bool | None = None,  # noqa: ARG002
    ) -> None:
        """Perform structural analysis on a frame/ set of frames.

        Depending on the passed parameters a frame, a list of particular
        frames, a range of frames (from, to), or all frames can be analysed
        with this function.

        The analysis is performed on each frame and each discrete molecule in
        that frame separately. The steps are as follows:

        1. A frame is extracted and returned as a :class:`MolecularSystem`.
        2. If `swap_atoms` is set the atom ids are swapped.
        3. If `forcefield` is set the atom ids are deciphered.
        4. If `rebuild` is set the molecules in the system are rebuild.
        5. Each discrete molecule is extracted as :class:`Molecule`
        6. Each molecule is analysed with :func:`Molecule.full_analysis()`
        7. Analysis output populates the :attr:`analysis_output` dictionary.

        As the analysis of trajectories often have to be unique, many options
        are conditional.

        A side effect of this function is that the analysed frames are also
        returned to the :attr:`frames` mimicking the behaviour of the
        :func:`get_frames()`.

        Parameters:
            frames : :class:`int` or :class:`list` or :class:`tuple` or
                :class:`str`
                Specified frame (:class:`int`), or frames (:class:`list`), or
                range (:class:`touple`), or `all`/`everything` (:class:`str`).
                (default='all')

            override : :class:`bool`
                If True, an output already storred in :attr:`analysis_output`
                can be override. (default=False)

            swap_atoms : :class:`dict`, optional
                If this kwarg is passed with an appropriate dictionary a
                :func:`pywindow.MolecularSystem.swap_atom_keys()`
                will be applied to the extracted frame.

            forcefield : :class:`str`, optional
                If this kwarg is passed with appropriate forcefield keyword a
                :func:`pywindow.MolecularSystem.decipher_atom_keys()`
                will be applied to the extracted frame.

            modular : :class:`bool`, optional
                If this kwarg is passed a
                :func:`pywindow.MolecularSystem.make_modular()`
                will be applied to the extracted frame. (default=False)

            rebuild : :class:`bool`, optional
                If this kwarg is passed a `rebuild=True` is passed to
                :func:`pywindow.MolecularSystem.make_modular()` that
                will be applied to the extracted frame. (default=False)

            ncpus : :class:`int`, optional
                If ncpus > 1, then the analysis is performed in parallel for
                the specified number of parallel jobs. Otherwise, it runs in
                serial. (default=1)

        Returns:
            None : :class:`NoneType`
                The function returns `None`, the analysis output is
                returned to :attr:`analysis_output` dictionary.

        """
        frames_for_analysis = []
        # First populate the frames_for_analysis list.
        if isinstance(frames, int):
            frames_for_analysis.append(frames)
        if isinstance(frames, list):
            for frame in frames:
                if isinstance(frame, int):
                    frames_for_analysis.append(frame)
                else:
                    msg = "The list should be populated with integers only."
                    raise _FunctionError(msg)
        if (
            isinstance(frames, tuple)
            and isinstance(frames[0], int)
            and isinstance(frames[1], int)
        ):
            for frame in range(frames[0], frames[1]):
                frames_for_analysis.append(frame)  # noqa: PERF402
            msg = (
                "The tuple should contain only two integers "
                "for the begining and the end of the frames range."
            )
            raise _FunctionError(msg)
        if isinstance(frames, str):
            if frames in ["all", "everything"]:
                for frame in range(self.no_of_frames):
                    frames_for_analysis.append(frame.copy())  # noqa: PERF401
            else:
                msg = "Didn't recognise the keyword. (see manual)"
                raise _FunctionError(msg)
        # The override keyword by default is False. So we check if any of the
        # frames were already analysed and if so we delete them from the list.
        # However, if the override is set to True, then we just proceed.
        if override is False:
            frames_for_analysis_new = []
            for frame in frames_for_analysis:
                if frame not in self.analysis_output:
                    frames_for_analysis_new.append(frame)  # noqa: PERF401
            frames_for_analysis = frames_for_analysis_new
        if ncpus == 1:
            for frame in frames_for_analysis:
                analysed_frame = self._analysis_serial(frame, 1)
                self.analysis_output[frame] = analysed_frame
        if ncpus > 1:
            self._analysis_parallel(frames_for_analysis, ncpus)

    def _analysis_serial(self, frame: dict, ncpus: int = 1) -> dict:  # noqa: C901
        settings = {
            "rebuild": False,
            "modular": False,
        }

        molecular_system = self._get_frame(
            self.trajectory_map[frame],
            frame,
            extract_data=True,
        )
        if settings["modular"] is True:
            molecular_system.make_modular(rebuild=settings["rebuild"])
            molecules = molecular_system.molecules
        else:
            molecules = {"0": molecular_system.system_to_molecule()}
        results = {}
        for molecule in molecules:
            mol = molecules[molecule]
            if "molsize" in settings:
                molsize = settings["molsize"]
                if isinstance(molsize, int) and mol.no_of_atoms == molsize:
                    results[molecule] = mol.full_analysis(_ncpus=ncpus)
                if isinstance(molsize, tuple) and isinstance(molsize[0], str):
                    if (
                        molsize[0] in ["bigger", "greater", "larger", "more"]
                        and mol.no_of_atoms > molsize[1]
                    ):
                        results[molecule] = mol.full_analysis(_ncpus=ncpus)
                    if (
                        molsize[0] in ["smaller", "less"]
                        and mol.no_of_atoms > molsize[1]
                    ):
                        results[molecule] = mol.full_analysis(_ncpus=ncpus)
                    if (
                        molsize[0] in ["not", "isnot", "notequal", "different"]
                        and mol.no_of_atoms != molsize[1]
                    ):
                        results[molecule] = mol.full_analysis(_ncpus=ncpus)
                    if (
                        molsize[0] in ["is", "equal", "exactly"]
                        and mol.no_of_atoms == molsize[1]
                    ):
                        results[molecule] = mol.full_analysis(_ncpus=ncpus)
                    if (
                        molsize[0] in ["between", "inbetween"]
                        and molsize[1] < mol.no_of_atoms < molsize[2]
                    ):
                        results[molecule] = mol.full_analysis(_ncpus=ncpus)
            else:
                results[molecule] = mol.full_analysis(_ncpus=ncpus)
        return results

    def _analysis_parallel_execute(self, frame: dict) -> None:  # noqa: C901
        settings = {
            "rebuild": False,
            "modular": False,
        }

        molecular_system = self._get_frame(
            self.trajectory_map[frame],
            frame,
            extract_data=True,
        )
        if settings["modular"] is True:
            molecular_system.make_modular(rebuild=settings["rebuild"])
            molecules = molecular_system.molecules
        else:
            molecules = {"0": molecular_system.system_to_molecule()}
        results = {}
        for molecule in molecules:
            mol = molecules[molecule]
            if "molsize" in settings:
                molsize = settings["molsize"]
                if isinstance(molsize, int) and mol.no_of_atoms == molsize:
                    results[molecule] = mol.full_analysis()
                if isinstance(molsize, tuple) and isinstance(molsize[0], str):
                    if (
                        molsize[0] in ["bigger", "greater", "larger", "more"]
                        and mol.no_of_atoms > molsize[1]
                    ):
                        results[molecule] = mol.full_analysis()
                    if (
                        molsize[0] in ["smaller", "less"]
                        and mol.no_of_atoms > molsize[1]
                    ):
                        results[molecule] = mol.full_analysis()
                    if (
                        molsize[0] in ["not", "isnot", "notequal", "different"]
                        and mol.no_of_atoms != molsize[1]
                    ):
                        results[molecule] = mol.full_analysis()
                    if (
                        molsize[0] in ["is", "equal", "exactly"]
                        and mol.no_of_atoms == molsize[1]
                    ):
                        results[molecule] = mol.full_analysis()
                    if (
                        molsize[0] in ["between", "inbetween"]
                        and molsize[1] < mol.no_of_atoms < molsize[2]
                    ):
                        results[molecule] = mol.full_analysis()
            else:
                results[molecule] = mol.full_analysis()
        return frame, results

    def _analysis_parallel(self, frames: list, ncpus: int) -> None:
        try:
            pool = Pool(processes=ncpus)
            parallel = [
                pool.apply_async(
                    self._analysis_parallel_execute, args=(frame,)
                )
                for frame in frames
            ]
            results = [p.get() for p in parallel if p.get()]
            pool.terminate()
            for i in results:
                self.analysis_output[i[0]] = i[1]
        except TypeError:
            pool.terminate()
            msg = "Parallel analysis failed."
            raise _ParallelAnalysisError(msg) from None

    def _check_history(self) -> None:
        self.check_log = ""
        line = 0
        binary_step = 0
        timestep = 0
        timestep_flag = "timestep"

        warning_1 = "No comment line is present as the file header.\n"
        warning_2 = (
            "Second header line is missing from the file that contains"
            " information on the system's periodicity and the type of the "
            "trajectory file.\n"
        )

        error_1 = "The trajectory is discontinous.\n"
        error_2 = "The file contains an empty line.\n"

        # We open the HISTORY trajectory file
        with (
            self.filepath.open() as trajectory_file,
            closing(
                mmap(trajectory_file.fileno(), 0, access=ACCESS_READ)
            ) as file_binary_map,
        ):
            # We use this binary mapping feature that instead of loading
            # the full file into memory beforehand it only
            # maps the content. Especially useful with enormous files
            while binary_step < len(file_binary_map):
                line += 1
                binary_line = file_binary_map.readline()
                binary_step = binary_step + len(binary_line)

                string_line = binary_line.decode("utf-8").strip("\n").split()

                # Warning 1
                if line == 1 and string_line[0] != "DLFIELD":
                    self.check_log = " ".join(
                        (
                            self.check_log,
                            f"Line {line}:",
                            warning_1,
                        )
                    )

                # Warning 2
                if line == 2 and len(string_line) != 3:  # noqa: PLR2004
                    self.check_log = " ".join(
                        (
                            self.check_log,
                            f"Line {line}:",
                            warning_2,
                        )
                    )

                # Error 1
                if string_line and string_line[0] == timestep_flag:
                    old_timestep = timestep
                    timestep = int(string_line[1])
                    if old_timestep > timestep:
                        error = " ".join(f"Line {line}:", error_1)
                        raise _TrajectoryError(error)

                # Error 2
                if len(string_line) == 0:
                    error = " ".join(f"Line {line}:", error_2)
                    raise _TrajectoryError(error)

    def save_analysis(
        self,
        filepath: str | pathlib.Path | None = None,
        override: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Dump the content of :attr:`analysis_output` as JSON dictionary."""
        filepath = pathlib.Path(filepath)
        # We pass a copy of the analysis attribute dictionary.
        dict_obj = deepcopy(self.analysis_output)
        # If no filepath is provided we create one.
        if filepath is None:
            filepath = f"{self.system_id}_pywindow_analysis"
            filepath = f"{pathlib.Path.cwd()}/{filepath}"
        # Dump the dictionary to json file.
        Output().dump2json(
            dict_obj,
            filepath,
            default=to_list,
            override=override,
        )

    def save_frames(  # noqa: C901, PLR0912
        self,
        frames: int | list | tuple | str,
        filepath: str | pathlib.Path | None = None,
        filetype: str = "pdb",
    ) -> None:
        filepath = pathlib.Path(filepath)
        settings = {
            "pdb": Output()._save_pdb,  # noqa: SLF001
            "xyz": Output()._save_xyz,  # noqa: SLF001
            "decipher": True,
            "forcefield": None,
        }

        if filetype.lower() not in settings:
            msg = f"The '{filetype}' file format is not supported"
            raise _FormatError(msg)
        frames_to_get = []
        if isinstance(frames, int):
            frames_to_get.append(frames)
        if isinstance(frames, list):
            frames_to_get = frames
        if isinstance(frames, tuple):
            for frame in range(frames[0], frames[1]):
                frames_to_get.append(frame)
        if isinstance(frames, str) and frames in ["all", "everything"]:
            for frame in range(self.no_of_frames):
                frames_to_get.append(frame)
        for frame in frames_to_get:
            if frame not in self.frames:
                _ = self.get_frames(frame)
        # If no filepath is provided we create one.
        if filepath is None:
            filepath = pathlib.Path.cwd() / str(self.system_id)

        for frame in frames_to_get:
            frame_molsys = self.frames[frame]
            if (
                settings["decipher"] is True
                and settings["forcefield"] is not None
            ):
                if "swap_atoms" in settings:
                    if isinstance(settings["swap_atoms"], dict):
                        frame_molsys.swap_atom_keys(settings["swap_atoms"])
                    else:
                        msg = (
                            "The swap_atom_keys function only accepts "
                            "'swap_atoms' argument in form of a dictionary."
                        )
                        raise _FunctionError(msg)
                frame_molsys.decipher_atom_keys(settings["forcefield"])
            ffilepath = "_".join((filepath, str(frame)))
            if "elements" not in frame_molsys.system:
                msg = (
                    "The frame (MolecularSystem object) needs to have "
                    "'elements' attribute within the system dictionary. "
                    "It is, therefore, neccessary that you set a decipher "
                    "keyword to True. (see manual)"
                )
                raise _FunctionError(msg)
            settings[filetype.lower()](frame_molsys.system, ffilepath)


class XYZ:
    """A container for an XYZ type trajectory.

    This function takes an XYZ trajectory file and maps it for the
    binary points in the file where each frame starts/ends. This way the
    process is fast, as it not require loading the trajectory into computer
    memory. When a frame is being extracted, it is only this frame that gets
    loaded to the memory.

    Frames can be accessed individually and loaded as an unmodified string,
    returned as a :class:`pywindow.MolecularSystem` (and analysed),
    dumped as PDB or XYZ or JSON (if dumped as a
    :attr:`pywindow.MolecularSystem.system`)

    Attributes:
        filepath : :class:`str`
            The filepath.

        filename : :class:`str`
            The filename.

        system_id : :class:`str`
            The system id inherited from the filename.

        frames : :class:`dict`
            A dictionary that is populated, on the fly, with the extracted
            frames.

        analysis_output : :class:`dict`
            A dictionary that is populated, on the fly, with the analysis
            output.

    """

    def __init__(self, filepath: pathlib.Path | str) -> None:
        self.filepath = pathlib.Path(filepath)
        self.filename = filepath.name()
        self.system_id = self.filename.split(".")[0]
        self.frames = {}
        self.analysis_output = {}
        # Map the trajectory file at init.
        self._map_trajectory()

    def _map_trajectory(self) -> None:
        """Return filepath as a class attribute."""
        self.trajectory_map = {}
        with self.filepath.open() as trajectory_file:
            with closing(
                mmap(trajectory_file.fileno(), 0, access=ACCESS_READ)
            ) as mapped_file:
                progress = 0
                line = 0
                frame = -1
                frame_start = 0
                while progress <= len(mapped_file):
                    line = line + 1
                    # We read a binary data from a mapped file.
                    bline = mapped_file.readline()
                    # If the bline length equals zero we terminate.
                    # We reached end of the file but still add the last frame!
                    if len(bline) == 0:
                        frame = frame + 1
                        self.trajectory_map[frame] = [frame_start, progress]
                        break
                    # We need to decode byte line into an utf-8 string.
                    sline = bline.decode("utf-8").strip("\n").split()
                    # We extract map's byte coordinates for each frame
                    if (
                        len(sline) == 1
                        and is_number(sline[0])
                        and progress > 0
                    ):
                        frame = frame + 1
                        self.trajectory_map[frame] = [frame_start, progress]
                        frame_start = progress
                    # Here we extract the map's byte coordinates for the header
                    # And also the periodic system type needed for later.
                    progress = progress + len(bline)
            self.no_of_frames = frame + 1

    def get_frames(  # noqa: C901
        self,
        frames: int | str | list[int] = "all",
        override: bool = False,  # noqa: FBT001, FBT002
    ) -> MolecularSystem | None:
        """Extract frames from the trajectory file.

        Depending on the passed parameters a frame, a list of particular
        frames, a range of frames (from, to), or all frames can be extracted
        with this function.

        Parameters:
            frames:
                Specified frame (:class:`int`), or frames (:class:`list`), or
                range (:class:`touple`), or `all`/`everything` (:class:`str`).
                (default=`all`)

            override : :class:`bool`
                If True, a frame already storred in :attr:`frames` can be
                override. (default=False)

            extract_data : :class:`bool`, optional
                If False, a frame is returned as a :class:`str` block as in the
                trajectory file. Ohterwise, it is extracted and returned as
                :class:`pywindow.MolecularSystem`. (default=True)

            swap_atoms : :class:`dict`, optional
                If this kwarg is passed with an appropriate dictionary a
                :func:`pywindow.MolecularSystem.swap_atom_keys()`
                will be applied to the extracted frame.

            forcefield : :class:`str`, optional
                If this kwarg is passed with appropriate forcefield keyword a
                :func:`pywindow.MolecularSystem.decipher_atom_keys()`
                will be applied to the extracted frame.

        Returns:
            :class:`pywindow.MolecularSystem`
                If a single frame is extracted. None : :class:`NoneType`
                If more than one frame is extracted, the frames are returned to
                :attr:`frames`

        """
        if override is True:
            self.frames = {}
        if isinstance(frames, int):
            frame = self._get_frame(
                self.trajectory_map[frames], frames, **kwargs
            )
            if frames not in self.frames:
                self.frames[frames] = frame
            return frame
        if isinstance(frames, list):
            for frame in frames:
                if frame not in self.frames:
                    self.frames[frame] = self._get_frame(
                        self.trajectory_map[frame], frame, **kwargs
                    )
        if isinstance(frames, tuple):
            for frame in range(frames[0], frames[1]):
                if frame not in self.frames:
                    self.frames[frame] = self._get_frame(
                        self.trajectory_map[frame], frame, **kwargs
                    )
        if isinstance(frames, str) and frames in ["all", "everything"]:
            for frame in range(self.no_of_frames):
                if frame not in self.frames:
                    self.frames[frame] = self._get_frame(
                        self.trajectory_map[frame], frame, **kwargs
                    )
        return None

    def _get_frame(
        self,
        frame_coordinates: list[int],
        frame_no: int,
    ) -> MolecularSystem:
        kwargs_ = {
            "extract_data": True,
        }

        start, end = frame_coordinates
        with (
            self.filepath.open() as trajectory_file,
            closing(
                mmap(trajectory_file.fileno(), 0, access=ACCESS_READ)
            ) as mapped_file,
        ):
            if kwargs_["extract_data"] is False:
                return mapped_file[start:end].decode("utf-8")
            # [:-1] because the split results in last list empty.
            frame = [
                i.split()
                for i in mapped_file[start:end].decode("utf-8").split("\n")
            ][:-1]
            decoded_frame = self._decode_frame(frame)
            molsys = MolecularSystem.load_system(
                decoded_frame,
                "_".join([self.system_id, str(frame_no)]),
            )
            if "swap_atoms" in kwargs:
                molsys.swap_atom_keys(kwargs["swap_atoms"])
            if "forcefield" in kwargs:
                molsys.decipher_atom_keys(kwargs["forcefield"])
            return molsys

    def _decode_frame(self, frame: list) -> dict:
        frame_data = {
            "frame_info": {
                "natms": int(frame[0][0]),
                "remarks": " ".join([*frame[1]]),
            }
        }
        start_line = 2
        elements = []
        coordinates = []
        for i in range(start_line, len(frame)):
            elements.append(frame[i][0])
            coordinates.append(frame[i][1:])
        frame_data["atom_ids"] = np.array(elements)
        frame_data["coordinates"] = np.array(coordinates, dtype=float)
        return frame_data

    def analysis(  # noqa: C901
        self,
        frames: int | list | tuple | str = "all",
        ncpus: int = 1,
        override: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Perform structural analysis on a frame/ set of frames.

        Depending on the passed parameters a frame, a list of particular
        frames, a range of frames (from, to), or all frames can be analysed
        with this function.

        The analysis is performed on each frame and each discrete molecule in
        that frame separately. The steps are as follows:

        1. A frame is extracted and returned as a :class:`MolecularSystem`.
        2. If `swap_atoms` is set the atom ids are swapped.
        3. If `forcefield` is set the atom ids are deciphered.
        4. If `rebuild` is set the molecules in the system are rebuild.
        5. Each discrete molecule is extracted as :class:`Molecule`
        6. Each molecule is analysed with :func:`Molecule.full_analysis()`
        7. Analysis output populates the :attr:`analysis_output` dictionary.

        As the analysis of trajectories often have to be unique, many options
        are conditional.

        A side effect of this function is that the analysed frames are also
        returned to the :attr:`frames` mimicking the behaviour of the
        :func:`get_frames()`.

        Parameters:
            frames:
                Specified frame (:class:`int`), or frames (:class:`list`), or
                range (:class:`touple`), or `all`/`everything` (:class:`str`).
                (default='all')

            override : :class:`bool`
                If True, an output already storred in :attr:`analysis_output`
                can be override. (default=False)

            swap_atoms : :class:`dict`, optional
                If this kwarg is passed with an appropriate dictionary a
                :func:`pywindow.MolecularSystem.swap_atom_keys()` will
                be applied to the extracted frame.

            forcefield : :class:`str`, optional
                If this kwarg is passed with appropriate forcefield keyword a
                :func:`pywindow.MolecularSystem.decipher_atom_keys()`
                will be applied to the extracted frame.

            modular : :class:`bool`, optional
                If this kwarg is passed a
                :func:`pywindow.MolecularSystem.make_modular()`
                will be applied to the extracted frame. (default=False)

            rebuild : :class:`bool`, optional
                If this kwarg is passed a `rebuild=True` is passed to
                :func:`pywindow.MolecularSystem.make_modular()` that
                will be applied to the extracted frame. (default=False)

            ncpus : :class:`int`, optional
                If ncpus > 1, then the analysis is performed in parallel for
                the specified number of parallel jobs. Otherwise, it runs in
                serial. (default=1)

        Returns:
            The function returns `None`, the analysis output is
            returned to :attr:`analysis_output` dictionary.

        """
        if override is True:
            self.analysis_output = {}
        if isinstance(frames, int):
            analysed_frame = self._analysis_serial(frames, ncpus)
            if frames not in self.analysis_output:
                self.analysis_output[frames] = analysed_frame
            return analysed_frame
        frames_for_analysis = []
        if isinstance(frames, list):
            for frame in frames:
                if frame not in self.analysis_output:
                    frames_for_analysis.append(frame)  # noqa: PERF401
        if isinstance(frames, tuple):
            for frame in range(frames[0], frames[1]):
                if frame not in self.analysis_output:
                    frames_for_analysis.append(frame)  # noqa: PERF401
        if isinstance(frames, str) and frames in ["all", "everything"]:
            for frame in range(self.no_of_frames):
                if frame not in self.analysis_output:
                    frames_for_analysis.append(frame)  # noqa: PERF401
        self._analysis_parallel(frames_for_analysis, ncpus)
        return None

    def _analysis_serial(self, frame: dict, ncpus: int = 1) -> dict:  # noqa: C901
        settings = {
            "rebuild": False,
            "modular": False,
        }

        molecular_system = self._get_frame(
            self.trajectory_map[frame],
            frame,
            extract_data=True,
        )
        if settings["modular"] is True:
            molecular_system.make_modular(rebuild=settings["rebuild"])
            molecules = molecular_system.molecules
        else:
            molecules = {"0": molecular_system.system_to_molecule()}
        results = {}
        for molecule in molecules:
            mol = molecules[molecule]
            if "molsize" in settings:
                molsize = settings["molsize"]
                if isinstance(molsize, int) and mol.no_of_atoms == molsize:
                    results[molecule] = mol.full_analysis(ncpus=ncpus)
                if isinstance(molsize, tuple) and isinstance(molsize[0], str):
                    if (
                        molsize[0] in ["bigger", "greater", "larger", "more"]
                        and mol.no_of_atoms > molsize[1]
                    ):
                        results[molecule] = mol.full_analysis(ncpus=ncpus)
                    if (
                        molsize[0] in ["smaller", "less"]
                        and mol.no_of_atoms > molsize[1]
                    ):
                        results[molecule] = mol.full_analysis(ncpus=ncpus)
                    if (
                        molsize[0] in ["not", "isnot", "notequal", "different"]
                        and mol.no_of_atoms != molsize[1]
                    ):
                        results[molecule] = mol.full_analysis(ncpus=ncpus)
                    if (
                        molsize[0] in ["is", "equal", "exactly"]
                        and mol.no_of_atoms == molsize[1]
                    ):
                        results[molecule] = mol.full_analysis(ncpus=ncpus)
                    if (
                        molsize[0] in ["between", "inbetween"]
                        and molsize[1] < mol.no_of_atoms < molsize[2]
                    ):
                        results[molecule] = mol.full_analysis(ncpus=ncpus)
            else:
                results[molecule] = mol.full_analysis(ncpus=ncpus)
        return results

    def _analysis_parallel_execute(self, frame: dict) -> None:  # noqa: C901
        settings = {
            "rebuild": False,
            "modular": False,
        }

        molecular_system = self._get_frame(
            self.trajectory_map[frame],
            frame,
            extract_data=True,
        )
        if settings["modular"] is True:
            molecular_system.make_modular(rebuild=settings["rebuild"])
            molecules = molecular_system.molecules
        else:
            molecules = {"0": molecular_system.system_to_molecule()}
        results = {}
        for molecule in molecules:
            mol = molecules[molecule]
            if "molsize" in settings:
                molsize = settings["molsize"]
                if isinstance(molsize, int) and mol.no_of_atoms == molsize:
                    results[molecule] = mol.full_analysis()
                if isinstance(molsize, tuple) and isinstance(molsize[0], str):
                    if (
                        molsize[0] in ["bigger", "greater", "larger", "more"]
                        and mol.no_of_atoms > molsize[1]
                    ):
                        results[molecule] = mol.full_analysis()
                    if (
                        molsize[0] in ["smaller", "less"]
                        and mol.no_of_atoms > molsize[1]
                    ):
                        results[molecule] = mol.full_analysis()
                    if (
                        molsize[0] in ["not", "isnot", "notequal", "different"]
                        and mol.no_of_atoms != molsize[1]
                    ):
                        results[molecule] = mol.full_analysis()
                    if (
                        molsize[0] in ["is", "equal", "exactly"]
                        and mol.no_of_atoms == molsize[1]
                    ):
                        results[molecule] = mol.full_analysis()
                    if (
                        molsize[0] in ["between", "inbetween"]
                        and molsize[1] < mol.no_of_atoms < molsize[2]
                    ):
                        results[molecule] = mol.full_analysis()
            else:
                results[molecule] = mol.full_analysis()
        return frame, results

    def _analysis_parallel(self, frames: list, ncpus: int) -> None:
        try:
            pool = Pool(processes=ncpus)
            parallel = [
                pool.apply_async(
                    self._analysis_parallel_execute, args=(frame,), kwds=kwargs
                )
                for frame in frames
            ]
            results = [p.get() for p in parallel if p.get()[1] is not None]
            pool.terminate()
            for i in results:
                self.analysis_output[i[0]] = i[1]
        except TypeError:
            pool.terminate()
            msg = "Parallel analysis failed."
            raise _ParallelAnalysisError(msg) from None

    def save_analysis(
        self,
        filepath: str | pathlib.Path | None = None,
    ) -> None:
        """Dump the content of :attr:`analysis_output` as JSON dictionary.

        Parameters:
            filepath:
                The filepath for the JSON file.

        Returns:
            None : :class:`NoneType`
        """
        # We pass a copy of the analysis attribute dictionary.
        dict_obj = deepcopy(self.analysis_output)
        # If no filepath is provided we create one.
        if filepath is None:
            filepath = f"{self.system_id}_pywindow_analysis"
            filepath = pathlib.Path.cwd() / filepath
        # Dump the dictionary to json file.
        Output().dump2json(dict_obj, filepath, default=to_list)

    def save_frames(  # noqa: C901, PLR0912
        self,
        frames: int | list | tuple | str,
        filepath: str | pathlib.Path | None = None,
        filetype: str = "pdb",
    ) -> None:
        filepath = pathlib.Path(filepath)

        settings = {
            "pdb": Output()._save_pdb,  # noqa: SLF001
            "xyz": Output()._save_xyz,  # noqa: SLF001
            "decipher": True,
            "forcefield": None,
        }

        if filetype.lower() not in settings:
            msg = f"The '{filetype}' file format is not supported"
            raise _FormatError(msg)
        frames_to_get = []
        if isinstance(frames, int):
            frames_to_get.append(frames)
        if isinstance(frames, list):
            frames_to_get = frames
        if isinstance(frames, tuple):
            for frame in range(frames[0], frames[1]):
                frames_to_get.append(frame)
        if isinstance(frames, str) and frames in ["all", "everything"]:
            for frame in range(self.no_of_frames):
                frames_to_get.append(frame)
        for frame in frames_to_get:
            if frame not in self.frames:
                _ = self.get_frames(frame)
        # If no filepath is provided we create one.
        if filepath is None:
            filepath = pathlib.Path.cwd() / str(self.system_id)

        for frame in frames_to_get:
            frame_molsys = self.frames[frame]
            if (
                settings["decipher"] is True
                and settings["forcefield"] is not None
            ):
                if "swap_atoms" in settings:
                    if isinstance(settings["swap_atoms"], dict):
                        frame_molsys.swap_atom_keys(settings["swap_atoms"])
                    else:
                        msg = (
                            "The swap_atom_keys function only accepts "
                            "'swap_atoms' argument in form of a dictionary."
                        )
                        raise _FunctionError(msg)
                frame_molsys.decipher_atom_keys(settings["forcefield"])
            ffilepath = "_".join((filepath, str(frame)))
            if "elements" not in frame_molsys.system:
                msg = (
                    "The frame (MolecularSystem object) needs to have "
                    "'elements' attribute within the system dictionary. "
                    "It is, therefore, neccessary that you set a decipher "
                    "keyword to True. (see manual)"
                )
                raise _FunctionError(msg)
            settings[filetype.lower()](frame_molsys.system, ffilepath)


class PDB:
    def __init__(self, filepath: pathlib.Path | str) -> None:
        """A container for an PDB type trajectory.

        This function takes an PDB trajectory file and maps it for the
        binary points in the file where each frame starts/ends. This way the
        process is fast, as it not require loading the trajectory into computer
        memory. When a frame is being extracted, it is only this frame that
        gets loaded to the memory.

        Frames can be accessed individually and loaded as an unmodified string,
        returned as a :class:`pywindow.MolecularSystem`
        (and analysed), dumped as PDB or XYZ or JSON (if dumped as a
        :attr:`pywindow.MolecularSystem.system`)

        Attributes:
            filepath : :class:`str`
                The filepath.

            filename : :class:`str`
                The filename.

            system_id : :class:`str`
                The system id inherited from the filename.

            frames : :class:`dict`
                A dictionary that is populated, on the fly, with the extracted
                frames.

            analysis_output : :class:`dict`
                A dictionary that is populated, on the fly, with the analysis
                output.

        """
        self.filepath = pathlib.Path(filepath)
        self.filename = filepath.name()
        self.system_id = self.filename.split(".")[0]
        self.frames = {}
        self.analysis_output = {}
        # Map the trajectory file at init.
        self._map_trajectory()

    def _map_trajectory(self) -> None:
        """Return filepath as a class attribute."""
        self.trajectory_map = {}
        with self.filepath.open() as trajectory_file:
            with closing(
                mmap(trajectory_file.fileno(), 0, access=ACCESS_READ)
            ) as mapped_file:
                progress = 0
                line = 0
                frame = -1
                frame_start = 0
                while progress <= len(mapped_file):
                    line = line + 1
                    # We read a binary data from a mapped file.
                    bline = mapped_file.readline()
                    # If the bline length equals zero we terminate.
                    # We reached end of the file but still add the last frame!
                    if len(bline) == 0:
                        frame = frame + 1
                        if progress - frame_start > 10:  # noqa: PLR2004
                            self.trajectory_map[frame] = [
                                frame_start,
                                progress,
                            ]
                        break
                    # We need to decode byte line into an utf-8 string.
                    sline = bline.decode("utf-8").strip("\n").split()
                    # We extract map's byte coordinates for each frame
                    if len(sline) == 1 and sline[0] == "END":
                        frame = frame + 1
                        self.trajectory_map[frame] = [frame_start, progress]
                        frame_start = progress
                    # Here we extract the map's byte coordinates for the header
                    # And also the periodic system type needed for later.
                    progress = progress + len(bline)
            self.no_of_frames = frame

    def get_frames(  # noqa: C901
        self,
        frames: int | str | list[int] = "all",
        override: bool = False,  # noqa: FBT001, FBT002
    ) -> MolecularSystem | None:
        """Extract frames from the trajectory file.

        Depending on the passed parameters a frame, a list of particular
        frames, a range of frames (from, to), or all frames can be extracted
        with this function.

        Parameters:
            frames : :class:`int` or :class:`list` or :class:`touple` or
                :class:`str`
                Specified frame (:class:`int`), or frames (:class:`list`), or
                range (:class:`touple`), or `all`/`everything` (:class:`str`).
                (default=`all`)

            override : :class:`bool`
                If True, a frame already storred in :attr:`frames` can be
                override. (default=False)

            extract_data : :class:`bool`, optional
                If False, a frame is returned as a :class:`str` block as in the
                trajectory file. Ohterwise, it is extracted and returned as
                :class:`pywindow.MolecularSystem`. (default=True)

            swap_atoms : :class:`dict`, optional
                If this kwarg is passed with an appropriate dictionary a
                :func:`pywindow.MolecularSystem.swap_atom_keys()` will
                be applied to the extracted frame.

            forcefield : :class:`str`, optional
                If this kwarg is passed with appropriate forcefield keyword a
                :func:`pywindow.MolecularSystem.decipher_atom_keys()`
                will be applied to the extracted frame.

        Returns:
            :class:`pywindow.MolecularSystem`
                If a single frame is extracted.
                None : :class:`NoneType`
                If more than one frame is extracted, the frames are returned to
                :attr:`frames`

        """
        if override is True:
            self.frames = {}
        if isinstance(frames, int):
            frame = self._get_frame(self.trajectory_map[frames], frames)
            if frames not in self.frames:
                self.frames[frames] = frame
            return frame
        if isinstance(frames, list):
            for frame in frames:
                if frame not in self.frames:
                    self.frames[frame] = self._get_frame(
                        self.trajectory_map[frame], frame
                    )
        if isinstance(frames, tuple):
            for frame in range(frames[0], frames[1]):
                if frame not in self.frames:
                    self.frames[frame] = self._get_frame(
                        self.trajectory_map[frame], frame
                    )
        if isinstance(frames, str) and frames in ["all", "everything"]:
            for frame in range(self.no_of_frames):
                if frame not in self.frames:
                    self.frames[frame] = self._get_frame(
                        self.trajectory_map[frame], frame
                    )
        return None

    def _get_frame(
        self,
        frame_coordinates: list[int],
        frame_no: int,
    ) -> MolecularSystem:
        kwargs_ = {"extract_data": True}

        start, end = frame_coordinates
        with (
            self.filepath.open() as trajectory_file,
            closing(
                mmap(trajectory_file.fileno(), 0, access=ACCESS_READ)
            ) as mapped_file,
        ):
            if kwargs_["extract_data"] is False:
                return mapped_file[start:end].decode("utf-8")
            # In case of PDB we do not split lines!
            frame = mapped_file[start:end].decode("utf-8").split("\n")
            decoded_frame = self._decode_frame(frame)
            molsys = MolecularSystem.load_system(
                decoded_frame,
                "_".join([self.system_id, str(frame_no)]),
            )
            if "swap_atoms" in kwargs:
                molsys.swap_atom_keys(kwargs["swap_atoms"])
            if "forcefield" in kwargs:
                molsys.decipher_atom_keys(kwargs["forcefield"])
            return molsys

    def _decode_frame(self, frame: list) -> dict:
        frame_data = {}
        elements = []
        coordinates = []
        for i in range(len(frame)):
            if frame[i][:6] == "REMARK":
                if "REMARKS" not in frame_data:
                    frame_data["REMARKS"] = []
                frame_data["REMARKS"].append(frame[i][6:])
            if frame[i][:6] == "CRYST1":
                cryst = np.array(
                    [
                        frame[i][6:15],
                        frame[i][15:24],
                        frame[i][24:33],
                        frame[i][33:40],
                        frame[i][40:47],
                        frame[i][47:54],
                    ],
                    dtype=float,
                )
                # This is in case of nonperiodic systems, often they have
                # a,b,c unit cell parameters as 0,0,0.
                if sum(cryst[0:3]) != 0:
                    frame_data["CRYST1"] = cryst
            if frame[i][:6] in ["HETATM", "ATOM  "]:
                elements.append(frame[i][12:16].strip())
                coordinates.append(
                    [frame[i][30:38], frame[i][38:46], frame[i][46:54]]
                )
        frame_data["atom_ids"] = np.array(elements, dtype="<U8")
        frame_data["coordinates"] = np.array(coordinates, dtype=float)
        return frame_data

    def analysis(  # noqa: C901
        self,
        frames: int | list | tuple | str = "all",
        ncpus: int = 1,
        override: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Perform structural analysis on a frame/ set of frames.

        Depending on the passed parameters a frame, a list of particular
        frames, a range of frames (from, to), or all frames can be analysed
        with this function.

        The analysis is performed on each frame and each discrete molecule in
        that frame separately. The steps are as follows:

        1. A frame is extracted and returned as a :class:`MolecularSystem`.
        2. If `swap_atoms` is set the atom ids are swapped.
        3. If `forcefield` is set the atom ids are deciphered.
        4. If `rebuild` is set the molecules in the system are rebuild.
        5. Each discrete molecule is extracted as :class:`Molecule`
        6. Each molecule is analysed with :func:`Molecule.full_analysis()`
        7. Analysis output populates the :attr:`analysis_output` dictionary.

        As the analysis of trajectories often have to be unique, many options
        are conditional.

        A side effect of this function is that the analysed frames are also
        returned to the :attr:`frames` mimicking the behaviour of the
        :func:`get_frames()`.

        Parameters:
            frames:
                Specified frame (:class:`int`), or frames (:class:`list`), or
                range (:class:`touple`), or `all`/`everything` (:class:`str`).
                (default='all')

            override : :class:`bool`
                If True, an output already storred in :attr:`analysis_output`
                can be override. (default=False)

            swap_atoms : :class:`dict`, optional
                If this kwarg is passed with an appropriate dictionary a
                :func:`pywindow.MolecularSystem.swap_atom_keys()` will
                be applied to the extracted frame.

            forcefield : :class:`str`, optional
                If this kwarg is passed with appropriate forcefield keyword a
                :func:`pywindow.MolecularSystem.decipher_atom_keys()`
                will be applied to the extracted frame.

            modular : :class:`bool`, optional
                If this kwarg is passed a
                :func:`pywindow.MolecularSystem.make_modular()`
                will be applied to the extracted frame. (default=False)

            rebuild : :class:`bool`, optional
                If this kwarg is passed a `rebuild=True` is passed to
                :func:`pywindow.MolecularSystem.make_modular()` that
                will be applied to the extracted frame. (default=False)

            ncpus : :class:`int`, optional
                If ncpus > 1, then the analysis is performed in parallel for
                the specified number of parallel jobs. Otherwise, it runs in
                serial. (default=1)

        Returns:
            The function returns `None`, the analysis output is
            returned to :attr:`analysis_output` dictionary.

        """
        if override is True:
            self.analysis_output = {}
        if isinstance(frames, int):
            analysed_frame = self._analysis_serial(frames, ncpus)
            if frames not in self.analysis_output:
                self.analysis_output[frames] = analysed_frame
            return analysed_frame
        frames_for_analysis = []
        if isinstance(frames, list):
            for frame in frames:
                if frame not in self.analysis_output:
                    frames_for_analysis.append(frame)  # noqa: PERF401
        if isinstance(frames, tuple):
            for frame in range(frames[0], frames[1]):
                if frame not in self.analysis_output:
                    frames_for_analysis.append(frame)  # noqa: PERF401
        if isinstance(frames, str) and frames in ["all", "everything"]:
            for frame in range(self.no_of_frames):
                if frame not in self.analysis_output:
                    frames_for_analysis.append(frame)  # noqa: PERF401
        self._analysis_parallel(frames_for_analysis, ncpus)
        return None

    def _analysis_serial(self, frame: dict, ncpus: int = 1) -> dict:  # noqa: C901
        settings = {
            "rebuild": False,
            "modular": False,
        }

        molecular_system = self._get_frame(
            self.trajectory_map[frame],
            frame,
            extract_data=True,
        )
        if settings["modular"] is True:
            molecular_system.make_modular(rebuild=settings["rebuild"])
            molecules = molecular_system.molecules
        else:
            molecules = {"0": molecular_system.system_to_molecule()}
        results = {}
        for molecule in molecules:
            mol = molecules[molecule]
            if "molsize" in settings:
                molsize = settings["molsize"]
                if isinstance(molsize, int) and mol.no_of_atoms == molsize:
                    results[molecule] = mol.full_analysis(ncpus=ncpus)
                if isinstance(molsize, tuple) and isinstance(molsize[0], str):
                    if (
                        molsize[0] in ["bigger", "greater", "larger", "more"]
                        and mol.no_of_atoms > molsize[1]
                    ):
                        results[molecule] = mol.full_analysis(ncpus=ncpus)
                    if (
                        molsize[0] in ["smaller", "less"]
                        and mol.no_of_atoms > molsize[1]
                    ):
                        results[molecule] = mol.full_analysis(ncpus=ncpus)
                    if (
                        molsize[0] in ["not", "isnot", "notequal", "different"]
                        and mol.no_of_atoms != molsize[1]
                    ):
                        results[molecule] = mol.full_analysis(ncpus=ncpus)
                    if (
                        molsize[0] in ["is", "equal", "exactly"]
                        and mol.no_of_atoms == molsize[1]
                    ):
                        results[molecule] = mol.full_analysis(ncpus=ncpus)
                    if (
                        molsize[0] in ["between", "inbetween"]
                        and molsize[1] < mol.no_of_atoms < molsize[2]
                    ):
                        results[molecule] = mol.full_analysis(ncpus=ncpus)
            else:
                results[molecule] = mol.full_analysis(ncpus=ncpus)
        return results

    def _analysis_parallel_execute(self, frame: dict) -> None:  # noqa: C901
        settings = {
            "rebuild": False,
            "modular": False,
        }

        molecular_system = self._get_frame(
            self.trajectory_map[frame],
            frame,
            extract_data=True,
        )
        if settings["modular"] is True:
            molecular_system.make_modular(rebuild=settings["rebuild"])
            molecules = molecular_system.molecules
        else:
            molecules = {"0": molecular_system.system_to_molecule()}
        results = {}
        for molecule in molecules:
            mol = molecules[molecule]
            if "molsize" in settings:
                molsize = settings["molsize"]
                if isinstance(molsize, int) and mol.no_of_atoms == molsize:
                    results[molecule] = mol.full_analysis()
                if isinstance(molsize, tuple) and isinstance(molsize[0], str):
                    if (
                        molsize[0] in ["bigger", "greater", "larger", "more"]
                        and mol.no_of_atoms > molsize[1]
                    ):
                        results[molecule] = mol.full_analysis()
                    if (
                        molsize[0] in ["smaller", "less"]
                        and mol.no_of_atoms > molsize[1]
                    ):
                        results[molecule] = mol.full_analysis()
                    if (
                        molsize[0] in ["not", "isnot", "notequal", "different"]
                        and mol.no_of_atoms != molsize[1]
                    ):
                        results[molecule] = mol.full_analysis()
                    if (
                        molsize[0] in ["is", "equal", "exactly"]
                        and mol.no_of_atoms == molsize[1]
                    ):
                        results[molecule] = mol.full_analysis()
                    if (
                        molsize[0] in ["between", "inbetween"]
                        and molsize[1] < mol.no_of_atoms < molsize[2]
                    ):
                        results[molecule] = mol.full_analysis()
            else:
                results[molecule] = mol.full_analysis()
        return frame, results

    def _analysis_parallel(self, frames: list, ncpus: int) -> None:
        try:
            pool = Pool(processes=ncpus)
            parallel = [
                pool.apply_async(
                    self._analysis_parallel_execute, args=(frame,), kwds=kwargs
                )
                for frame in frames
            ]
            results = [p.get() for p in parallel if p.get()[1] is not None]
            pool.terminate()
            for i in results:
                self.analysis_output[i[0]] = i[1]
        except TypeError:
            pool.terminate()
            msg = "Parallel analysis failed."
            raise _ParallelAnalysisError(msg) from None

    def save_analysis(
        self,
        filepath: str | pathlib.Path | None = None,
    ) -> None:
        """Dump the content of :attr:`analysis_output` as JSON dictionary.

        Parameters:
            filepath:
                The filepath for the JSON file.

        Returns:
            :class:`NoneType`
        """
        filepath = pathlib.Path(filepath)
        # We pass a copy of the analysis attribute dictionary.
        dict_obj = deepcopy(self.analysis_output)
        # If no filepath is provided we create one.
        if filepath is None:
            filepath = f"{self.system_id}_pywindow_analysis"
            filepath = f"{pathlib.Path.cwd()}/{filepath}"
        # Dump the dictionary to json file.
        Output().dump2json(dict_obj, filepath, default=to_list, **kwargs)

    def save_frames(  # noqa: C901, PLR0912
        self,
        frames: int | list | tuple | str,
        filepath: str | pathlib.Path | None = None,
        filetype: str = "pdb",
    ) -> None:
        filepath = pathlib.Path(filepath)
        settings = {
            "pdb": Output()._save_pdb,  # noqa: SLF001
            "xyz": Output()._save_xyz,  # noqa: SLF001
            "decipher": True,
            "forcefield": None,
        }

        if filetype.lower() not in settings:
            msg = f"The '{filetype}' file format is not supported"
            raise _FormatError(msg)
        frames_to_get = []
        if isinstance(frames, int):
            frames_to_get.append(frames)
        if isinstance(frames, list):
            frames_to_get = frames
        if isinstance(frames, tuple):
            for frame in range(frames[0], frames[1]):
                frames_to_get.append(frame)
        if isinstance(frames, str) and frames in ["all", "everything"]:
            for frame in range(self.no_of_frames):
                frames_to_get.append(frame)
        for frame in frames_to_get:
            if frame not in self.frames:
                _ = self.get_frames(frame)
        # If no filepath is provided we create one.
        if filepath is None:
            filepath = pathlib.Path.cwd() / str(self.system_id)
        for frame in frames_to_get:
            frame_molsys = self.frames[frame]
            if (
                settings["decipher"] is True
                and settings["forcefield"] is not None
            ):
                if "swap_atoms" in settings:
                    if isinstance(settings["swap_atoms"], dict):
                        frame_molsys.swap_atom_keys(settings["swap_atoms"])
                    else:
                        msg = (
                            "The swap_atom_keys function only accepts "
                            "'swap_atoms' argument in form of a dictionary."
                        )
                        raise _FunctionError(msg)
                frame_molsys.decipher_atom_keys(settings["forcefield"])
            ffilepath = "_".join((filepath, str(frame)))
            if "elements" not in frame_molsys.system:
                msg = (
                    "The frame (MolecularSystem object) needs to have "
                    "'elements' attribute within the system dictionary. "
                    "It is, therefore, neccessary that you set a decipher "
                    "keyword to True. (see manual)"
                )
                raise _FunctionError(msg)
            settings[filetype.lower()](
                frame_molsys.system, ffilepath, **kwargs
            )
