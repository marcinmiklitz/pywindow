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
from typing import Any, Literal

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
    system: dict,  # type: ignore[type-arg]
    supercell: list[float] | None = None,
) -> MolecularSystem:
    """Return a supercell.

    This functions takes the input unitcell and creates a supercell of it that
    is returned as a new :class:`pywindow.MolecularSystem`.

    Parameters:
        system:
            The unit cell for creation of the supercell

        supercell:
            A list that specifies the size of the supercell in the a, b and c
            direction. (default=[1, 1, 1])

    Returns:
        Returns the created supercell as a new :class:`MolecularSystem`.

    """
    if supercell is None:
        supercell = [1, 1, 1]
    user_supercell = [[1, supercell[0]], [1, supercell[1]], [1, supercell[1]]]
    system = create_supercell(system=system, supercell=user_supercell)
    return MolecularSystem.load_system(system)


class Trajectory:
    """A base class container for trajectories."""

    def __init__(self) -> None:
        self.frames = {}  # type: ignore[var-annotated]
        self.analysis_output = {}  # type: ignore[var-annotated]
        msg = "This is an abstract class"
        raise NotImplementedError(msg)

    def get_frames(  # noqa: C901
        self,
        frames: int
        | list[int]
        | tuple[int, int]
        | Literal["all", "everything"] = "all",
        override: bool = False,  # noqa: FBT001, FBT002
        swap_atoms: dict | None = None,  # type: ignore[type-arg]
        forcefield: str | None = None,
        extract_data: bool = True,  # noqa: FBT001, FBT002
    ) -> dict[int, MolecularSystem]:
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
            Dictionary of frames.

        """
        collected_frames = {}
        if override is True:
            self.frames = {}

        if isinstance(frames, int):
            frame_system = self._get_frame(
                frame_coordinates=self.trajectory_map[frames],  # type: ignore[attr-defined]
                frame_no=frames,
                swap_atoms=swap_atoms,
                forcefield=forcefield,
                extract_data=extract_data,
            )
            if frames not in self.frames:
                self.frames[frames] = frame_system
                collected_frames[frames] = frame_system
            return collected_frames

        if isinstance(frames, list):
            for frame in frames:
                if frame not in self.frames:
                    self.frames[frame] = self._get_frame(
                        frame_coordinates=self.trajectory_map[frame],  # type: ignore[attr-defined]
                        frame_no=frame,
                        swap_atoms=swap_atoms,
                        forcefield=forcefield,
                        extract_data=extract_data,
                    )
                collected_frames[frame] = self.frames[frame]

        if isinstance(frames, tuple):
            for frame in range(frames[0], frames[1]):
                if frame not in self.frames:
                    self.frames[frame] = self._get_frame(
                        frame_coordinates=self.trajectory_map[frame],  # type: ignore[attr-defined]
                        frame_no=frame,
                        swap_atoms=swap_atoms,
                        forcefield=forcefield,
                        extract_data=extract_data,
                    )
                collected_frames[frame] = self.frames[frame]

        if isinstance(frames, str) and frames in ["all", "everything"]:
            for frame in range(self.no_of_frames):  # type: ignore[attr-defined]
                if frame not in self.frames:
                    self.frames[frame] = self._get_frame(
                        frame_coordinates=self.trajectory_map[frame],  # type: ignore[attr-defined]
                        frame_no=frame,
                        swap_atoms=swap_atoms,
                        forcefield=forcefield,
                        extract_data=extract_data,
                    )
                collected_frames[frame] = self.frames[frame]

        return collected_frames

    def _decode_frame(self, frame: list) -> dict:  # type:ignore[type-arg]
        raise NotImplementedError

    def _get_frame(
        self,
        frame_coordinates: list[int],
        frame_no: int,
        swap_atoms: dict | None = None,  # type: ignore[type-arg]
        forcefield: str | None = None,
        extract_data: bool = True,  # noqa: FBT001, FBT002
    ) -> MolecularSystem:
        start, end = frame_coordinates
        with (
            self.filepath.open() as trajectory_file,  # type: ignore[attr-defined]
            closing(
                mmap(trajectory_file.fileno(), 0, access=ACCESS_READ)
            ) as mapped_file,
        ):
            if extract_data is False:
                return mapped_file[start:end].decode("utf-8")  # type: ignore[return-value]
            # [:-1] because the split results in last list empty.
            frame = [
                i.split()
                for i in mapped_file[start:end].decode("utf-8").split("\n")
            ][:-1]
            decoded_frame = self._decode_frame(frame)
            molsys = MolecularSystem.load_system(
                decoded_frame,
                "_".join([self.system_id, str(frame_no)]),  # type: ignore[attr-defined]
            )

            if swap_atoms is not None:
                molsys.swap_atom_keys(swap_atoms)
            if forcefield is not None:
                molsys.decipher_atom_keys(forcefield)
            return molsys

    def save_analysis(
        self,
        filepath: str | pathlib.Path | None = None,
        override: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Dump the content of :attr:`analysis_output` as JSON dictionary."""
        # We pass a copy of the analysis attribute dictionary.
        dict_obj = deepcopy(self.analysis_output)  # type: ignore[attr-defined]
        # If no filepath is provided we create one.
        if filepath is None:
            filepath = f"{self.system_id}_pywindow_analysis"  # type: ignore[attr-defined]
            filepath = pathlib.Path.cwd() / filepath
        filepath = pathlib.Path(filepath)

        # Dump the dictionary to json file.
        Output().dump2json(
            dict_obj,
            filepath,
            default=to_list,
            override=override,
        )

    def save_frames(  # noqa: C901, PLR0912
        self,
        frames: int | list | tuple | Literal["all", "everything"] = "all",  # type: ignore[type-arg]
        filepath: str | pathlib.Path | None = None,
        decipher: bool = True,  # noqa: FBT001, FBT002
        swap_atoms: dict | None = None,  # type: ignore[type-arg]
        forcefield: str | None = None,
    ) -> None:
        # If no filepath is provided we create one.
        if filepath is None:
            filepath = pathlib.Path.cwd() / str(self.system_id)  # type: ignore[attr-defined]
        filepath = pathlib.Path(filepath)

        frames_to_get = []
        if isinstance(frames, int):
            frames_to_get.append(frames)
        if isinstance(frames, list):
            frames_to_get = frames
        if isinstance(frames, tuple):
            for frame in range(frames[0], frames[1]):
                frames_to_get.append(frame)
        if isinstance(frames, str) and frames in ["all", "everything"]:
            for frame in range(self.no_of_frames):  # type: ignore[attr-defined]
                frames_to_get.append(frame)
        for frame in frames_to_get:
            if frame not in self.frames:
                _ = self.get_frames(frame)

        for frame in frames_to_get:
            frame_molsys = self.frames[frame]
            if decipher is True and forcefield is not None:
                if swap_atoms is not None:
                    if isinstance(swap_atoms, dict):
                        frame_molsys.swap_atom_keys(swap_atoms)
                    else:
                        msg = (  # type: ignore[unreachable]
                            "The swap_atom_keys function only accepts "
                            "'swap_atoms' argument in form of a dictionary."
                        )
                        raise _FunctionError(msg)
                frame_molsys.decipher_atom_keys(forcefield)
            ffilepath = "_".join((str(filepath), str(frame)))

            if "elements" not in frame_molsys.system:
                msg = (
                    "The frame (MolecularSystem object) needs to have "
                    "'elements' attribute within the system dictionary. "
                    "It is, therefore, neccessary that you set a decipher "
                    "keyword to True. (see manual)"
                )
                raise _FunctionError(msg)

            if filepath.suffix == ".pdb":
                Output()._save_pdb(  # noqa: SLF001
                    system=frame_molsys.system,
                    filepath=ffilepath,
                    atom_ids_key="elements"
                    if "atom_ids" not in frame_molsys.system
                    else "atom_ids",
                    forcefield=forcefield,
                    decipher=decipher,
                )
            elif filepath.suffix == ".xyz":
                Output()._save_xyz(  # noqa: SLF001
                    system=frame_molsys.system,
                    filepath=ffilepath,
                    forcefield=forcefield,
                    decipher=decipher,
                )
            else:
                msg = (
                    f"The {filepath.suffix} file extension is "
                    "not supported for dumping a MolecularSystem or a"
                    " Molecule. Please use XYZ or PDB."
                )
                raise _FormatError(msg)

    def analysis(  # noqa: C901, PLR0912, PLR0913
        self,
        frames: int | list | tuple | str = "all",  # type: ignore[type-arg]
        ncpus: int = 1,
        ncpus_analysis: int = 1,
        override: bool = False,  # noqa: FBT001, FBT002
        modular: bool = False,  # noqa: FBT001, FBT002
        rebuild: bool = False,  # noqa: FBT001, FBT002
        swap_atoms: dict | None = None,  # type: ignore[type-arg]
        forcefield: str | None = None,
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

            ncpus_analysis : :class:`int`, optional
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
                for frame in range(self.no_of_frames):  # type: ignore[attr-defined]
                    frames_for_analysis.append(frame)  # noqa: PERF402

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
                analysed_frame = self._analysis_serial(
                    frame=frame,
                    ncpus_analysis=ncpus_analysis,
                    rebuild=rebuild,
                    modular=modular,
                    forcefield=forcefield,
                    swap_atoms=swap_atoms,
                )
                self.analysis_output[frame] = analysed_frame

        if ncpus > 1:
            self._analysis_parallel(
                frames=frames_for_analysis,
                ncpus=ncpus,
                ncpus_analysis=ncpus_analysis,
                rebuild=rebuild,
                modular=modular,
                forcefield=forcefield,
                swap_atoms=swap_atoms,
            )

    def _analysis_serial(  # noqa: PLR0913
        self,
        frame: int,
        ncpus_analysis: int = 1,
        rebuild: bool = False,  # noqa: FBT001, FBT002
        modular: bool = False,  # noqa: FBT001, FBT002
        swap_atoms: dict | None = None,  # type: ignore[type-arg]
        forcefield: str | None = None,
    ) -> dict:  # type: ignore[type-arg]
        molecular_system = self._get_frame(
            frame_coordinates=self.trajectory_map[frame],  # type: ignore[attr-defined]
            frame_no=frame,
            extract_data=True,
            swap_atoms=swap_atoms,
            forcefield=forcefield,
        )
        if modular is True:
            molecular_system.make_modular(rebuild=rebuild)
            molecules = molecular_system.molecules
        else:
            molecules = {"0": molecular_system.system_to_molecule()}

        results = {}
        for molecule in molecules:
            mol = molecules[molecule]
            results[molecule] = mol.full_analysis(ncpus=ncpus_analysis)
        return results

    def _analysis_parallel_execute(  # type: ignore[no-untyped-def]
        self,
        frame: int,
        # Using kwargs here for parallel execution.
        **kwargs,  # noqa: ANN003
    ) -> tuple[int, dict]:  # type: ignore[type-arg]
        settings = {"rebuild": False, "modular": False}
        settings.update(kwargs)

        molecular_system = self._get_frame(
            frame_coordinates=self.trajectory_map[frame],  # type: ignore[attr-defined]
            frame_no=frame,  # type: ignore[arg-type]
            extract_data=True,
            swap_atoms=settings["swap_atoms"],  # type: ignore[arg-type]
            forcefield=settings["forcefield"],  # type: ignore[arg-type]
        )
        if settings["modular"] is True:
            molecular_system.make_modular(rebuild=settings["rebuild"])
            molecules = molecular_system.molecules
        else:
            molecules = {"0": molecular_system.system_to_molecule()}
        results = {}
        for molecule in molecules:
            mol = molecules[molecule]
            results[molecule] = mol.full_analysis(
                ncpus=settings["ncpus_analysis"]
            )
        return frame, results

    def _analysis_parallel(  # noqa: PLR0913
        self,
        frames: list,  # type: ignore[type-arg]
        ncpus: int,
        ncpus_analysis: int,
        rebuild: bool = False,  # noqa: FBT001, FBT002
        modular: bool = False,  # noqa: FBT001, FBT002
        swap_atoms: dict | None = None,  # type: ignore[type-arg]
        forcefield: str | None = None,
    ) -> None:
        try:
            pool = Pool(processes=ncpus)
            parallel = [
                pool.apply_async(
                    self._analysis_parallel_execute,
                    args=(frame,),
                    kwds={
                        "swap_atoms": swap_atoms,
                        "rebuild": rebuild,
                        "forcefield": forcefield,
                        "modular": modular,
                        "ncpus_analysis": ncpus_analysis,
                    },
                )
                for frame in frames
            ]
            results = [p.get() for p in parallel if p.get()]
            pool.terminate()
            for i in results:
                self.analysis_output[i[0]] = i[1]  # type: ignore[index, attr-defined]
        except TypeError:
            pool.terminate()
            msg = "Parallel analysis failed."
            raise _ParallelAnalysisError(msg) from None


class DLPOLY(Trajectory):
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
        self.system_id = self.filepath.name.split(".")[0]
        self.frames = {}  # type: ignore[var-annotated]
        self.analysis_output = {}  # type: ignore[var-annotated]
        # Check the history file at init, if no errors, proceed to mapping.
        self._check_history()
        # Map the trajectory file at init.
        self._map_history()

    def _map_history(self) -> None:
        """Map HISTORY file."""
        self.trajectory_map: dict[str | int, Any] = {}
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
                        frame_start: int = progress
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

    def _decode_head(
        self,
        header_coordinates: list[int],
    ) -> list[list[str]]:
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
            header = [int(i) for i in header[1]]  # type:ignore[misc]
        self.periodic_boundary = self._imcon[header[1]]  # type:ignore[index]
        self.content_type = self._keytrj[header[0]]  # type:ignore[index]
        self.no_of_atoms = header[2]  # type:ignore[index]
        return header

    def _decode_frame(self, frame: list) -> dict:  # type:ignore[type-arg]  # noqa: C901, PLR0912
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
            frame_data["lattice"] = np.array(frame[1:4], dtype=float).T  # type:ignore[assignment]
            frame_data["unit_cell"] = lattice_array_to_unit_cell(  # type:ignore[assignment]
                frame_data["lattice"]  # type:ignore[arg-type]
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

        frame_data["atom_ids"] = np.array(elements)  # type:ignore[assignment]
        frame_data["coordinates"] = np.array(coordinates, dtype=float)  # type:ignore[assignment]
        if velocities:
            frame_data["velocities"] = np.array(velocities, dtype=float)  # type:ignore[assignment]
        if forces:
            frame_data["forces"] = np.array(forces, dtype=float)  # type:ignore[assignment]
        return frame_data

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
                        error = " ".join((f"Line {line}:", error_1))
                        raise _TrajectoryError(error)

                # Error 2
                if len(string_line) == 0:
                    error = " ".join((f"Line {line}:", error_2))
                    raise _TrajectoryError(error)


class XYZ(Trajectory):
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
        self.filename = self.filepath.name
        self.system_id = self.filename.split(".")[0]
        self.frames = {}  # type: ignore[var-annotated]
        self.analysis_output = {}  # type: ignore[var-annotated]
        # Map the trajectory file at init.
        self._map_trajectory()

    def _map_trajectory(self) -> None:
        """Map xyz trajectory."""
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

    def _decode_frame(self, frame: list) -> dict:  # type: ignore[type-arg]
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
        frame_data["atom_ids"] = np.array(elements)  # type: ignore[assignment]
        frame_data["coordinates"] = np.array(coordinates, dtype=float)  # type: ignore[assignment]
        return frame_data


class PDB(Trajectory):
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
            filepath:
                The filepath.

            filename:
                The filename.

            system_id:
                The system id inherited from the filename.

            frames:
                A dictionary that is populated, on the fly, with the extracted
                frames.

            analysis_output:
                A dictionary that is populated, on the fly, with the analysis
                output.

        """
        self.filepath = pathlib.Path(filepath)
        self.filename = self.filepath.name
        self.system_id = self.filename.split(".")[0]
        self.frames: dict = {}  # type: ignore[type-arg]
        self.analysis_output: dict = {}  # type: ignore[type-arg]
        # Map the trajectory file at init.
        self._map_trajectory()

    def _map_trajectory(self) -> None:
        """Map pdb trajectory."""
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

    def _decode_frame(self, frame: list) -> dict:  # type: ignore[type-arg]
        frame_data = {}  # type: ignore[var-annotated]
        elements = []
        coordinates = []
        for i in range(len(frame)):
            if frame[i][:6] == "REMARK":
                if "REMARKS" not in frame_data:
                    frame_data["REMARKS"] = []  # type: ignore[assignment]
                frame_data["REMARKS"].append(frame[i][6:])  # type: ignore[attr-defined]
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
