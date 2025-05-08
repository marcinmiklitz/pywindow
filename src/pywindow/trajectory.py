"""
Module intended for the analysis of molecular dynamics trajectories.

The trajectory file (DL_POLY_C:HISTORY, PDB or XYZ) should be loaded with
the one of the corresponding classes (DLPOLY, PDB or XYZ, respectively).

Example
-------
In this example a DL_POLY_C HISTORY trajectory file is loaded.

.. code-block:: python

    pywindow.trajectory.DLPOLY('path/to/HISTORY')

Then, each of the trajectory frames can be extracted and returned as a
:class:`pywindow.molecular.MolecularSystem` object for analysis. See
:mod:`pywindow.molecular` docstring for more information.

Alternatively, the analysis can be performed on a whole or a chunk of
the trajectory with the :func:`analysis()` function. The benefit is
that the analysis can be performed in parallel and the results stored as a
single JSON dictionary in a straightforward way. Also, the deciphering of the
force field atom ids and the rebuilding of molecules can be applied to each
frame in a consitent and automated manner. The downfall is that at the
moment it is not possible to choose the set of parameters that are being
calculated in the :class:`pywindow.molecular.Molecule` as the
:func:`pywindow.molecular.Molecule.full_analysis()` is invoked by default.
However, the computational cost of calculating majority of the structural
properties is miniscule and it is usually the
:func:`pywindow.molecular.MolecularSystem.rebuild_system()` step that is the
bottleneck.

"""
import os
import numpy as np
from copy import deepcopy
from mmap import mmap, ACCESS_READ
from contextlib import closing
from multiprocessing import Pool

from .io_tools import Input, Output
from .utilities import (
    is_number, create_supercell, lattice_array_to_unit_cell, to_list
)
from .molecular import MolecularSystem


class _ParallelAnalysisError(Exception):
    def __init__(self, message):
        self.message = message


class _TrajectoryError(Exception):
    def __init__(self, message):
        self.message = message


class _FormatError(Exception):
    def __init__(self, message):
        self.message = message


class _FunctionError(Exception):
    def __init__(self, message):
        self.message = message


def make_supercell(system, matrix, supercell=[1, 1, 1]):
    """
    Return a supercell.

    This functions takes the input unitcell and creates a supercell of it that
    is returned as a new :class:`pywindow.molecular.MolecularSystem`.

    Parameters
    ----------
    system : :attr:`pywindow.molecular.MolecularSystem.system`
        The unit cell for creation of the supercell

    matrix : :class:`numpy.array`
        The unit cell parameters in form of a lattice.

    supercell : :class:`list`, optional
        A list that specifies the size of the supercell in the a, b and c
        direction. (default=[1, 1, 1])

    Returns
    -------
    :class:`pywindow.molecular.MolecularSystem`
        Returns the created supercell as a new :class:`MolecularSystem`.

    """
    user_supercell = [[1, supercell[0]], [1, supercell[1]], [1, supercell[1]]]
    system = create_supercell(system, matrix, supercell=user_supercell)
    return MolecularSystem.load_system(system)


class DLPOLY(object):
    """
    A container for a DL_POLY_C type trajectory (HISTORY).

    This function takes a DL_POLY_C trajectory file and maps it for the
    binary points in the file where each frame starts/ends. This way the
    process is fast, as it not require loading the trajectory into computer
    memory. When a frame is being extracted, it is only this frame that gets
    loaded to the memory.

    Frames can be accessed individually and loaded as an unmodified string,
    returned as a :class:`pywindow.molecular.MolecularSystem` (and analysed),
    dumped as PDB or XYZ or JSON (if dumped as a
    :attr:`pywindow.molecular.MolecularSystem.system`)

    Attributes
    ----------
    filepath : :class:`str`
        The filepath.

    system_id : :class:`str`
        The system id inherited from the filename.

    frames : :class:`dict`
        A dictionary that is populated, on the fly, with the extracted frames.

    analysis_output : :class:`dict`
        A dictionary that is populated, on the fly, with the analysis output.

    """
    def __init__(self, filepath):
        # Image conventions - periodic boundary key.
        self._imcon = {
            0: 'nonperiodic',
            1: 'cubic',
            2: 'orthorhombic',
            3: 'parallelepiped',
            4: 'truncated octahedral',
            5: 'rhombic dodecahedral',
            6: 'x-y parallelogram',
            7: 'hexagonal prism',
        }
        # Trajectory key - content type.
        self._keytrj = {
            0: 'coordinates',
            1: 'coordinates and velocities',
            2: 'coordinates, velocities and forces',
        }
        self.filepath = filepath
        self.system_id = os.path.basename(filepath)
        self.frames = {}
        self.analysis_output = {}
        # Check the history file at init, if no errors, proceed to mapping.
        self._check_HISTORY()
        # Map the trajectory file at init.
        self._map_HISTORY()

    def _map_HISTORY(self):
        """ """
        self.trajectory_map = {}
        with open(self.filepath, 'r') as trajectory_file:
            with closing(
                    mmap(
                        trajectory_file.fileno(), 0,
                        access=ACCESS_READ)) as mapped_file:
                progress = 0
                line = 0
                frame = 0
                cell_param_line = 0
                # We need to first process trajectory file's header.
                header_flag = True
                while progress <= len(mapped_file):
                    line = line + 1
                    # We read a binary data from a mapped file.
                    bline = mapped_file.readline()
                    # If the bline length equals zero we terminate.
                    # We reached end of the file but still add the last frame!
                    if len(bline) == 0:
                        self.trajectory_map[frame] = [frame_start, progress]
                        frame = frame + 1
                        break
                    # We need to decode byte line into an utf-8 string.
                    sline = bline.decode("utf-8").strip('\n').split()
                    # We extract map's byte coordinates for each frame
                    if header_flag is False:
                        if sline[0] == 'timestep':
                            self.trajectory_map[frame] = [
                                frame_start, progress
                            ]
                            frame_start = progress
                            frame = frame + 1
                    # Here we extract the map's byte coordinates for the header
                    # And also the periodic system type needed for later.
                    if header_flag is True:
                        if sline[0] == 'timestep':
                            self.trajectory_map['header'] = self._decode_head(
                                [0, progress])
                            frame_start = progress
                            header_flag = False
                    progress = progress + len(bline)
            self.no_of_frames = frame

    def _decode_head(self, header_coordinates):
        start, end = header_coordinates
        with open(self.filepath, 'r') as trajectory_file:
            with closing(
                    mmap(
                        trajectory_file.fileno(), 0,
                        access=ACCESS_READ)) as mapped_file:
                header = [
                    i.split()
                    for i in mapped_file[start:end].decode("utf-8").split('\n')
                ]
                header = [int(i) for i in header[1]]
        self.periodic_boundary = self._imcon[header[1]]
        self.content_type = self._keytrj[header[0]]
        self.no_of_atoms = header[2]
        return header

    def get_frames(self, frames='all', override=False, **kwargs):
        """
        Extract frames from the trajectory file.

        Depending on the passed parameters a frame, a list of particular
        frames, a range of frames (from, to), or all frames can be extracted
        with this function.

        Parameters
        ----------
        frames : :class:`int` or :class:`list` or :class:`touple` or :class:`str`
            Specified frame (:class:`int`), or frames (:class:`list`), or
            range (:class:`touple`), or `all`/`everything` (:class:`str`).
            (default=`all`)

        override : :class:`bool`
            If True, a frame already storred in :attr:`frames` can be override.
            (default=False)

        extract_data : :class:`bool`, optional
            If False, a frame is returned as a :class:`str` block as in the
            trajectory file. Ohterwise, it is extracted and returned as
            :class:`pywindow.molecular.MolecularSystem`. (default=True)

        swap_atoms : :class:`dict`, optional
            If this kwarg is passed with an appropriate dictionary a
            :func:`pywindow.molecular.MolecularSystem.swap_atom_keys()` will
            be applied to the extracted frame.

        forcefield : :class:`str`, optional
            If this kwarg is passed with appropriate forcefield keyword a
            :func:`pywindow.molecular.MolecularSystem.decipher_atom_keys()`
            will be applied to the extracted frame.

        Returns
        -------
        :class:`pywindow.molecular.MolecularSystem`
            If a single frame is extracted.

        None : :class:`NoneType`
            If more than one frame is extracted, the frames are returned to
            :attr:`frames`

        """
        if override is True:
            self.frames = {}
        if isinstance(frames, int):
            frame = self._get_frame(
                self.trajectory_map[frames], frames, **kwargs)
            if frames not in self.frames.keys():
                self.frames[frames] = frame
            return frame
        if isinstance(frames, list):
            for frame in frames:
                if frame not in self.frames.keys():
                    self.frames[frame] = self._get_frame(
                        self.trajectory_map[frame], frame, **kwargs)
        if isinstance(frames, tuple):
            for frame in range(frames[0], frames[1]):
                if frame not in self.frames.keys():
                    self.frames[frame] = self._get_frame(
                        self.trajectory_map[frame], frame, **kwargs)
        if isinstance(frames, str):
            if frames in ['all', 'everything']:
                for frame in range(0, self.no_of_frames):
                    if frame not in self.frames.keys():
                        self.frames[frame] = self._get_frame(
                            self.trajectory_map[frame], frame, **kwargs)

    def _get_frame(self, frame_coordinates, frame_no, **kwargs):
        kwargs_ = {
            "extract_data": True
        }
        kwargs_.update(kwargs)
        start, end = frame_coordinates
        with open(self.filepath, 'r') as trajectory_file:
            with closing(
                    mmap(
                        trajectory_file.fileno(), 0,
                        access=ACCESS_READ)) as mapped_file:
                if kwargs_["extract_data"] is False:
                    return mapped_file[start:end].decode("utf-8")
                else:
                    # [:-1] because the split results in last list empty.
                    frame = [
                        i.split()
                        for i in mapped_file[start:end].decode("utf-8").split(
                            '\n')
                    ][:-1]
                    decoded_frame = self._decode_frame(frame)
                    molsys = MolecularSystem.load_system(
                        decoded_frame,
                        "_".join([self.system_id, str(frame_no)]))
                    if 'swap_atoms' in kwargs:
                        molsys.swap_atom_keys(kwargs['swap_atoms'])
                    if 'forcefield' in kwargs:
                        molsys.decipher_atom_keys(kwargs['forcefield'])
                    return molsys

    def _decode_frame(self, frame):
        frame_data = {
            'frame_info': {
                'nstep': int(frame[0][1]),
                'natms': int(frame[0][2]),
                'keytrj': int(frame[0][3]),
                'imcon': int(frame[0][4]),
                'tstep': float(frame[0][5])
            }
        }
        start_line = 1
        if frame_data['frame_info']['imcon'] in [1, 2, 3]:
            frame_data['lattice'] = np.array(frame[1:4], dtype=float).T
            frame_data['unit_cell'] = lattice_array_to_unit_cell(frame_data[
                'lattice'])
            start_line = 4
        # Depending on what the trajectory key is (see __init__) we need
        # to extract every second/ third/ fourth line for elements and coor.
        elements = []
        coordinates = []
        velocities = []
        forces = []
        for i in range(len(frame[start_line:])):
            i_ = i + start_line
            if frame_data['frame_info']['keytrj'] == 0:
                if i % 2 == 0:
                    elements.append(frame[i_][0])
                if i % 2 == 1:
                    coordinates.append(frame[i_])
            if frame_data['frame_info']['keytrj'] == 1:
                if i % 3 == 0:
                    elements.append(frame[i_][0])
                if i % 3 == 1:
                    coordinates.append(frame[i_])
                if i % 3 == 2:
                    velocities.append(frame[i_])
            if frame_data['frame_info']['keytrj'] == 2:
                if i % 4 == 0:
                    elements.append(frame[i_][0])
                if i % 4 == 1:
                    coordinates.append(frame[i_])
                if i % 4 == 2:
                    velocities.append(frame[i_])
                if i % 4 == 3:
                    forces.append(frame[i_])
        frame_data['atom_ids'] = np.array(elements)
        frame_data['coordinates'] = np.array(coordinates, dtype=float)
        if velocities:
            frame_data['velocities'] = np.array(velocities, dtype=float)
        if forces:
            frame_data['forces'] = np.array(forces, dtype=float)
        return frame_data

    def analysis(
            self, frames='all', ncpus=1, _ncpus=1, override=False, **kwargs
                ):
        """
        Perform structural analysis on a frame/ set of frames.

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

        Parameters
        ----------
        frames : :class:`int` or :class:`list` or :class:`touple` or :class:`str`
            Specified frame (:class:`int`), or frames (:class:`list`), or
            range (:class:`touple`), or `all`/`everything` (:class:`str`).
            (default='all')

        override : :class:`bool`
            If True, an output already storred in :attr:`analysis_output` can
            be override. (default=False)

        swap_atoms : :class:`dict`, optional
            If this kwarg is passed with an appropriate dictionary a
            :func:`pywindow.molecular.MolecularSystem.swap_atom_keys()` will
            be applied to the extracted frame.

        forcefield : :class:`str`, optional
            If this kwarg is passed with appropriate forcefield keyword a
            :func:`pywindow.molecular.MolecularSystem.decipher_atom_keys()`
            will be applied to the extracted frame.

        modular : :class:`bool`, optional
            If this kwarg is passed a
            :func:`pywindow.molecular.MolecularSystem.make_modular()`
            will be applied to the extracted frame. (default=False)

        rebuild : :class:`bool`, optional
            If this kwarg is passed a `rebuild=True` is passed to
            :func:`pywindow.molecular.MolecularSystem.make_modular()` that
            will be applied to the extracted frame. (default=False)

        ncpus : :class:`int`, optional
            If ncpus > 1, then the analysis is performed in parallel for the
            specified number of parallel jobs. Otherwise, it runs in serial.
            (default=1)

        Returns
        -------
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
                    raise _FunctionError(
                        "The list should be populated with integers only."
                    )
        if isinstance(frames, tuple):
            if isinstance(frames[0], int) and isinstance(frames[1], int):
                for frame in range(frames[0], frames[1]):
                    frames_for_analysis.append(frame)
                else:
                    raise _FunctionError(
                        "The tuple should contain only two integers "
                        "for the begining and the end of the frames range."
                    )
        if isinstance(frames, str):
            if frames in ['all', 'everything']:
                for frame in range(0, self.no_of_frames):
                    frames_for_analysis.append(frame)
            else:
                raise _FunctionError(
                    "Didn't recognise the keyword. (see manual)"
                )
        # The override keyword by default is False. So we check if any of the
        # frames were already analysed and if so we delete them from the list.
        # However, if the override is set to True, then we just proceed.
        if override is False:
            frames_for_analysis_new = []
            for frame in frames_for_analysis:
                if frame not in self.analysis_output.keys():
                    frames_for_analysis_new.append(frame)
            frames_for_analysis = frames_for_analysis_new
        if ncpus == 1:
            for frame in frames_for_analysis:
                analysed_frame = self._analysis_serial(frame, _ncpus, **kwargs)
                self.analysis_output[frame] = analysed_frame
        if ncpus > 1:
            self._analysis_parallel(frames_for_analysis, ncpus, **kwargs)

    def _analysis_serial(self, frame, _ncpus, **kwargs):
        settings = {
            'rebuild': False,
            'modular': False,
        }
        settings.update(kwargs)
        molecular_system = self._get_frame(
            self.trajectory_map[frame], frame, extract_data=True, **kwargs
        )
        if settings['modular'] is True:
            molecular_system.make_modular(rebuild=settings['rebuild'])
            molecules = molecular_system.molecules
        else:
            molecules = {'0': molecular_system.system_to_molecule()}
        results = {}
        for molecule in molecules:
            mol = molecules[molecule]
            if 'molsize' in settings:
                molsize = settings['molsize']
                if isinstance(molsize, int):
                    if mol.no_of_atoms == molsize:
                        results[molecule] = mol.full_analysis(
                            _ncpus=_ncpus, **kwargs)
                if isinstance(molsize, tuple) and isinstance(molsize[0], str):
                    if molsize[0] in ['bigger', 'greater', 'larger', 'more']:
                        if mol.no_of_atoms > molsize[1]:
                            results[molecule] = mol.full_analysis(
                                _ncpus=_ncpus, **kwargs)
                    if molsize[0] in ['smaller', 'less']:
                        if mol.no_of_atoms > molsize[1]:
                            results[molecule] = mol.full_analysis(
                                _ncpus=_ncpus, **kwargs)
                    if molsize[0] in ['not', 'isnot', 'notequal', 'different']:
                        if mol.no_of_atoms != molsize[1]:
                            results[molecule] = mol.full_analysis(
                                _ncpus=_ncpus, **kwargs)
                    if molsize[0] in ['is', 'equal', 'exactly']:
                        if mol.no_of_atoms == molsize[1]:
                            results[molecule] = mol.full_analysis(
                                _ncpus=_ncpus, **kwargs)
                    if molsize[0] in ['between', 'inbetween']:
                        if molsize[1] < mol.no_of_atoms < molsize[2]:
                            results[molecule] = mol.full_analysis(
                                _ncpus=_ncpus, **kwargs)
            else:
                results[molecule] = mol.full_analysis(_ncpus=_ncpus, **kwargs)
        return results

    def _analysis_parallel_execute(self, frame, **kwargs):
        settings = {
            'rebuild': False,
            'modular': False,
        }
        settings.update(kwargs)
        molecular_system = self._get_frame(
            self.trajectory_map[frame], frame, extract_data=True, **kwargs
        )
        if settings['modular'] is True:
            molecular_system.make_modular(rebuild=settings['rebuild'])
            molecules = molecular_system.molecules
        else:
            molecules = {'0': molecular_system.system_to_molecule()}
        results = {}
        for molecule in molecules:
            mol = molecules[molecule]
            if 'molsize' in settings:
                molsize = settings['molsize']
                if isinstance(molsize, int):
                    if mol.no_of_atoms == molsize:
                        results[molecule] = mol.full_analysis(**kwargs)
                if isinstance(molsize, tuple) and isinstance(molsize[0], str):
                    if molsize[0] in ['bigger', 'greater', 'larger', 'more']:
                        if mol.no_of_atoms > molsize[1]:
                            results[molecule] = mol.full_analysis(**kwargs)
                    if molsize[0] in ['smaller', 'less']:
                        if mol.no_of_atoms > molsize[1]:
                            results[molecule] = mol.full_analysis(**kwargs)
                    if molsize[0] in ['not', 'isnot', 'notequal', 'different']:
                        if mol.no_of_atoms != molsize[1]:
                            results[molecule] = mol.full_analysis(**kwargs)
                    if molsize[0] in ['is', 'equal', 'exactly']:
                        if mol.no_of_atoms == molsize[1]:
                            results[molecule] = mol.full_analysis(**kwargs)
                    if molsize[0] in ['between', 'inbetween']:
                        if molsize[1] < mol.no_of_atoms < molsize[2]:
                            results[molecule] = mol.full_analysis(**kwargs)
            else:
                results[molecule] = mol.full_analysis(**kwargs)
        return frame, results

    def _analysis_parallel(self, frames, ncpus, **kwargs):
        try:
            pool = Pool(processes=ncpus)
            parallel = [
                pool.apply_async(
                    self._analysis_parallel_execute,
                    args=(frame, ),
                    kwds=kwargs) for frame in frames
            ]
            results = [p.get() for p in parallel if p.get()]
            pool.terminate()
            for i in results:
                self.analysis_output[i[0]] = i[1]
        except TypeError:
            pool.terminate()
            raise _ParallelAnalysisError("Parallel analysis failed.")

    def _check_HISTORY(self):
        """
        """
        self.check_log = ""
        line = 0
        binary_step = 0
        timestep = 0
        timestep_flag = 'timestep'
        progress = 0

        warning_1 = "No comment line is present as the file header.\n"
        warning_2 = " ".join(
            (
                "Second header line is missing from the file",
                "that contains information on the system's periodicity",
                "and the type of the trajectory file.\n"
             )
        )
        warning_3 = " ".join(
            (
                "Comment line encountered in the middle of",
                "the trajectory file.\n"
            )
        )

        error_1 = "The trajectory is discontinous.\n"
        error_2 = "The file contains an empty line.\n"

        with open(self.filepath, 'r') as trajectory_file:
            # We open the HISTORY trajectory file
            with closing(
                    mmap(
                        trajectory_file.fileno(), 0,
                        access=ACCESS_READ)) as file_binary_map:
                # We use this binary mapping feature that instead of loading
                # the full file into memory beforehand it only
                # maps the content. Especially useful with enormous files
                while binary_step < len(file_binary_map):
                    line += 1
                    binary_line = file_binary_map.readline()
                    binary_step = binary_step + len(binary_line)
                    progress_old = progress
                    progress = round(binary_step * 100 / len(file_binary_map),
                                     0)
                    string_line = binary_line.decode("utf-8").strip(
                        '\n').split()

                    # Warning 1
                    if line == 1:
                        if string_line[0] != 'DLFIELD':
                            self.check_log = " ".join(
                                (self.check_log, "Line {0}:".format(line),
                                 warning_1)
                            )

                    # Warning 2
                    if line == 2:
                        if len(string_line) != 3:
                            self.check_log = " ".join(
                                (self.check_log, "Line {0}:".format(line),
                                 warning_2)
                            )

                    # Error 1
                    if string_line:
                        if string_line[0] == timestep_flag:
                            old_timestep = timestep
                            timestep = int(string_line[1])
                            if old_timestep > timestep:
                                error = " ".join(
                                    "Line {0}:".format(line), error_1
                                )
                                raise _TrajectoryError(error)

                    # Error 2
                    if len(string_line) == 0:
                        error = " ".join(
                            "Line {0}:".format(line), error_2
                        )
                        raise _TrajectoryError(error)

    def save_analysis(self, filepath=None, **kwargs):
        """
        Dump the content of :attr:`analysis_output` as JSON dictionary.

        Parameters
        ----------
        filepath : :class:`str`
            The filepath for the JSON file.

        Returns
        -------
        None : :class:`NoneType`
        """
        # We pass a copy of the analysis attribute dictionary.
        dict_obj = deepcopy(self.analysis_output)
        # If no filepath is provided we create one.
        if filepath is None:
            filepath = "_".join(
                (str(self.system_id), "pywindow_analysis")
            )
            filepath = '/'.join((os.getcwd(), filepath))
        # Dump the dictionary to json file.
        Output().dump2json(dict_obj, filepath, default=to_list, **kwargs)
        return

    def save_frames(self, frames, filepath=None, filetype='pdb', **kwargs):
        settings = {
            "pdb": Output()._save_pdb,
            "xyz": Output()._save_xyz,
            "decipher": True,
            "forcefield": None,
        }
        settings.update(kwargs)
        if filetype.lower() not in settings.keys():
            raise _FormatError("The '{0}' file format is not supported".format(
                filetype))
        frames_to_get = []
        if isinstance(frames, int):
            frames_to_get.append(frames)
        if isinstance(frames, list):
            frames_to_get = frames
        if isinstance(frames, tuple):
            for frame in range(frames[0], frames[1]):
                frames_to_get.append(frame)
        if isinstance(frames, str):
            if frames in ['all', 'everything']:
                for frame in range(0, self.no_of_frames):
                    frames_to_get.append(frame)
        for frame in frames_to_get:
            if frame not in self.frames.keys():
                _ = self.get_frames(frame)
        # If no filepath is provided we create one.
        if filepath is None:
            filepath = '/'.join((os.getcwd(), str(self.system_id)))
        for frame in frames_to_get:
            frame_molsys = self.frames[frame]
            if settings[
                    'decipher'] is True and settings['forcefield'] is not None:
                if "swap_atoms" in settings.keys():
                    if isinstance(settings["swap_atoms"], dict):
                        frame_molsys.swap_atom_keys(settings["swap_atoms"])
                    else:
                        raise _FunctionError(
                            "The swap_atom_keys function only accepts "
                            "'swap_atoms' argument in form of a dictionary.")
                frame_molsys.decipher_atom_keys(settings["forcefield"])
            ffilepath = '_'.join((filepath, str(frame)))
            if 'elements' not in frame_molsys.system.keys():
                raise _FunctionError(
                    "The frame (MolecularSystem object) needs to have "
                    "'elements' attribute within the system dictionary. "
                    "It is, therefore, neccessary that you set a decipher "
                    "keyword to True. (see manual)")
            settings[filetype.lower()](frame_molsys.system, ffilepath, **
                                       kwargs)


class XYZ(object):
    """
    A container for an XYZ type trajectory.

    This function takes an XYZ trajectory file and maps it for the
    binary points in the file where each frame starts/ends. This way the
    process is fast, as it not require loading the trajectory into computer
    memory. When a frame is being extracted, it is only this frame that gets
    loaded to the memory.

    Frames can be accessed individually and loaded as an unmodified string,
    returned as a :class:`pywindow.molecular.MolecularSystem` (and analysed),
    dumped as PDB or XYZ or JSON (if dumped as a
    :attr:`pywindow.molecular.MolecularSystem.system`)

    Attributes
    ----------
    filepath : :class:`str`
        The filepath.

    filename : :class:`str`
        The filename.

    system_id : :class:`str`
        The system id inherited from the filename.

    frames : :class:`dict`
        A dictionary that is populated, on the fly, with the extracted frames.

    analysis_output : :class:`dict`
        A dictionary that is populated, on the fly, with the analysis output.

    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.system_id = self.filename.split(".")[0]
        self.frames = {}
        self.analysis_output = {}
        # Map the trajectory file at init.
        self._map_trajectory()

    def _map_trajectory(self):
        """ Return filepath as a class attribute"""
        self.trajectory_map = {}
        with open(self.filepath, 'r') as trajectory_file:
            with closing(
                    mmap(
                        trajectory_file.fileno(), 0,
                        access=ACCESS_READ)) as mapped_file:
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
                    sline = bline.decode("utf-8").strip('\n').split()
                    # We extract map's byte coordinates for each frame
                    if (len(sline) == 1 and is_number(sline[0]) and
                            progress > 0):
                        frame = frame + 1
                        self.trajectory_map[frame] = [frame_start, progress]
                        frame_start = progress
                    # Here we extract the map's byte coordinates for the header
                    # And also the periodic system type needed for later.
                    progress = progress + len(bline)
            self.no_of_frames = frame + 1

    def get_frames(self, frames='all', override=False, **kwargs):
        """
        Extract frames from the trajectory file.

        Depending on the passed parameters a frame, a list of particular
        frames, a range of frames (from, to), or all frames can be extracted
        with this function.

        Parameters
        ----------
        frames : :class:`int` or :class:`list` or :class:`touple` or :class:`str`
            Specified frame (:class:`int`), or frames (:class:`list`), or
            range (:class:`touple`), or `all`/`everything` (:class:`str`).
            (default=`all`)

        override : :class:`bool`
            If True, a frame already storred in :attr:`frames` can be override.
            (default=False)

        extract_data : :class:`bool`, optional
            If False, a frame is returned as a :class:`str` block as in the
            trajectory file. Ohterwise, it is extracted and returned as
            :class:`pywindow.molecular.MolecularSystem`. (default=True)

        swap_atoms : :class:`dict`, optional
            If this kwarg is passed with an appropriate dictionary a
            :func:`pywindow.molecular.MolecularSystem.swap_atom_keys()` will
            be applied to the extracted frame.

        forcefield : :class:`str`, optional
            If this kwarg is passed with appropriate forcefield keyword a
            :func:`pywindow.molecular.MolecularSystem.decipher_atom_keys()`
            will be applied to the extracted frame.

        Returns
        -------
        :class:`pywindow.molecular.MolecularSystem`
            If a single frame is extracted.

        None : :class:`NoneType`
            If more than one frame is extracted, the frames are returned to
            :attr:`frames`

        """
        if override is True:
            self.frames = {}
        if isinstance(frames, int):
            frame = self._get_frame(
                self.trajectory_map[frames], frames, **kwargs)
            if frames not in self.frames.keys():
                self.frames[frames] = frame
            return frame
        if isinstance(frames, list):
            for frame in frames:
                if frame not in self.frames.keys():
                    self.frames[frame] = self._get_frame(
                        self.trajectory_map[frame], frame, **kwargs)
        if isinstance(frames, tuple):
            for frame in range(frames[0], frames[1]):
                if frame not in self.frames.keys():
                    self.frames[frame] = self._get_frame(
                        self.trajectory_map[frame], frame, **kwargs)
        if isinstance(frames, str):
            if frames in ['all', 'everything']:
                for frame in range(0, self.no_of_frames):
                    if frame not in self.frames.keys():
                        self.frames[frame] = self._get_frame(
                            self.trajectory_map[frame], frame, **kwargs)

    def _get_frame(self, frame_coordinates, frame_no, **kwargs):
        kwargs_ = {
            "extract_data": True,
            }
        kwargs_.update(kwargs)
        start, end = frame_coordinates
        with open(self.filepath, 'r') as trajectory_file:
            with closing(
                    mmap(
                        trajectory_file.fileno(), 0,
                        access=ACCESS_READ)) as mapped_file:
                if kwargs_["extract_data"] is False:
                    return mapped_file[start:end].decode("utf-8")
                else:
                    # [:-1] because the split results in last list empty.
                    frame = [
                        i.split()
                        for i in mapped_file[start:end].decode("utf-8").split(
                            '\n')
                    ][:-1]
                    decoded_frame = self._decode_frame(frame)
                    molsys = MolecularSystem.load_system(
                        decoded_frame,
                        "_".join([self.system_id, str(frame_no)]))
                    if 'swap_atoms' in kwargs:
                        molsys.swap_atom_keys(kwargs['swap_atoms'])
                    if 'forcefield' in kwargs:
                        molsys.decipher_atom_keys(kwargs['forcefield'])
                    return molsys

    def _decode_frame(self, frame):
        frame_data = {
            'frame_info': {
                'natms': int(frame[0][0]),
                'remarks': " ".join([*frame[1]]),
            }
        }
        start_line = 2
        elements = []
        coordinates = []
        for i in range(start_line, len(frame)):
            elements.append(frame[i][0])
            coordinates.append(frame[i][1:])
        frame_data['atom_ids'] = np.array(elements)
        frame_data['coordinates'] = np.array(coordinates, dtype=float)
        return frame_data

    def analysis(self, frames='all', ncpus=1, override=False, **kwargs):
        """
        Perform structural analysis on a frame/ set of frames.

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

        Parameters
        ----------
        frames : :class:`int` or :class:`list` or :class:`touple` or :class:`str`
            Specified frame (:class:`int`), or frames (:class:`list`), or
            range (:class:`touple`), or `all`/`everything` (:class:`str`).
            (default='all')

        override : :class:`bool`
            If True, an output already storred in :attr:`analysis_output` can
            be override. (default=False)

        swap_atoms : :class:`dict`, optional
            If this kwarg is passed with an appropriate dictionary a
            :func:`pywindow.molecular.MolecularSystem.swap_atom_keys()` will
            be applied to the extracted frame.

        forcefield : :class:`str`, optional
            If this kwarg is passed with appropriate forcefield keyword a
            :func:`pywindow.molecular.MolecularSystem.decipher_atom_keys()`
            will be applied to the extracted frame.

        modular : :class:`bool`, optional
            If this kwarg is passed a
            :func:`pywindow.molecular.MolecularSystem.make_modular()`
            will be applied to the extracted frame. (default=False)

        rebuild : :class:`bool`, optional
            If this kwarg is passed a `rebuild=True` is passed to
            :func:`pywindow.molecular.MolecularSystem.make_modular()` that
            will be applied to the extracted frame. (default=False)

        ncpus : :class:`int`, optional
            If ncpus > 1, then the analysis is performed in parallel for the
            specified number of parallel jobs. Otherwise, it runs in serial.
            (default=1)

        Returns
        -------
        None : :class:`NoneType`
            The function returns `None`, the analysis output is
            returned to :attr:`analysis_output` dictionary.

        """
        if override is True:
            self.analysis_output = {}
        if isinstance(frames, int):
            analysed_frame = self._analysis_serial(frames, ncpus, **kwargs)
            if frames not in self.analysis_output.keys():
                self.analysis_output[frames] = analysed_frame
            return analysed_frame
        else:
            frames_for_analysis = []
            if isinstance(frames, list):
                for frame in frames:
                    if frame not in self.analysis_output.keys():
                        frames_for_analysis.append(frame)
            if isinstance(frames, tuple):
                for frame in range(frames[0], frames[1]):
                    if frame not in self.analysis_output.keys():
                        frames_for_analysis.append(frame)
            if isinstance(frames, str):
                if frames in ['all', 'everything']:
                    for frame in range(0, self.no_of_frames):
                        if frame not in self.analysis_output.keys():
                            frames_for_analysis.append(frame)
            self._analysis_parallel(frames_for_analysis, ncpus, **kwargs)

    def _analysis_serial(self, frame, ncpus, **kwargs):
        settings = {
            'rebuild': False,
            'modular': False,
        }
        settings.update(kwargs)
        molecular_system = self._get_frame(
            self.trajectory_map[frame], frame, extract_data=True, **kwargs
        )
        if settings['modular'] is True:
            molecular_system.make_modular(rebuild=settings['rebuild'])
            molecules = molecular_system.molecules
        else:
            molecules = {'0': molecular_system.system_to_molecule()}
        results = {}
        for molecule in molecules:
            mol = molecules[molecule]
            if 'molsize' in settings:
                molsize = settings['molsize']
                if isinstance(molsize, int):
                    if mol.no_of_atoms == molsize:
                        results[molecule] = mol.full_analysis(
                            ncpus=ncpus, **kwargs)
                if isinstance(molsize, tuple) and isinstance(molsize[0], str):
                    if molsize[0] in ['bigger', 'greater', 'larger', 'more']:
                        if mol.no_of_atoms > molsize[1]:
                            results[molecule] = mol.full_analysis(
                                ncpus=ncpus, **kwargs)
                    if molsize[0] in ['smaller', 'less']:
                        if mol.no_of_atoms > molsize[1]:
                            results[molecule] = mol.full_analysis(
                                ncpus=ncpus, **kwargs)
                    if molsize[0] in ['not', 'isnot', 'notequal', 'different']:
                        if mol.no_of_atoms != molsize[1]:
                            results[molecule] = mol.full_analysis(
                                ncpus=ncpus, **kwargs)
                    if molsize[0] in ['is', 'equal', 'exactly']:
                        if mol.no_of_atoms == molsize[1]:
                            results[molecule] = mol.full_analysis(
                                ncpus=ncpus, **kwargs)
                    if molsize[0] in ['between', 'inbetween']:
                        if molsize[1] < mol.no_of_atoms < molsize[2]:
                            results[molecule] = mol.full_analysis(
                                ncpus=ncpus, **kwargs)
            else:
                results[molecule] = mol.full_analysis(ncpus=ncpus, **kwargs)
        return results

    def _analysis_parallel_execute(self, frame, **kwargs):
        settings = {
            'rebuild': False,
            'modular': False,
        }
        settings.update(kwargs)
        molecular_system = self._get_frame(
            self.trajectory_map[frame], frame, extract_data=True, **kwargs
        )
        if settings['modular'] is True:
            molecular_system.make_modular(rebuild=settings['rebuild'])
            molecules = molecular_system.molecules
        else:
            molecules = {'0': molecular_system.system_to_molecule()}
        results = {}
        for molecule in molecules:
            mol = molecules[molecule]
            if 'molsize' in settings:
                molsize = settings['molsize']
                if isinstance(molsize, int):
                    if mol.no_of_atoms == molsize:
                        results[molecule] = mol.full_analysis(**kwargs)
                if isinstance(molsize, tuple) and isinstance(molsize[0], str):
                    if molsize[0] in ['bigger', 'greater', 'larger', 'more']:
                        if mol.no_of_atoms > molsize[1]:
                            results[molecule] = mol.full_analysis(**kwargs)
                    if molsize[0] in ['smaller', 'less']:
                        if mol.no_of_atoms > molsize[1]:
                            results[molecule] = mol.full_analysis(**kwargs)
                    if molsize[0] in ['not', 'isnot', 'notequal', 'different']:
                        if mol.no_of_atoms != molsize[1]:
                            results[molecule] = mol.full_analysis(**kwargs)
                    if molsize[0] in ['is', 'equal', 'exactly']:
                        if mol.no_of_atoms == molsize[1]:
                            results[molecule] = mol.full_analysis(**kwargs)
                    if molsize[0] in ['between', 'inbetween']:
                        if molsize[1] < mol.no_of_atoms < molsize[2]:
                            results[molecule] = mol.full_analysis(**kwargs)
            else:
                results[molecule] = mol.full_analysis(**kwargs)
        return frame, results

    def _analysis_parallel(self, frames, ncpus, **kwargs):
        try:
            pool = Pool(processes=ncpus)
            parallel = [
                pool.apply_async(
                    self._analysis_parallel_execute,
                    args=(frame, ),
                    kwds=kwargs) for frame in frames
            ]
            results = [p.get() for p in parallel if p.get()[1] is not None]
            pool.terminate()
            for i in results:
                self.analysis_output[i[0]] = i[1]
        except TypeError:
            pool.terminate()
            raise _ParallelAnalysisError("Parallel analysis failed.")

    def save_analysis(self, filepath=None, **kwargs):
        """
        Dump the content of :attr:`analysis_output` as JSON dictionary.

        Parameters
        ----------
        filepath : :class:`str`
            The filepath for the JSON file.

        Returns
        -------
        None : :class:`NoneType`
        """
        # We pass a copy of the analysis attribute dictionary.
        dict_obj = deepcopy(self.analysis_output)
        # If no filepath is provided we create one.
        if filepath is None:
            filepath = "_".join(
                (str(self.system_id), "pywindow_analysis")
            )
            filepath = '/'.join((os.getcwd(), filepath))
        # Dump the dictionary to json file.
        Output().dump2json(dict_obj, filepath, default=to_list, **kwargs)
        return

    def save_frames(self, frames, filepath=None, filetype='pdb', **kwargs):
        settings = {
            "pdb": Output()._save_pdb,
            "xyz": Output()._save_xyz,
            "decipher": True,
            "forcefield": None,
        }
        settings.update(kwargs)
        if filetype.lower() not in settings.keys():
            raise _FormatError("The '{0}' file format is not supported".format(
                filetype))
        frames_to_get = []
        if isinstance(frames, int):
            frames_to_get.append(frames)
        if isinstance(frames, list):
            frames_to_get = frames
        if isinstance(frames, tuple):
            for frame in range(frames[0], frames[1]):
                frames_to_get.append(frame)
        if isinstance(frames, str):
            if frames in ['all', 'everything']:
                for frame in range(0, self.no_of_frames):
                    frames_to_get.append(frame)
        for frame in frames_to_get:
            if frame not in self.frames.keys():
                _ = self.get_frames(frame)
        # If no filepath is provided we create one.
        if filepath is None:
            filepath = '/'.join((os.getcwd(), str(self.system_id)))
        for frame in frames_to_get:
            frame_molsys = self.frames[frame]
            if settings[
                    'decipher'] is True and settings['forcefield'] is not None:
                if "swap_atoms" in settings.keys():
                    if isinstance(settings["swap_atoms"], dict):
                        frame_molsys.swap_atom_keys(settings["swap_atoms"])
                    else:
                        raise _FunctionError(
                            "The swap_atom_keys function only accepts "
                            "'swap_atoms' argument in form of a dictionary.")
                frame_molsys.decipher_atom_keys(settings["forcefield"])
            ffilepath = '_'.join((filepath, str(frame)))
            if 'elements' not in frame_molsys.system.keys():
                raise _FunctionError(
                    "The frame (MolecularSystem object) needs to have "
                    "'elements' attribute within the system dictionary. "
                    "It is, therefore, neccessary that you set a decipher "
                    "keyword to True. (see manual)")
            settings[filetype.lower()](frame_molsys.system, ffilepath, **
                                       kwargs)


class PDB(object):
    def __init__(self, filepath):
        """
        A container for an PDB type trajectory.

        This function takes an PDB trajectory file and maps it for the
        binary points in the file where each frame starts/ends. This way the
        process is fast, as it not require loading the trajectory into computer
        memory. When a frame is being extracted, it is only this frame that gets
        loaded to the memory.

        Frames can be accessed individually and loaded as an unmodified string,
        returned as a :class:`pywindow.molecular.MolecularSystem` (and analysed),
        dumped as PDB or XYZ or JSON (if dumped as a
        :attr:`pywindow.molecular.MolecularSystem.system`)

        Attributes
        ----------
        filepath : :class:`str`
            The filepath.

        filename : :class:`str`
            The filename.

        system_id : :class:`str`
            The system id inherited from the filename.

        frames : :class:`dict`
            A dictionary that is populated, on the fly, with the extracted frames.

        analysis_output : :class:`dict`
            A dictionary that is populated, on the fly, with the analysis output.

        """
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.system_id = self.filename.split(".")[0]
        self.frames = {}
        self.analysis_output = {}
        # Map the trajectory file at init.
        self._map_trajectory()

    def _map_trajectory(self):
        """ Return filepath as a class attribute"""
        self.trajectory_map = {}
        with open(self.filepath, 'r') as trajectory_file:
            with closing(
                    mmap(
                        trajectory_file.fileno(), 0,
                        access=ACCESS_READ)) as mapped_file:
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
                        if progress - frame_start > 10:
                            self.trajectory_map[frame] = [
                                frame_start, progress
                            ]
                        break
                    # We need to decode byte line into an utf-8 string.
                    sline = bline.decode("utf-8").strip('\n').split()
                    # We extract map's byte coordinates for each frame
                    if len(sline) == 1 and sline[0] == 'END':
                        frame = frame + 1
                        self.trajectory_map[frame] = [frame_start, progress]
                        frame_start = progress
                    # Here we extract the map's byte coordinates for the header
                    # And also the periodic system type needed for later.
                    progress = progress + len(bline)
            self.no_of_frames = frame

    def get_frames(self, frames='all', override=False, **kwargs):
        """
        Extract frames from the trajectory file.

        Depending on the passed parameters a frame, a list of particular
        frames, a range of frames (from, to), or all frames can be extracted
        with this function.

        Parameters
        ----------
        frames : :class:`int` or :class:`list` or :class:`touple` or :class:`str`
            Specified frame (:class:`int`), or frames (:class:`list`), or
            range (:class:`touple`), or `all`/`everything` (:class:`str`).
            (default=`all`)

        override : :class:`bool`
            If True, a frame already storred in :attr:`frames` can be override.
            (default=False)

        extract_data : :class:`bool`, optional
            If False, a frame is returned as a :class:`str` block as in the
            trajectory file. Ohterwise, it is extracted and returned as
            :class:`pywindow.molecular.MolecularSystem`. (default=True)

        swap_atoms : :class:`dict`, optional
            If this kwarg is passed with an appropriate dictionary a
            :func:`pywindow.molecular.MolecularSystem.swap_atom_keys()` will
            be applied to the extracted frame.

        forcefield : :class:`str`, optional
            If this kwarg is passed with appropriate forcefield keyword a
            :func:`pywindow.molecular.MolecularSystem.decipher_atom_keys()`
            will be applied to the extracted frame.

        Returns
        -------
        :class:`pywindow.molecular.MolecularSystem`
            If a single frame is extracted.

        None : :class:`NoneType`
            If more than one frame is extracted, the frames are returned to
            :attr:`frames`

        """
        if override is True:
            self.frames = {}
        if isinstance(frames, int):
            frame = self._get_frame(
                self.trajectory_map[frames], frames, **kwargs)
            if frames not in self.frames.keys():
                self.frames[frames] = frame
            return frame
        if isinstance(frames, list):
            for frame in frames:
                if frame not in self.frames.keys():
                    self.frames[frame] = self._get_frame(
                        self.trajectory_map[frame], frame, **kwargs)
        if isinstance(frames, tuple):
            for frame in range(frames[0], frames[1]):
                if frame not in self.frames.keys():
                    self.frames[frame] = self._get_frame(
                        self.trajectory_map[frame], frame, **kwargs)
        if isinstance(frames, str):
            if frames in ['all', 'everything']:
                for frame in range(0, self.no_of_frames):
                    if frame not in self.frames.keys():
                        self.frames[frame] = self._get_frame(
                            self.trajectory_map[frame], frame, **kwargs)

    def _get_frame(self, frame_coordinates, frame_no, **kwargs):
        kwargs_ = {
            "extract_data": True
        }
        kwargs_.update(kwargs)
        start, end = frame_coordinates
        with open(self.filepath, 'r') as trajectory_file:
            with closing(
                    mmap(
                        trajectory_file.fileno(), 0,
                        access=ACCESS_READ)) as mapped_file:
                if kwargs_["extract_data"] is False:
                    return mapped_file[start:end].decode("utf-8")
                else:
                    # In case of PDB we do not split lines!
                    frame = mapped_file[start:end].decode("utf-8").split('\n')
                    decoded_frame = self._decode_frame(frame)
                    molsys = MolecularSystem.load_system(
                        decoded_frame,
                        "_".join([self.system_id, str(frame_no)]))
                    if 'swap_atoms' in kwargs:
                        molsys.swap_atom_keys(kwargs['swap_atoms'])
                    if 'forcefield' in kwargs:
                        molsys.decipher_atom_keys(kwargs['forcefield'])
                    return molsys

    def _decode_frame(self, frame):
        frame_data = {}
        elements = []
        coordinates = []
        for i in range(len(frame)):
            if frame[i][:6] == 'REMARK':
                if 'REMARKS' not in frame_data.keys():
                    frame_data['REMARKS'] = []
                frame_data['REMARKS'].append(frame[i][6:])
            if frame[i][:6] == 'CRYST1':
                cryst = np.array(
                    [
                        frame[i][6:15], frame[i][15:24], frame[i][24:33],
                        frame[i][33:40], frame[i][40:47], frame[i][47:54]
                    ],
                    dtype=float)
                # This is in case of nonperiodic systems, often they have
                # a,b,c unit cell parameters as 0,0,0.
                if sum(cryst[0:3]) != 0:
                    frame_data['CRYST1'] = cryst
            if frame[i][:6] in ['HETATM', 'ATOM  ']:
                elements.append(frame[i][12:16].strip())
                coordinates.append(
                    [frame[i][30:38], frame[i][38:46], frame[i][46:54]])
        frame_data['atom_ids'] = np.array(elements, dtype='<U8')
        frame_data['coordinates'] = np.array(coordinates, dtype=float)
        return frame_data

    def analysis(self, frames='all', ncpus=1, override=False, **kwargs):
        """
        Perform structural analysis on a frame/ set of frames.

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

        Parameters
        ----------
        frames : :class:`int` or :class:`list` or :class:`touple` or :class:`str`
            Specified frame (:class:`int`), or frames (:class:`list`), or
            range (:class:`touple`), or `all`/`everything` (:class:`str`).
            (default='all')

        override : :class:`bool`
            If True, an output already storred in :attr:`analysis_output` can
            be override. (default=False)

        swap_atoms : :class:`dict`, optional
            If this kwarg is passed with an appropriate dictionary a
            :func:`pywindow.molecular.MolecularSystem.swap_atom_keys()` will
            be applied to the extracted frame.

        forcefield : :class:`str`, optional
            If this kwarg is passed with appropriate forcefield keyword a
            :func:`pywindow.molecular.MolecularSystem.decipher_atom_keys()`
            will be applied to the extracted frame.

        modular : :class:`bool`, optional
            If this kwarg is passed a
            :func:`pywindow.molecular.MolecularSystem.make_modular()`
            will be applied to the extracted frame. (default=False)

        rebuild : :class:`bool`, optional
            If this kwarg is passed a `rebuild=True` is passed to
            :func:`pywindow.molecular.MolecularSystem.make_modular()` that
            will be applied to the extracted frame. (default=False)

        ncpus : :class:`int`, optional
            If ncpus > 1, then the analysis is performed in parallel for the
            specified number of parallel jobs. Otherwise, it runs in serial.
            (default=1)

        Returns
        -------
        None : :class:`NoneType`
            The function returns `None`, the analysis output is
            returned to :attr:`analysis_output` dictionary.

        """
        if override is True:
            self.analysis_output = {}
        if isinstance(frames, int):
            analysed_frame = self._analysis_serial(frames, ncpus, **kwargs)
            if frames not in self.analysis_output.keys():
                self.analysis_output[frames] = analysed_frame
            return analysed_frame
        else:
            frames_for_analysis = []
            if isinstance(frames, list):
                for frame in frames:
                    if frame not in self.analysis_output.keys():
                        frames_for_analysis.append(frame)
            if isinstance(frames, tuple):
                for frame in range(frames[0], frames[1]):
                    if frame not in self.analysis_output.keys():
                        frames_for_analysis.append(frame)
            if isinstance(frames, str):
                if frames in ['all', 'everything']:
                    for frame in range(0, self.no_of_frames):
                        if frame not in self.analysis_output.keys():
                            frames_for_analysis.append(frame)
            self._analysis_parallel(frames_for_analysis, ncpus, **kwargs)

    def _analysis_serial(self, frame, ncpus, **kwargs):
        settings = {
            'rebuild': False,
            'modular': False,
        }
        settings.update(kwargs)
        molecular_system = self._get_frame(
            self.trajectory_map[frame], frame, extract_data=True, **kwargs
        )
        if settings['modular'] is True:
            molecular_system.make_modular(rebuild=settings['rebuild'])
            molecules = molecular_system.molecules
        else:
            molecules = {'0': molecular_system.system_to_molecule()}
        results = {}
        for molecule in molecules:
            mol = molecules[molecule]
            if 'molsize' in settings:
                molsize = settings['molsize']
                if isinstance(molsize, int):
                    if mol.no_of_atoms == molsize:
                        results[molecule] = mol.full_analysis(
                            ncpus=ncpus, **kwargs)
                if isinstance(molsize, tuple) and isinstance(molsize[0], str):
                    if molsize[0] in ['bigger', 'greater', 'larger', 'more']:
                        if mol.no_of_atoms > molsize[1]:
                            results[molecule] = mol.full_analysis(
                                ncpus=ncpus, **kwargs)
                    if molsize[0] in ['smaller', 'less']:
                        if mol.no_of_atoms > molsize[1]:
                            results[molecule] = mol.full_analysis(
                                ncpus=ncpus, **kwargs)
                    if molsize[0] in ['not', 'isnot', 'notequal', 'different']:
                        if mol.no_of_atoms != molsize[1]:
                            results[molecule] = mol.full_analysis(
                                ncpus=ncpus, **kwargs)
                    if molsize[0] in ['is', 'equal', 'exactly']:
                        if mol.no_of_atoms == molsize[1]:
                            results[molecule] = mol.full_analysis(
                                ncpus=ncpus, **kwargs)
                    if molsize[0] in ['between', 'inbetween']:
                        if molsize[1] < mol.no_of_atoms < molsize[2]:
                            results[molecule] = mol.full_analysis(
                                ncpus=ncpus, **kwargs)
            else:
                results[molecule] = mol.full_analysis(ncpus=ncpus, **kwargs)
        return results

    def _analysis_parallel_execute(self, frame, **kwargs):
        settings = {
            'rebuild': False,
            'modular': False,
        }
        settings.update(kwargs)
        molecular_system = self._get_frame(
            self.trajectory_map[frame], frame, extract_data=True, **kwargs
        )
        if settings['modular'] is True:
            molecular_system.make_modular(rebuild=settings['rebuild'])
            molecules = molecular_system.molecules
        else:
            molecules = {'0': molecular_system.system_to_molecule()}
        results = {}
        for molecule in molecules:
            mol = molecules[molecule]
            if 'molsize' in settings:
                molsize = settings['molsize']
                if isinstance(molsize, int):
                    if mol.no_of_atoms == molsize:
                        results[molecule] = mol.full_analysis(**kwargs)
                if isinstance(molsize, tuple) and isinstance(molsize[0], str):
                    if molsize[0] in ['bigger', 'greater', 'larger', 'more']:
                        if mol.no_of_atoms > molsize[1]:
                            results[molecule] = mol.full_analysis(**kwargs)
                    if molsize[0] in ['smaller', 'less']:
                        if mol.no_of_atoms > molsize[1]:
                            results[molecule] = mol.full_analysis(**kwargs)
                    if molsize[0] in ['not', 'isnot', 'notequal', 'different']:
                        if mol.no_of_atoms != molsize[1]:
                            results[molecule] = mol.full_analysis(**kwargs)
                    if molsize[0] in ['is', 'equal', 'exactly']:
                        if mol.no_of_atoms == molsize[1]:
                            results[molecule] = mol.full_analysis(**kwargs)
                    if molsize[0] in ['between', 'inbetween']:
                        if molsize[1] < mol.no_of_atoms < molsize[2]:
                            results[molecule] = mol.full_analysis(**kwargs)
            else:
                results[molecule] = mol.full_analysis(**kwargs)
        return frame, results

    def _analysis_parallel(self, frames, ncpus, **kwargs):
        try:
            pool = Pool(processes=ncpus)
            parallel = [
                pool.apply_async(
                    self._analysis_parallel_execute,
                    args=(frame, ),
                    kwds=kwargs) for frame in frames
            ]
            results = [p.get() for p in parallel if p.get()[1] is not None]
            pool.terminate()
            for i in results:
                self.analysis_output[i[0]] = i[1]
        except TypeError:
            pool.terminate()
            raise _ParallelAnalysisError("Parallel analysis failed.")

    def save_analysis(self, filepath=None, **kwargs):
        """
        Dump the content of :attr:`analysis_output` as JSON dictionary.

        Parameters
        ----------
        filepath : :class:`str`
            The filepath for the JSON file.

        Returns
        -------
        None : :class:`NoneType`
        """
        # We pass a copy of the analysis attribute dictionary.
        dict_obj = deepcopy(self.analysis_output)
        # If no filepath is provided we create one.
        if filepath is None:
            filepath = "_".join(
                (str(self.system_id), "pywindow_analysis")
            )
            filepath = '/'.join((os.getcwd(), filepath))
        # Dump the dictionary to json file.
        Output().dump2json(dict_obj, filepath, default=to_list, **kwargs)
        return

    def save_frames(self, frames, filepath=None, filetype='pdb', **kwargs):
        settings = {
            "pdb": Output()._save_pdb,
            "xyz": Output()._save_xyz,
            "decipher": True,
            "forcefield": None,
        }
        settings.update(kwargs)
        if filetype.lower() not in settings.keys():
            raise _FormatError("The '{0}' file format is not supported".format(
                filetype))
        frames_to_get = []
        if isinstance(frames, int):
            frames_to_get.append(frames)
        if isinstance(frames, list):
            frames_to_get = frames
        if isinstance(frames, tuple):
            for frame in range(frames[0], frames[1]):
                frames_to_get.append(frame)
        if isinstance(frames, str):
            if frames in ['all', 'everything']:
                for frame in range(0, self.no_of_frames):
                    frames_to_get.append(frame)
        for frame in frames_to_get:
            if frame not in self.frames.keys():
                _ = self.get_frames(frame)
        # If no filepath is provided we create one.
        if filepath is None:
            filepath = '/'.join((os.getcwd(), str(self.system_id)))
        for frame in frames_to_get:
            frame_molsys = self.frames[frame]
            if settings[
                    'decipher'] is True and settings['forcefield'] is not None:
                if "swap_atoms" in settings.keys():
                    if isinstance(settings["swap_atoms"], dict):
                        frame_molsys.swap_atom_keys(settings["swap_atoms"])
                    else:
                        raise _FunctionError(
                            "The swap_atom_keys function only accepts "
                            "'swap_atoms' argument in form of a dictionary.")
                frame_molsys.decipher_atom_keys(settings["forcefield"])
            ffilepath = '_'.join((filepath, str(frame)))
            if 'elements' not in frame_molsys.system.keys():
                raise _FunctionError(
                    "The frame (MolecularSystem object) needs to have "
                    "'elements' attribute within the system dictionary. "
                    "It is, therefore, neccessary that you set a decipher "
                    "keyword to True. (see manual)")
            settings[filetype.lower()](frame_molsys.system, ffilepath, **
                                       kwargs)
