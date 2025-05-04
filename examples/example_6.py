"""Example 6."""

import logging
import pathlib

import numpy as np

import pywindow as pw

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

known_properties = {
    "PUDXES_no_solvent_rebuild_mol_0": {
        "no_of_atoms": 168,
        "pore_volume_opt": 82.311543851544172,
        "pore_diameter_opt": {
            "diameter": 5.3970201773100221,
            "centre_of_mass": np.array([12.4, 12.4, 12.4]),
            "atom_1": 6,
        },
        "pore_diameter": {"diameter": 5.3970201773100221, "atom": 6},
        "maximum_diameter": {
            "diameter": 22.179369990077188,
            "atom_2": 160,
            "atom_1": 39,
        },
        "centre_of_mass": np.array([12.4, 12.4, 12.4]),
        "pore_volume": 82.311543851544172,
        "windows": {
            "centre_of_mass": np.array(
                [
                    [10.77105707, 10.77097709, 14.02893953],
                    [14.01544862, 14.01541279, 14.01539861],
                    [13.9296554, 10.8702975, 10.87034148],
                    [10.77542236, 14.02453217, 10.77546634],
                ]
            ),
            "diameters": np.array(
                [3.63778745, 3.63562103, 3.62896512, 3.63707237]
            ),
        },
    },
    "PUDXES_no_solvent_rebuild_mol_1": {
        "no_of_atoms": 168,
        "pore_volume_opt": 82.311543851544741,
        "pore_diameter_opt": {
            "diameter": 5.3970201773100346,
            "centre_of_mass": np.array([18.6, 6.2, 6.2]),
            "atom_1": 24,
        },
        "pore_diameter": {"diameter": 5.3970201773100346, "atom": 24},
        "maximum_diameter": {
            "diameter": 22.179369990077188,
            "atom_2": 166,
            "atom_1": 3,
        },
        "centre_of_mass": np.array([18.6, 6.2, 6.2]),
        "pore_volume": 82.311543851544741,
        "windows": {
            "centre_of_mass": np.array(
                [
                    [17.00806474, 7.7918699, 7.79185888],
                    [20.17788435, 4.62211328, 7.77794697],
                    [16.95030126, 4.55022382, 4.55030523],
                    [20.24158072, 7.84159138, 4.55836779],
                ]
            ),
            "diameters": np.array(
                [3.63251205, 3.63115494, 3.64177269, 3.64015484]
            ),
        },
    },
    "PUDXES_no_solvent_rebuild_mol_2": {
        "no_of_atoms": 168,
        "pore_volume_opt": 82.311543851544172,
        "pore_diameter_opt": {
            "diameter": 5.3970201773100221,
            "centre_of_mass": np.array([6.2, 18.6, 6.2]),
            "atom_1": 25,
        },
        "pore_diameter": {"diameter": 5.3970201773100221, "atom": 25},
        "maximum_diameter": {
            "diameter": 22.179369990077188,
            "atom_2": 166,
            "atom_1": 3,
        },
        "centre_of_mass": np.array([6.2, 18.6, 6.2]),
        "pore_volume": 82.311543851544172,
        "windows": {
            "centre_of_mass": np.array(
                [
                    [4.6080649, 20.19186974, 7.79185873],
                    [7.77788435, 17.02211328, 7.77794697],
                    [4.55030111, 16.95022367, 4.5503051],
                    [7.84158072, 20.24159138, 4.55836779],
                ]
            ),
            "diameters": np.array(
                [3.63251205, 3.63115494, 3.64177275, 3.64015484]
            ),
        },
    },
    "PUDXES_no_solvent_rebuild_mol_3": {
        "no_of_atoms": 168,
        "pore_volume_opt": 82.311543851544172,
        "pore_diameter_opt": {
            "diameter": 5.3970201773100221,
            "centre_of_mass": np.array([6.2, 6.2, 18.6]),
            "atom_1": 24,
        },
        "pore_diameter": {"diameter": 5.3970201773100221, "atom": 24},
        "maximum_diameter": {
            "diameter": 22.179369990077188,
            "atom_2": 164,
            "atom_1": 3,
        },
        "centre_of_mass": np.array([6.2, 6.2, 18.6]),
        "pore_volume": 82.311543851544172,
        "windows": {
            "centre_of_mass": np.array(
                [
                    [4.60806477, 7.79186987, 20.19185885],
                    [7.77788435, 4.62211328, 20.17794697],
                    [4.55030126, 4.55022382, 16.95030523],
                    [7.84158072, 7.84159138, 16.95836779],
                ]
            ),
            "diameters": np.array(
                [3.63251205, 3.63115494, 3.64177269, 3.64015484]
            ),
        },
    },
    "PUDXES_no_solvent_rebuild_mol_4": {
        "no_of_atoms": 168,
        "pore_volume_opt": 82.311543851544741,
        "pore_diameter_opt": {
            "diameter": 5.3970201773100346,
            "centre_of_mass": np.array(
                [1.24000000e01, -2.92072975e-17, -2.54322046e-16]
            ),
            "atom_1": 22,
        },
        "pore_diameter": {"diameter": 5.3970201773100346, "atom": 22},
        "maximum_diameter": {
            "diameter": 22.179369990077191,
            "atom_2": 116,
            "atom_1": 104,
        },
        "centre_of_mass": np.array(
            [1.24000000e01, -2.92072975e-17, -2.54322046e-16]
        ),
        "pore_volume": 82.311543851544741,
        "windows": {
            "centre_of_mass": np.array(
                [
                    [10.77105693, -1.62902304, 1.6289397],
                    [14.01544846, 1.6154126, 1.61539845],
                    [13.9296554, -1.5297025, -1.52965852],
                    [10.77542236, 1.62453217, -1.62453366],
                ]
            ),
            "diameters": np.array(
                [3.63778751, 3.63562103, 3.62896512, 3.63707237]
            ),
        },
    },
    "PUDXES_no_solvent_rebuild_mol_5": {
        "no_of_atoms": 168,
        "pore_volume_opt": 82.311543851544741,
        "pore_diameter_opt": {
            "diameter": 5.3970201773100346,
            "centre_of_mass": np.array(
                [-3.56050865e-16, 1.24000000e01, 9.51720783e-17]
            ),
            "atom_1": 22,
        },
        "pore_diameter": {"diameter": 5.3970201773100346, "atom": 22},
        "maximum_diameter": {
            "diameter": 22.179369990077191,
            "atom_2": 122,
            "atom_1": 108,
        },
        "centre_of_mass": np.array(
            [-3.56050865e-16, 1.24000000e01, 9.51720783e-17]
        ),
        "pore_volume": 82.311543851544741,
        "windows": {
            "centre_of_mass": np.array(
                [
                    [-1.62894307, 10.77097696, 1.6289397],
                    [1.61544848, 14.01541263, 1.61539847],
                    [1.52965524, 10.87029766, -1.52965837],
                    [-1.62457764, 14.02453217, -1.62453366],
                ]
            ),
            "diameters": np.array(
                [3.63778751, 3.63562103, 3.62896512, 3.63707237]
            ),
        },
    },
    "PUDXES_no_solvent_rebuild_mol_6": {
        "no_of_atoms": 168,
        "pore_volume_opt": 82.311543851544741,
        "pore_diameter_opt": {
            "diameter": 5.3970201773100346,
            "centre_of_mass": np.array(
                [-6.95411846e-18, -1.52593228e-16, 1.24000000e01]
            ),
            "atom_1": 22,
        },
        "pore_diameter": {"diameter": 5.3970201773100346, "atom": 22},
        "maximum_diameter": {
            "diameter": 22.179369990077191,
            "atom_2": 122,
            "atom_1": 104,
        },
        "centre_of_mass": np.array(
            [-6.95411846e-18, -1.52593228e-16, 1.24000000e01]
        ),
        "pore_volume": 82.311543851544741,
        "windows": {
            "centre_of_mass": np.array(
                [
                    [-1.62894281, -1.6290228, 14.0289394],
                    [1.61544862, 1.61541279, 14.01539861],
                    [1.52965524, -1.52970234, 10.87034163],
                    [-1.62457764, 1.62453217, 10.77546634],
                ]
            ),
            "diameters": np.array(
                [3.63778739, 3.63562103, 3.62896512, 3.63707237]
            ),
        },
    },
    "PUDXES_no_solvent_rebuild_mol_7": {
        "no_of_atoms": 168,
        "pore_volume_opt": 82.311543851544172,
        "pore_diameter_opt": {
            "diameter": 5.3970201773100221,
            "centre_of_mass": np.array([18.6, 18.6, 18.6]),
            "atom_1": 23,
        },
        "pore_diameter": {"diameter": 5.3970201773100221, "atom": 23},
        "maximum_diameter": {
            "diameter": 22.179369990077191,
            "atom_2": 166,
            "atom_1": 9,
        },
        "centre_of_mass": np.array([18.6, 18.6, 18.6]),
        "pore_volume": 82.311543851544172,
        "windows": {
            "centre_of_mass": np.array(
                [
                    [17.0080649, 20.19186974, 20.19185873],
                    [20.17788435, 17.02211328, 20.17794697],
                    [16.95030126, 16.95022382, 16.95030523],
                    [20.24158072, 20.24159138, 16.95836779],
                ]
            ),
            "diameters": np.array(
                [3.63251205, 3.63115494, 3.64177269, 3.64015484]
            ),
        },
    },
    "EPIRUR_no_solvent_rebuild_mol_0": {
        "no_of_atoms": 132,
        "pore_volume_opt": 77.956489351993511,
        "pore_diameter_opt": {
            "diameter": 5.3001059373333721,
            "centre_of_mass": np.array(
                [-3.93106222e-05, 9.35731005e00, 1.56504978e01]
            ),
            "atom_1": 96,
        },
        "pore_diameter": {"diameter": 5.2999265295219633, "atom": 96},
        "maximum_diameter": {
            "diameter": 16.043905765919597,
            "atom_2": 130,
            "atom_1": 23,
        },
        "centre_of_mass": np.array(
            [9.40462517e-06, 9.35727968e00, 1.56504250e01]
        ),
        "pore_volume": 77.948573172645609,
        "windows": {
            "centre_of_mass": np.array(
                [
                    [-3.57508502, 6.58799807, 17.22902676],
                    [-0.64890246, 15.22138439, 18.47068556],
                    [4.25016141, 7.61391107, 17.28768364],
                    [5.42756096, 11.73622114, 12.80250252],
                    [-0.26913822, 5.94455552, 14.69016071],
                    [-2.80582149, 11.2716919, 14.70103931],
                ]
            ),
            "diameters": np.array(
                [
                    2.00509093,
                    3.29616929,
                    2.00663377,
                    3.3370681,
                    2.30719175,
                    2.32221521,
                ]
            ),
        },
    },
    "EPIRUR_no_solvent_rebuild_mol_1": {
        "no_of_atoms": 132,
        "pore_volume_opt": 77.95863474639259,
        "pore_diameter_opt": {
            "diameter": 5.3001545572467741,
            "centre_of_mass": np.array([8.10374623, 4.67878724, 7.82557001]),
            "atom_1": 89,
        },
        "pore_diameter": {"diameter": 5.2993422655565112, "atom": 89},
        "maximum_diameter": {
            "diameter": 16.043058073792327,
            "atom_2": 71,
            "atom_1": 57,
        },
        "centre_of_mass": np.array([8.10361663, 4.67865262, 7.8252094]),
        "pore_volume": 77.922796859455104,
        "windows": {
            "centre_of_mass": np.array(
                [
                    [4.52805649, 1.90916572, 9.4038027],
                    [7.45449451, 10.54360135, 10.64604147],
                    [12.35378246, 2.93530751, 9.460373],
                    [13.53068823, 7.05777652, 4.97722858],
                    [7.83446287, 1.2655924, 6.86523099],
                    [5.29799464, 6.5932151, 6.87567761],
                ]
            ),
            "diameters": np.array(
                [
                    2.00514587,
                    3.29658077,
                    2.0076291,
                    3.33669202,
                    2.30676811,
                    2.32216627,
                ]
            ),
        },
    },
    "EPIRUR_no_solvent_rebuild_mol_2": {
        "no_of_atoms": 132,
        "pore_volume_opt": 77.981955217293134,
        "pore_diameter_opt": {
            "diameter": 5.3006829989339703,
            "centre_of_mass": np.array(
                [1.10603914e-04, 3.00322655e-05, 2.34759699e01]
            ),
            "atom_1": 5,
        },
        "pore_diameter": {"diameter": 5.3002853308997366, "atom": 5},
        "maximum_diameter": {
            "diameter": 16.042925268499431,
            "atom_2": 116,
            "atom_1": 52,
        },
        "centre_of_mass": np.array(
            [8.33333333e-05, 1.18356318e-04, 2.34757938e01]
        ),
        "pore_volume": 77.964405438671278,
        "windows": {
            "centre_of_mass": np.array(
                [
                    [-3.57555297, -2.76906725, 25.05544927],
                    [-0.64894239, 5.86476986, 26.29610162],
                    [4.24985196, -1.74348104, 25.11251292],
                    [5.42724047, 2.37933973, 20.62637274],
                    [-0.26928676, -3.41227298, 22.51588972],
                    [-2.80537935, 1.91557804, 22.52649144],
                ]
            ),
            "diameters": np.array(
                [
                    2.0044412,
                    3.29680913,
                    2.00736204,
                    3.33799379,
                    2.30763494,
                    2.32165226,
                ]
            ),
        },
    },
    "TATVER_no_solvent_rebuild_mol_0": {
        "no_of_atoms": 244,
        "pore_volume_opt": 477.5395402757328,
        "pore_diameter_opt": {
            "diameter": 9.6977333431968784,
            "centre_of_mass": np.array([9.61568815, 15.38945731, 18.356231]),
            "atom_1": 175,
        },
        "pore_diameter": {"diameter": 9.5618906544273035, "atom": 71},
        "maximum_diameter": {
            "diameter": 29.718709420567329,
            "atom_2": 243,
            "atom_1": 13,
        },
        "centre_of_mass": np.array([9.8069714, 15.21626983, 18.38254609]),
        "pore_volume": 457.75167408412602,
        "windows": {
            "centre_of_mass": np.array(
                [
                    [8.64613173, 13.99368092, 20.82749396],
                    [15.27623749, 14.03211251, 18.78371326],
                    [7.93359245, 20.06576608, 18.90305695],
                    [8.48616679, 14.27853608, 14.72280858],
                ]
            ),
            "diameters": np.array(
                [8.49367369, 9.09581185, 8.81853449, 7.75061815]
            ),
        },
    },
    "TATVER_no_solvent_rebuild_mol_1": {
        "no_of_atoms": 244,
        "pore_volume_opt": 477.54078784832939,
        "pore_diameter_opt": {
            "diameter": 9.6977417883026842,
            "centre_of_mass": np.array([19.17126673, 5.68399392, 3.24614256]),
            "atom_1": 78,
        },
        "pore_diameter": {"diameter": 9.5617853509221113, "atom": 78},
        "maximum_diameter": {
            "diameter": 29.71742024614931,
            "atom_2": 243,
            "atom_1": 14,
        },
        "centre_of_mass": np.array([19.09548327, 5.7537699, 3.24490551]),
        "pore_volume": 457.73655082241476,
        "windows": {
            "centre_of_mass": np.array(
                [
                    [20.57635233, 6.79058955, 7.2579067],
                    [15.64167248, 6.58681621, 3.26832179],
                    [20.39831277, 3.0476932, 3.26010817],
                    [20.26530191, 6.98860404, 0.78049266],
                ]
            ),
            "diameters": np.array(
                [7.81671522, 8.52499906, 8.76121118, 8.49314974]
            ),
        },
    },
}


def main() -> None:
    """Run script."""
    script_directory = pathlib.Path(__file__).parent.resolve()

    data_directory = script_directory / "data"
    input_directory = data_directory / "input"
    output_directory = data_directory / "output"

    input_files = [
        input_directory / "PUDXES_no_solvent.pdb",
        input_directory / "EPIRUR_no_solvent.pdb",
        input_directory / "TATVER_no_solvent.pdb",
    ]
    for input_file in input_files:
        name = input_file.name.split(".")[0]

        molsys = pw.MolecularSystem.load_file(input_file)
        rebuild_molsys = molsys.rebuild_system()
        rebuild_molsys.dump_system(
            output_directory / f"{name}_rebuild.pdb",
            override=True,
        )
        rebuild_molsys.make_modular()

        for molecule, mol in rebuild_molsys.molecules.items():
            logger.info(
                "Analysing molecule %s out of %s of %s",
                molecule,
                len(rebuild_molsys.molecules),
                name,
            )
            mol.full_analysis()
            logger.info(
                "pore size: %s",
                mol.properties["pore_diameter"]["diameter"],  # type: ignore[index]
            )

            (same_dict, failed_prop) = pw.compare_properties_dict(
                dict1=mol.properties,  # type:ignore[arg-type]
                dict2=known_properties[f"{name}_rebuild_mol_{molecule}"],  # type:ignore[arg-type]
            )
            # Each molecule can be saved separately
            mol.dump_molecule(
                output_directory / f"{name}_rebuild_mol_{molecule}.pdb",
                include_coms=True,
                override=True,
            )


if __name__ == "__main__":
    main()
